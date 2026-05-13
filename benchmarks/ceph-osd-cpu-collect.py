#!/usr/bin/env python3

"""Per-thread CPU collector for Ceph OSD processes.

Samples /proc/<pid>/task/<tid>/stat for each ceph-osd process, bucketizes
threads by name (op-pipeline, bluestore-kv, rocksdb-compaction, messenger,
other), and emits per-OSD per-bucket CPU usage as CSV.

Designed to run on each Ceph node during a validation workload. Stdlib only.
"""

import argparse
import csv
import json
import glob
import os
import re
import signal
import sys
import time

CLK_TCK = os.sysconf('SC_CLK_TCK')

THREAD_BUCKETS = {
    'op_pipeline': re.compile(r'^tp_osd_tp|^osd_op_tp|^osd_srv_tp'),
    'bluestore_kv': re.compile(r'^bstore_kv'),
    'rocksdb_compact': re.compile(r'^rocksdb:'),
    'messenger': re.compile(r'^ms_dispatch|^ms_async|^msgr-worker'),
    'finisher': re.compile(r'^fn_'),
    'recovery': re.compile(r'^tp_osd_recov|^osd_recov'),
}
DEFAULT_BUCKET = 'other'


def classify_thread(comm):
    for bucket, pattern in THREAD_BUCKETS.items():
        if pattern.search(comm):
            return bucket
    return DEFAULT_BUCKET


def find_osd_pids():
    """Discover ceph-osd PIDs and their OSD IDs."""
    osds = {}
    for entry in glob.glob('/proc/[0-9]*/comm'):
        try:
            pid = int(entry.split('/')[2])
            with open(entry) as f:
                comm = f.read().strip()
            if comm != 'ceph-osd':
                continue
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().decode('utf-8', errors='replace')
            args = cmdline.split('\x00')
            osd_id = None
            for i, arg in enumerate(args):
                if arg == '--id' and i + 1 < len(args):
                    osd_id = args[i + 1]
                    break
                if arg == '-i' and i + 1 < len(args):
                    osd_id = args[i + 1]
                    break
            if osd_id is None:
                osd_id = str(pid)
            osds[pid] = osd_id
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return osds


def read_thread_stats(pid):
    """Read per-thread utime/stime and comm for all threads of a PID."""
    threads = []
    task_dir = f'/proc/{pid}/task'
    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError:
        return threads
    for tid in tids:
        try:
            with open(f'{task_dir}/{tid}/comm') as f:
                comm = f.read().strip()
            with open(f'{task_dir}/{tid}/stat') as f:
                stat_line = f.read()
            # /proc/<pid>/stat format: pid (comm) state ... field14=utime field15=stime
            # comm can contain spaces and parens, so find the last ')'
            close_paren = stat_line.rfind(')')
            fields = stat_line[close_paren + 2:].split()
            utime = int(fields[11])  # field 14 (0-indexed from after comm: index 11)
            stime = int(fields[12])  # field 15
            threads.append({
                'tid': int(tid),
                'comm': comm,
                'bucket': classify_thread(comm),
                'utime': utime,
                'stime': stime,
            })
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return threads


def read_process_ctx_switches(pid):
    """Read voluntary and nonvoluntary context switches from /proc/<pid>/status."""
    vol, nonvol = 0, 0
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('voluntary_ctxt_switches:'):
                    vol = int(line.split(':')[1].strip())
                elif line.startswith('nonvoluntary_ctxt_switches:'):
                    nonvol = int(line.split(':')[1].strip())
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return vol, nonvol


def read_schedstat(pid):
    """Read total run time and runqueue wait from /proc/<pid>/schedstat."""
    try:
        with open(f'/proc/{pid}/schedstat') as f:
            parts = f.read().split()
            return int(parts[0]), int(parts[1])  # run_ns, wait_ns
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return 0, 0


def read_system_softirq():
    """Read system-wide softirq ticks from /proc/stat."""
    try:
        with open('/proc/stat') as f:
            line = f.readline()  # cpu  user nice system idle iowait irq softirq ...
            fields = line.split()
            return int(fields[7])  # softirq is field 8 (0-indexed: 7)
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return 0


def collect_cpu_metadata():
    """Collect per-run hardware/topology metadata."""
    meta = {
        'hostname': os.uname().nodename,
        'clk_tck': CLK_TCK,
        'cpu_count': os.cpu_count(),
        'cpu_model': '',
        'cpu_freqs': [],
        'cpu_governor': '',
        'numa_topology': '',
    }

    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    meta['cpu_model'] = line.split(':', 1)[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass

    for cpu_dir in sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq')):
        freq_file = os.path.join(cpu_dir, 'scaling_cur_freq')
        try:
            with open(freq_file) as f:
                meta['cpu_freqs'].append(int(f.read().strip()))
        except (FileNotFoundError, PermissionError, ValueError):
            pass

    gov_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
    try:
        with open(gov_file) as f:
            meta['cpu_governor'] = f.read().strip()
    except (FileNotFoundError, PermissionError):
        pass

    try:
        import subprocess
        result = subprocess.run(['numactl', '--hardware'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            meta['numa_topology'] = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return meta


def aggregate_by_bucket(threads):
    """Aggregate thread stats by bucket. Returns {bucket: {utime, stime, count}}."""
    buckets = {}
    for t in threads:
        b = t['bucket']
        if b not in buckets:
            buckets[b] = {'utime': 0, 'stime': 0, 'count': 0}
        buckets[b]['utime'] += t['utime']
        buckets[b]['stime'] += t['stime']
        buckets[b]['count'] += 1
    return buckets


def compute_deltas(prev_buckets, curr_buckets, interval):
    """Compute per-bucket CPU microseconds from tick deltas."""
    deltas = {}
    all_bucket_names = set(list(prev_buckets.keys()) + list(curr_buckets.keys()))
    for b in all_bucket_names:
        prev = prev_buckets.get(b, {'utime': 0, 'stime': 0, 'count': 0})
        curr = curr_buckets.get(b, {'utime': 0, 'stime': 0, 'count': 0})
        du = (curr['utime'] - prev['utime']) / CLK_TCK * 1_000_000
        ds = (curr['stime'] - prev['stime']) / CLK_TCK * 1_000_000
        deltas[b] = {
            'cpu_us_user': max(0.0, du),
            'cpu_us_system': max(0.0, ds),
            'cpu_us_total': max(0.0, du + ds),
            'thread_count': curr['count'],
        }
    return deltas


def main():
    parser = argparse.ArgumentParser(
        description='Collect per-thread CPU usage for ceph-osd processes')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--duration', type=float, default=0,
                        help='Collection duration in seconds (0 = until interrupted)')
    parser.add_argument('--run-id', default='',
                        help='Run identifier for joining with workload data')
    parser.add_argument('--output-dir', default='.',
                        help='Directory for output files')
    parser.add_argument('--meta-only', action='store_true',
                        help='Collect metadata only, then exit')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    meta = collect_cpu_metadata()
    meta['run_id'] = args.run_id
    meta['interval'] = args.interval

    osd_pids = find_osd_pids()
    if not osd_pids:
        print("Error: No ceph-osd processes found.", file=sys.stderr)
        sys.exit(1)

    meta['osd_pids'] = {str(pid): osd_id for pid, osd_id in osd_pids.items()}
    print(f"Found {len(osd_pids)} OSD(s): "
          f"{', '.join(f'osd.{v} (pid {k})' for k, v in sorted(osd_pids.items(), key=lambda x: x[1]))}")

    osd_cpusets = {}
    for pid in osd_pids:
        try:
            with open(f'/proc/{pid}/cpuset') as f:
                osd_cpusets[str(pid)] = f.read().strip()
        except (FileNotFoundError, PermissionError):
            pass
    if osd_cpusets:
        meta['osd_cpusets'] = osd_cpusets

    meta_path = os.path.join(args.output_dir, 'cpu_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {meta_path}")

    if args.meta_only:
        return

    csv_path = os.path.join(args.output_dir, 'cpu.csv')
    csv_fields = [
        'timestamp', 'run_id', 'osd_id', 'pid', 'bucket',
        'cpu_us_user', 'cpu_us_system', 'cpu_us_total',
        'thread_count', 'vol_ctx_switches', 'nonvol_ctx_switches',
        'schedstat_run_ns', 'schedstat_wait_ns', 'softirq_ticks',
    ]

    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    prev_state = {}
    prev_ctx = {}
    prev_sched = {}
    prev_softirq = 0
    start_time = time.time()

    print(f"Collecting CPU data (interval={args.interval}s, "
          f"duration={'unlimited' if args.duration <= 0 else f'{args.duration}s'})...")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        sample_count = 0
        while not stop:
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

            now = time.time()
            curr_softirq = read_system_softirq()

            for pid, osd_id in list(osd_pids.items()):
                if not os.path.exists(f'/proc/{pid}'):
                    print(f"OSD pid {pid} (osd.{osd_id}) gone, removing.",
                          file=sys.stderr)
                    del osd_pids[pid]
                    continue

                threads = read_thread_stats(pid)
                curr_buckets = aggregate_by_bucket(threads)
                vol, nonvol = read_process_ctx_switches(pid)
                run_ns, wait_ns = read_schedstat(pid)

                if pid in prev_state:
                    deltas = compute_deltas(prev_state[pid], curr_buckets,
                                            args.interval)
                    prev_v, prev_nv = prev_ctx.get(pid, (0, 0))
                    prev_r, prev_w = prev_sched.get(pid, (0, 0))
                    d_softirq = curr_softirq - prev_softirq

                    for bucket, d in deltas.items():
                        writer.writerow({
                            'timestamp': f'{now:.3f}',
                            'run_id': args.run_id,
                            'osd_id': osd_id,
                            'pid': pid,
                            'bucket': bucket,
                            'cpu_us_user': f'{d["cpu_us_user"]:.1f}',
                            'cpu_us_system': f'{d["cpu_us_system"]:.1f}',
                            'cpu_us_total': f'{d["cpu_us_total"]:.1f}',
                            'thread_count': d['thread_count'],
                            'vol_ctx_switches': vol - prev_v,
                            'nonvol_ctx_switches': nonvol - prev_nv,
                            'schedstat_run_ns': run_ns - prev_r,
                            'schedstat_wait_ns': wait_ns - prev_w,
                            'softirq_ticks': d_softirq,
                        })

                prev_state[pid] = curr_buckets
                prev_ctx[pid] = (vol, nonvol)
                prev_sched[pid] = (run_ns, wait_ns)

            prev_softirq = curr_softirq
            sample_count += 1

            elapsed = time.time() - now
            sleep_time = max(0, args.interval - elapsed)
            if sleep_time > 0 and not stop:
                time.sleep(sleep_time)

    print(f"Collection complete. {sample_count} samples written to {csv_path}")


if __name__ == '__main__':
    main()
