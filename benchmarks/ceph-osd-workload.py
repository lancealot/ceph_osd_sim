#!/usr/bin/env python3

"""Workload driver for Ceph OSD validation.

Wraps rados bench with consistent invocation, captures ceph daemon perf dump
snapshots at measurement start/end, and emits structured workload metadata.

Designed to run from a bench client host. Stdlib only.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time


def run_cmd(cmd, timeout=30, check=True):
    """Run a command and return stdout. Raises on failure if check=True."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd,
                                            result.stdout, result.stderr)
    return result.stdout


def get_osd_ids():
    """Get list of OSD IDs in the cluster."""
    out = run_cmd(['ceph', 'osd', 'ls', '--format=json'])
    return json.loads(out)


def get_perf_dump(osd_id):
    """Capture ceph daemon perf dump for one OSD."""
    try:
        out = run_cmd(['ceph', 'daemon', f'osd.{osd_id}', 'perf', 'dump'],
                      timeout=10, check=False)
        return json.loads(out)
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def get_perf_dumps(osd_ids):
    """Capture perf dumps for all specified OSDs."""
    dumps = {}
    for osd_id in osd_ids:
        dump = get_perf_dump(osd_id)
        if dump is not None:
            dumps[str(osd_id)] = dump
    return dumps


def get_ceph_config():
    """Capture ceph config dump."""
    try:
        out = run_cmd(['ceph', 'config', 'dump', '--format=json'], timeout=15)
        return json.loads(out)
    except (json.JSONDecodeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return None


def get_pool_info(pool_name):
    """Get pool details (pg_num, size, etc.)."""
    try:
        out = run_cmd(['ceph', 'osd', 'pool', 'get', pool_name, 'all',
                        '--format=json'], timeout=10)
        return json.loads(out)
    except (json.JSONDecodeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return None


def create_pool(pool_name, pg_num=128, pool_type='replicated',
                size=3, ec_profile=None):
    """Create a fresh pool for the validation run."""
    if pool_type == 'erasure':
        if ec_profile:
            run_cmd(['ceph', 'osd', 'pool', 'create', pool_name,
                      str(pg_num), 'erasure', ec_profile])
        else:
            run_cmd(['ceph', 'osd', 'pool', 'create', pool_name,
                      str(pg_num), 'erasure'])
    else:
        run_cmd(['ceph', 'osd', 'pool', 'create', pool_name, str(pg_num)])
        run_cmd(['ceph', 'osd', 'pool', 'set', pool_name, 'size', str(size)])

    run_cmd(['ceph', 'osd', 'pool', 'application', 'enable',
              pool_name, 'rados'], check=False)
    print(f"Created pool '{pool_name}' (pg_num={pg_num}, type={pool_type})")


def delete_pool(pool_name):
    """Delete a pool."""
    try:
        run_cmd(['ceph', 'osd', 'pool', 'delete', pool_name, pool_name,
                  '--yes-i-really-really-mean-it'], timeout=30)
        print(f"Deleted pool '{pool_name}'")
    except subprocess.CalledProcessError:
        print(f"Warning: could not delete pool '{pool_name}'", file=sys.stderr)


def run_rados_bench(pool_name, mode, duration, obj_size, threads,
                    no_cleanup=True):
    """Run rados bench and capture output."""
    cmd = ['rados', 'bench', '-p', pool_name, str(duration), mode,
           '-b', str(obj_size), '-t', str(threads)]
    if no_cleanup and mode == 'write':
        cmd.append('--no-cleanup')
    if mode in ('seq', 'rand'):
        cmd.append('--no-cleanup')

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=duration + 120)
    end_time = time.time()

    return {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode,
        'start_time': start_time,
        'end_time': end_time,
    }


def parse_rados_bench_output(output):
    """Parse rados bench stdout into structured data."""
    lines = output.strip().split('\n')
    summary = {}
    samples = []

    for line in lines:
        # Per-second lines look like: "  sec Cur ops   started  ..."
        # Data lines look like: "   1       16        48  ..."
        if re.match(r'^\s+\d+\s+\d+', line):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    samples.append({
                        'sec': int(parts[0]),
                        'cur_ops': int(parts[1]),
                        'started': int(parts[2]),
                        'finished': int(parts[3]),
                        'avg_mbps': float(parts[4]),
                        'cur_mbps': float(parts[5]),
                    })
                except (ValueError, IndexError):
                    pass

        # Summary lines
        for key in ['Total time run:', 'Total writes made:',
                    'Total reads made:', 'Write size:',
                    'Object size:', 'Bandwidth (MB/sec):',
                    'Stddev Bandwidth:', 'Max bandwidth (MB/sec):',
                    'Min bandwidth (MB/sec):',
                    'Average IOPS:', 'Stddev IOPS:',
                    'Max IOPS:', 'Min IOPS:',
                    'Average Latency(s):', 'Stddev Latency(s):',
                    'Max latency(s):', 'Min latency(s):']:
            if key in line:
                val = line.split(key)[-1].strip()
                summary_key = (key.rstrip(':').lower()
                               .replace(' ', '_').replace('(', '_')
                               .replace(')', '').replace('/', '_per_'))
                try:
                    summary[summary_key] = float(val)
                except ValueError:
                    summary[summary_key] = val

    return summary, samples


def compute_perf_dump_deltas(start_dumps, end_dumps):
    """Compute deltas between start and end perf dump snapshots."""
    deltas = {}
    for osd_id in end_dumps:
        if osd_id not in start_dumps:
            continue
        start = start_dumps[osd_id]
        end = end_dumps[osd_id]
        osd_delta = {}
        for section in end:
            if section not in start:
                continue
            if not isinstance(end[section], dict):
                continue
            sec_delta = {}
            for key in end[section]:
                if not isinstance(end[section][key], (int, float)):
                    continue
                if key not in start[section]:
                    continue
                d = end[section][key] - start[section][key]
                if d != 0:
                    sec_delta[key] = d
            if sec_delta:
                osd_delta[section] = sec_delta
        if osd_delta:
            deltas[osd_id] = osd_delta
    return deltas


def main():
    parser = argparse.ArgumentParser(
        description='Drive rados bench workload for OSD validation')
    parser.add_argument('--pool', default='',
                        help='Pool name (default: auto-generated)')
    parser.add_argument('--object-size', default='4m',
                        help='Object size (e.g., 4k, 128k, 4m; default: 4m)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Measurement duration in seconds (default: 300)')
    parser.add_argument('--warmup', type=int, default=120,
                        help='Warmup duration in seconds (default: 120)')
    parser.add_argument('--threads', type=int, default=16,
                        help='Number of concurrent threads (default: 16)')
    parser.add_argument('--mode', choices=['write', 'seq', 'rand'],
                        default='write',
                        help='Benchmark mode (default: write)')
    parser.add_argument('--rw-ratio', type=float, default=0.0,
                        help='Read ratio 0.0-1.0. 0=write-only, 1=read-only. '
                             'For mixed: runs write then read phase.')
    parser.add_argument('--pg-num', type=int, default=128,
                        help='PG count for auto-created pool (default: 128)')
    parser.add_argument('--pool-type', choices=['replicated', 'erasure'],
                        default='replicated',
                        help='Pool type (default: replicated)')
    parser.add_argument('--pool-size', type=int, default=3,
                        help='Replica count for replicated pools (default: 3)')
    parser.add_argument('--ec-profile', default='',
                        help='EC profile name (for erasure pools)')
    parser.add_argument('--compress', default='',
                        help='Compression algorithm (e.g., zstd, lz4)')
    parser.add_argument('--run-id', default='',
                        help='Run identifier for joining with CPU data')
    parser.add_argument('--output-dir', default='.',
                        help='Directory for output files')
    parser.add_argument('--keep-pool', action='store_true',
                        help='Do not delete pool after benchmark')
    parser.add_argument('--osd-ids', default='',
                        help='Comma-separated OSD IDs for perf dumps '
                             '(default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    size_map = {
        '4k': 4096, '8k': 8192, '16k': 16384, '32k': 32768,
        '64k': 65536, '128k': 131072, '256k': 262144, '512k': 524288,
        '1m': 1048576, '4m': 4194304, '16m': 16777216,
    }
    obj_size_str = args.object_size.lower()
    obj_size = size_map.get(obj_size_str)
    if obj_size is None:
        try:
            obj_size = int(obj_size_str)
        except ValueError:
            print(f"Error: Unknown object size '{args.object_size}'",
                  file=sys.stderr)
            sys.exit(1)

    pool_name = args.pool or f'val_{args.run_id or "bench"}_{int(time.time())}'
    auto_pool = not args.pool

    if args.osd_ids:
        osd_ids = [int(x.strip()) for x in args.osd_ids.split(',')]
    else:
        osd_ids = get_osd_ids()

    if auto_pool:
        create_pool(pool_name, args.pg_num, args.pool_type,
                     args.pool_size, args.ec_profile or None)

    if args.compress:
        run_cmd(['ceph', 'osd', 'pool', 'set', pool_name,
                  'compression_algorithm', args.compress])
        run_cmd(['ceph', 'osd', 'pool', 'set', pool_name,
                  'compression_mode', 'force'])
        print(f"Enabled compression: {args.compress} (force mode)")

    pool_info = get_pool_info(pool_name)
    ceph_config = get_ceph_config()

    # Run warmup phase
    phases = []
    measurement_start = None
    measurement_end = None

    try:
        if args.warmup > 0:
            print(f"\n--- Warmup phase ({args.warmup}s, results discarded) ---")
            warmup_result = run_rados_bench(pool_name, 'write', args.warmup,
                                            obj_size, args.threads)
            if warmup_result['returncode'] != 0:
                print(f"Warning: warmup failed: {warmup_result['stderr']}",
                      file=sys.stderr)

        # Capture perf dumps at measurement start
        print(f"\nCapturing perf dump snapshots (start)...")
        start_dumps = get_perf_dumps(osd_ids)

        # Determine phases based on rw_ratio
        if args.rw_ratio >= 1.0:
            phases = [('rand', args.duration)]
        elif args.rw_ratio <= 0.0:
            phases = [('write', args.duration)]
        else:
            write_dur = int(args.duration * (1.0 - args.rw_ratio))
            read_dur = args.duration - write_dur
            phases = [('write', write_dur), ('rand', read_dur)]

        all_results = []
        for mode, dur in phases:
            print(f"\n--- Measurement phase: {mode} ({dur}s) ---")
            result = run_rados_bench(pool_name, mode, dur, obj_size,
                                     args.threads)
            summary, samples = parse_rados_bench_output(result['stdout'])
            all_results.append({
                'mode': mode,
                'duration': dur,
                'summary': summary,
                'samples': samples,
                'start_time': result['start_time'],
                'end_time': result['end_time'],
                'returncode': result['returncode'],
            })
            if result['returncode'] != 0:
                print(f"Warning: rados bench {mode} exited with "
                      f"{result['returncode']}", file=sys.stderr)

        if all_results:
            measurement_start = all_results[0]['start_time']
            measurement_end = all_results[-1]['end_time']

        # Capture perf dumps at measurement end
        print(f"\nCapturing perf dump snapshots (end)...")
        end_dumps = get_perf_dumps(osd_ids)
        perf_deltas = compute_perf_dump_deltas(start_dumps, end_dumps)

    finally:
        if auto_pool and not args.keep_pool:
            delete_pool(pool_name)

    # Write workload CSV (per-second samples from each phase)
    csv_path = os.path.join(args.output_dir, 'workload.csv')
    csv_fields = ['timestamp', 'run_id', 'phase', 'sec', 'cur_ops',
                  'started', 'finished', 'avg_mbps', 'cur_mbps']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for result in all_results:
            base_time = result['start_time']
            for s in result['samples']:
                writer.writerow({
                    'timestamp': f'{base_time + s["sec"]:.3f}',
                    'run_id': args.run_id,
                    'phase': result['mode'],
                    'sec': s['sec'],
                    'cur_ops': s['cur_ops'],
                    'started': s['started'],
                    'finished': s['finished'],
                    'avg_mbps': f'{s["avg_mbps"]:.2f}',
                    'cur_mbps': f'{s["cur_mbps"]:.2f}',
                })
    print(f"\nWorkload CSV written to {csv_path}")

    # Write workload metadata
    meta = {
        'run_id': args.run_id,
        'pool_name': pool_name,
        'pool_type': args.pool_type,
        'pool_size': args.pool_size,
        'ec_profile': args.ec_profile,
        'pg_num': args.pg_num,
        'compression': args.compress or 'none',
        'object_size': obj_size,
        'object_size_str': args.object_size,
        'threads': args.threads,
        'warmup_duration': args.warmup,
        'measurement_duration': args.duration,
        'rw_ratio': args.rw_ratio,
        'measurement_window': {
            'start': measurement_start,
            'end': measurement_end,
        },
        'phases': [
            {
                'mode': r['mode'],
                'duration': r['duration'],
                'summary': r['summary'],
                'start_time': r['start_time'],
                'end_time': r['end_time'],
            }
            for r in all_results
        ],
        'perf_dump_deltas': perf_deltas,
        'pool_info': pool_info,
        'ceph_config': ceph_config,
    }

    meta_path = os.path.join(args.output_dir, 'workload_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Workload metadata written to {meta_path}")

    # Print summary
    for r in all_results:
        s = r['summary']
        print(f"\n--- {r['mode']} summary ---")
        for key in sorted(s.keys()):
            print(f"  {key}: {s[key]}")


if __name__ == '__main__':
    main()
