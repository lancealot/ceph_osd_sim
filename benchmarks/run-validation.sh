#!/bin/bash
set -euo pipefail

# run-validation.sh — Orchestrates a validation run across cluster nodes.
#
# Starts CPU collectors on each OSD node (via ssh), runs the workload driver
# from the bench client, retrieves data, and bundles everything into a
# validation directory with a finalized measured.json summary.
#
# Usage:
#   ./run-validation.sh --profile <name> --nodes node1,node2 [options]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
PROFILE=""
NODES=""
OSD_IDS=""
POOL_TYPE="replicated"
POOL_SIZE=3
EC_PROFILE=""
OBJECT_SIZE="4m"
DURATION=300
WARMUP=120
THREADS=16
RW_RATIO=0.0
PG_NUM=128
COMPRESS=""
OUTPUT_BASE="$REPO_DIR/validation"
COLLECTOR_INTERVAL=1.0
KEEP_POOL=0
SSH_USER=""
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
COLLECTOR_PATH=""

usage() {
    cat <<'EOF'
Usage: run-validation.sh --profile NAME --nodes HOST1,HOST2,... [options]

Required:
  --profile NAME        Profile name (used in output directory)
  --nodes HOST1,...     Comma-separated OSD node hostnames

Cluster config:
  --pool-type TYPE      replicated or erasure (default: replicated)
  --pool-size N         Replica count (default: 3)
  --ec-profile NAME     EC profile name (for erasure pools)
  --pg-num N            PG count (default: 128)
  --compress ALGO       Compression algorithm (e.g., zstd, lz4)

Workload config:
  --object-size SIZE    Object size (default: 4m)
  --duration SECS       Measurement duration (default: 300)
  --warmup SECS         Warmup duration (default: 120)
  --threads N           Concurrent threads (default: 16)
  --rw-ratio RATIO      Read ratio 0.0-1.0 (default: 0.0 = write-only)

Collection config:
  --interval SECS       CPU sampling interval (default: 1.0)
  --osd-ids IDS         Comma-separated OSD IDs for perf dumps

Output:
  --output-base DIR     Base output directory (default: <repo>/validation)
  --keep-pool           Don't delete the benchmark pool

SSH:
  --ssh-user USER       SSH username (default: current user)
  --collector-path PATH Path to ceph-osd-cpu-collect.py on remote nodes
                        (default: copy from local benchmarks/)
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)      PROFILE="$2"; shift 2;;
        --nodes)        NODES="$2"; shift 2;;
        --pool-type)    POOL_TYPE="$2"; shift 2;;
        --pool-size)    POOL_SIZE="$2"; shift 2;;
        --ec-profile)   EC_PROFILE="$2"; shift 2;;
        --pg-num)       PG_NUM="$2"; shift 2;;
        --compress)     COMPRESS="$2"; shift 2;;
        --object-size)  OBJECT_SIZE="$2"; shift 2;;
        --duration)     DURATION="$2"; shift 2;;
        --warmup)       WARMUP="$2"; shift 2;;
        --threads)      THREADS="$2"; shift 2;;
        --rw-ratio)     RW_RATIO="$2"; shift 2;;
        --interval)     COLLECTOR_INTERVAL="$2"; shift 2;;
        --osd-ids)      OSD_IDS="$2"; shift 2;;
        --output-base)  OUTPUT_BASE="$2"; shift 2;;
        --keep-pool)    KEEP_POOL=1; shift;;
        --ssh-user)     SSH_USER="$2"; shift 2;;
        --collector-path) COLLECTOR_PATH="$2"; shift 2;;
        --help|-h)      usage;;
        *)              echo "Unknown option: $1"; usage;;
    esac
done

if [[ -z "$PROFILE" ]] || [[ -z "$NODES" ]]; then
    echo "Error: --profile and --nodes are required"
    usage
fi

# Generate run ID and output directory
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
GIT_SHA=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RUN_ID="${TIMESTAMP}_${PROFILE}_${GIT_SHA}"
RUN_DIR="${OUTPUT_BASE}/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "============================================"
echo "  Validation Run: $RUN_ID"
echo "============================================"
echo "Profile:     $PROFILE"
echo "Nodes:       $NODES"
echo "Output:      $RUN_DIR"
echo ""

IFS=',' read -ra NODE_ARRAY <<< "$NODES"

SSH_PREFIX=""
if [[ -n "$SSH_USER" ]]; then
    SSH_PREFIX="${SSH_USER}@"
fi

# Preflight checks
echo "--- Preflight checks ---"

echo "Checking cluster health..."
HEALTH=$(ceph health --format=json 2>/dev/null || echo '{"status":"UNKNOWN"}')
HEALTH_STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
echo "Cluster health: $HEALTH_STATUS"
if [[ "$HEALTH_STATUS" == "UNKNOWN" ]]; then
    echo "Warning: Could not determine cluster health"
fi

for NODE in "${NODE_ARRAY[@]}"; do
    echo "Checking connectivity to $NODE..."
    if ! ssh $SSH_OPTS "${SSH_PREFIX}${NODE}" "true" 2>/dev/null; then
        echo "Error: Cannot SSH to $NODE"
        exit 1
    fi
    echo "Checking chrony on $NODE..."
    OFFSET=$(ssh $SSH_OPTS "${SSH_PREFIX}${NODE}" \
        "chronyc tracking 2>/dev/null | grep 'Last offset' | awk '{print \$4}'" \
        2>/dev/null || echo "unknown")
    echo "  Clock offset: ${OFFSET}s"
done

echo ""

# Save run configuration
CONFIG_FILE="${RUN_DIR}/config.json"
python3 -c "
import json, sys
config = {
    'run_id': '${RUN_ID}',
    'profile': '${PROFILE}',
    'timestamp': '${TIMESTAMP}',
    'git_sha': '${GIT_SHA}',
    'nodes': '${NODES}'.split(','),
    'pool_type': '${POOL_TYPE}',
    'pool_size': int('${POOL_SIZE}'),
    'ec_profile': '${EC_PROFILE}',
    'pg_num': int('${PG_NUM}'),
    'compression': '${COMPRESS}' or 'none',
    'object_size': '${OBJECT_SIZE}',
    'duration': int('${DURATION}'),
    'warmup': int('${WARMUP}'),
    'threads': int('${THREADS}'),
    'rw_ratio': float('${RW_RATIO}'),
    'collector_interval': float('${COLLECTOR_INTERVAL}'),
    'cluster_health': '${HEALTH_STATUS}',
}
json.dump(config, sys.stdout, indent=2)
" > "$CONFIG_FILE"
echo "Config written to $CONFIG_FILE"

# Copy collector to remote nodes if needed
REMOTE_COLLECTOR="${COLLECTOR_PATH:-/tmp/ceph-osd-cpu-collect.py}"
if [[ -z "$COLLECTOR_PATH" ]]; then
    echo ""
    echo "--- Deploying CPU collector to nodes ---"
    for NODE in "${NODE_ARRAY[@]}"; do
        scp $SSH_OPTS "$SCRIPT_DIR/ceph-osd-cpu-collect.py" \
            "${SSH_PREFIX}${NODE}:${REMOTE_COLLECTOR}" 2>/dev/null
        ssh $SSH_OPTS "${SSH_PREFIX}${NODE}" "chmod +x ${REMOTE_COLLECTOR}" 2>/dev/null
        echo "  Deployed to $NODE"
    done
fi

# Calculate total collection duration (warmup + measurement + buffer)
TOTAL_COLLECT=$((WARMUP + DURATION + 30))

# Start CPU collectors on each node
echo ""
echo "--- Starting CPU collectors ---"
COLLECTOR_PIDS=()
for NODE in "${NODE_ARRAY[@]}"; do
    REMOTE_DIR="/tmp/ceph_validation_${RUN_ID}"
    ssh $SSH_OPTS "${SSH_PREFIX}${NODE}" \
        "mkdir -p ${REMOTE_DIR} && \
         nohup python3 ${REMOTE_COLLECTOR} \
         --interval ${COLLECTOR_INTERVAL} \
         --duration ${TOTAL_COLLECT} \
         --run-id '${RUN_ID}' \
         --output-dir ${REMOTE_DIR} \
         > ${REMOTE_DIR}/collector.log 2>&1 &
         echo \$!" 2>/dev/null
    echo "  Started collector on $NODE"
done

# Brief pause to let collectors initialize
sleep 2

# Run workload driver
echo ""
echo "--- Running workload ---"
WORKLOAD_ARGS=(
    --object-size "$OBJECT_SIZE"
    --duration "$DURATION"
    --warmup "$WARMUP"
    --threads "$THREADS"
    --rw-ratio "$RW_RATIO"
    --pg-num "$PG_NUM"
    --pool-type "$POOL_TYPE"
    --pool-size "$POOL_SIZE"
    --run-id "$RUN_ID"
    --output-dir "$RUN_DIR"
)

if [[ -n "$EC_PROFILE" ]]; then
    WORKLOAD_ARGS+=(--ec-profile "$EC_PROFILE")
fi
if [[ -n "$COMPRESS" ]]; then
    WORKLOAD_ARGS+=(--compress "$COMPRESS")
fi
if [[ -n "$OSD_IDS" ]]; then
    WORKLOAD_ARGS+=(--osd-ids "$OSD_IDS")
fi
if [[ "$KEEP_POOL" -eq 1 ]]; then
    WORKLOAD_ARGS+=(--keep-pool)
fi

python3 "$SCRIPT_DIR/ceph-osd-workload.py" "${WORKLOAD_ARGS[@]}"

# Wait for collectors to finish (they have a duration limit)
echo ""
echo "--- Waiting for CPU collectors to finish ---"
sleep 5

# Retrieve CPU data from each node
echo ""
echo "--- Retrieving CPU data ---"
for NODE in "${NODE_ARRAY[@]}"; do
    REMOTE_DIR="/tmp/ceph_validation_${RUN_ID}"
    NODE_DIR="${RUN_DIR}/cpu_${NODE}"
    mkdir -p "$NODE_DIR"

    scp $SSH_OPTS "${SSH_PREFIX}${NODE}:${REMOTE_DIR}/cpu.csv" \
        "$NODE_DIR/cpu.csv" 2>/dev/null || \
        echo "Warning: no cpu.csv from $NODE"

    scp $SSH_OPTS "${SSH_PREFIX}${NODE}:${REMOTE_DIR}/cpu_meta.json" \
        "$NODE_DIR/cpu_meta.json" 2>/dev/null || \
        echo "Warning: no cpu_meta.json from $NODE"

    scp $SSH_OPTS "${SSH_PREFIX}${NODE}:${REMOTE_DIR}/collector.log" \
        "$NODE_DIR/collector.log" 2>/dev/null || true

    # Cleanup remote temp files
    ssh $SSH_OPTS "${SSH_PREFIX}${NODE}" \
        "rm -rf ${REMOTE_DIR}" 2>/dev/null || true

    echo "  Retrieved data from $NODE"
done

# Merge CPU CSVs from all nodes into a single file
echo ""
echo "--- Merging CPU data ---"
MERGED_CPU="${RUN_DIR}/cpu.csv"
FIRST=1
for NODE in "${NODE_ARRAY[@]}"; do
    NODE_CSV="${RUN_DIR}/cpu_${NODE}/cpu.csv"
    if [[ -f "$NODE_CSV" ]]; then
        if [[ "$FIRST" -eq 1 ]]; then
            cp "$NODE_CSV" "$MERGED_CPU"
            FIRST=0
        else
            tail -n +2 "$NODE_CSV" >> "$MERGED_CPU"
        fi
    fi
done
echo "Merged CPU data to $MERGED_CPU"

# Merge CPU metadata
echo "--- Merging CPU metadata ---"
python3 -c "
import json, glob, os, sys

meta_files = sorted(glob.glob('${RUN_DIR}/cpu_*/cpu_meta.json'))
merged = {'nodes': {}}
for mf in meta_files:
    node = os.path.basename(os.path.dirname(mf)).replace('cpu_', '')
    with open(mf) as f:
        merged['nodes'][node] = json.load(f)

with open('${RUN_DIR}/cpu_meta.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(meta_files)} CPU metadata files')
"

# Generate measured.json summary
echo ""
echo "--- Generating measured.json summary ---"
python3 -c "
import csv, json, os, sys

run_dir = '${RUN_DIR}'

# Load workload metadata
with open(os.path.join(run_dir, 'workload_meta.json')) as f:
    workload = json.load(f)

# Load merged CPU data
cpu_rows = []
cpu_path = os.path.join(run_dir, 'cpu.csv')
if os.path.exists(cpu_path):
    with open(cpu_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpu_rows.append(row)

# Trim CPU samples to measurement window
window = workload.get('measurement_window', {})
t_start = window.get('start', 0)
t_end = window.get('end', float('inf'))

trimmed = [r for r in cpu_rows
           if t_start <= float(r['timestamp']) <= t_end]

# Aggregate per-OSD CPU
per_osd = {}
for row in trimmed:
    osd_id = row['osd_id']
    if osd_id not in per_osd:
        per_osd[osd_id] = {
            'total_cpu_us': 0.0,
            'user_cpu_us': 0.0,
            'system_cpu_us': 0.0,
            'samples': 0,
            'buckets': {},
        }
    entry = per_osd[osd_id]
    entry['total_cpu_us'] += float(row['cpu_us_total'])
    entry['user_cpu_us'] += float(row['cpu_us_user'])
    entry['system_cpu_us'] += float(row['cpu_us_system'])
    entry['samples'] += 1

    bucket = row['bucket']
    if bucket not in entry['buckets']:
        entry['buckets'][bucket] = {'cpu_us': 0.0, 'samples': 0}
    entry['buckets'][bucket]['cpu_us'] += float(row['cpu_us_total'])
    entry['buckets'][bucket]['samples'] += 1

# Compute per-OSD cpu_us_per_op using perf dump deltas
perf_deltas = workload.get('perf_dump_deltas', {})
for osd_id, data in per_osd.items():
    delta = perf_deltas.get(osd_id, {})
    osd_section = delta.get('osd', {})
    op_count = osd_section.get('op', 0)
    if op_count > 0:
        data['op_count'] = op_count
        data['cpu_us_per_op'] = data['total_cpu_us'] / op_count
    else:
        data['op_count'] = 0
        data['cpu_us_per_op'] = None

# Cluster-wide averages
total_ops = sum(d.get('op_count', 0) for d in per_osd.values())
total_cpu = sum(d['total_cpu_us'] for d in per_osd.values())
avg_cpu_per_op = total_cpu / total_ops if total_ops > 0 else None

# Bucket aggregates across all OSDs
bucket_totals = {}
for osd_data in per_osd.values():
    for b, bdata in osd_data['buckets'].items():
        if b not in bucket_totals:
            bucket_totals[b] = 0.0
        bucket_totals[b] += bdata['cpu_us']

measured = {
    'run_id': '${RUN_ID}',
    'measurement_window': window,
    'sample_count': len(trimmed),
    'osd_count': len(per_osd),
    'total_ops': total_ops,
    'total_cpu_us': total_cpu,
    'avg_cpu_us_per_op': avg_cpu_per_op,
    'per_osd': per_osd,
    'bucket_totals': bucket_totals,
}

with open(os.path.join(run_dir, 'measured.json'), 'w') as f:
    json.dump(measured, f, indent=2)
print(f'Measured summary: {len(per_osd)} OSDs, {total_ops} ops, '
      f'avg {avg_cpu_per_op:.2f} us/op' if avg_cpu_per_op else
      f'Measured summary: {len(per_osd)} OSDs (no op count available)')
"

echo ""
echo "============================================"
echo "  Validation run complete: $RUN_ID"
echo "============================================"
echo "Output directory: $RUN_DIR"
echo ""
echo "Bundle contents:"
ls -la "$RUN_DIR/"
echo ""
echo "Next step: run the simulator comparison with:"
echo "  ./ceph-cpu-io-sim.py --validate $RUN_DIR"
