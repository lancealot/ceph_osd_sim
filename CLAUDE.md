# CLAUDE.md — ceph_osd_sim

## Project Vision

Build a comprehensive CPU capacity planning framework for Ceph storage clusters. The goal is to accurately predict whether a given CPU can sustain a specific Ceph workload — not just micro-benchmark individual operations, but model the full spectrum of CPU costs that OSD daemons impose under realistic conditions.

**Design principle:** Every model in this tool must be validated against real cluster data before it ships. Predictions without validation are guesses.

## Project Structure

```
ceph-osd-cpu-simulator/
├── ceph-cpu-io-sim.py          # Main simulator (single-script)
├── test_ceph_cpu_io_sim.py     # Unit and integration tests
├── README.md                   # User-facing documentation
├── CLAUDE.md                   # This file — governance, architecture, roadmap
├── LICENSE                     # GPL-3.0
├── benchmarks/                 # Scripts for running tests against real hardware
│   └── (future)
└── validation/                 # Real cluster data + prediction comparisons
    └── (future)
```

## Architecture

The simulator has four layers (in execution order):

1. **LibraryManager** — Detects available libraries (CRC32C, compression, erasure coding, RocksDB), provides unified wrappers with fallback tiers.
2. **CephBenchmarks** — Runs timed CPU micro-benchmarks: CRC32C, SHA256, compression, erasure coding, serialization, RocksDB simulation, CRUSH placement, recovery paths.
3. **OSDCapacityModel** — Translates benchmark results into "how many OSDs can this CPU sustain" using weighted per-IO CPU cost modeling.
4. **ReportGenerator** — Formats and prints results to terminal.

## Quick Reference

```bash
# Run tests
python3 -m pytest test_ceph_cpu_io_sim.py -v

# Quick smoke test
./ceph-cpu-io-sim.py --quick

# Full benchmark
./ceph-cpu-io-sim.py --drive-type hdd --drive-count 24

# Parallel contention test
./ceph-cpu-io-sim.py --quick --parallel 4
```

## Development Rules

1. **Propose before implementing.** State what you'll add, what it replaces, and how it will be validated.
2. **No speculative features.** Only build what's explicitly requested or on the roadmap.
3. **No "while I'm here" improvements.** Don't refactor adjacent code, add docstrings, or improve error messages in code you didn't change.
4. **Test before committing.** Run `python3 -m pytest test_ceph_cpu_io_sim.py -v` and `./ceph-cpu-io-sim.py --quick`.
5. **Small commits.** Each commit should be a single logical change that could be reverted independently.
6. **Validate against real data.** New models or cost estimates must include a comparison against actual Ceph cluster measurements before landing in the main branch.

## Anti-Patterns — DO NOT

- Add packaging infrastructure (setup.py, pyproject.toml, __init__.py) until the project genuinely needs distribution.
- Add logging frameworks. Use print().
- Add configuration files (YAML, TOML, INI). CLI args are sufficient.
- Split into multiple Python modules prematurely. Single-script simplicity is a feature until the code genuinely outgrows it.
- Add features "for completeness" or "while we're at it."
- Ship a new cost model without validation data to support it.

## Roadmap

### Phase 1: Foundation (current)

CPU micro-benchmark simulator. Measures CRC32C, SHA256, compression, erasure coding, serialization, RocksDB simulation, CRUSH placement. Models per-IO CPU cost. Estimates OSD capacity. Supports mixed media, parallel contention, recovery impact analysis.

**Known gaps in Phase 1:**
- CPU-only model — no memory bandwidth, cache, or NUMA effects
- No network CPU cost modeling
- Simplified BlueStore overhead (flat multipliers)
- Generic workload patterns (read/write ratio only)
- No async I/O or context switching modeling
- No validation against real cluster data

### Phase 2: Accuracy Validation

Compare simulator predictions against real Ceph cluster measurements. Establish error margins. Identify which operations the simulator models well and which it doesn't.

**Work items:**
- Build benchmark scripts that run against live clusters and collect per-OSD CPU usage under controlled workloads
- Collect validation datasets (hardware specs, workload description, actual CPU usage, simulator prediction)
- Document prediction accuracy per operation type
- Calibrate cost model weights based on observed data

### Phase 3: Memory and NUMA Modeling

Add memory bandwidth profiling and NUMA-aware benchmarking. Real Ceph workloads can be memory-bandwidth-bound, especially with large object sizes and high OSD counts.

**Work items:**
- Benchmark memory bandwidth per NUMA node
- Pin benchmark workers to NUMA nodes to measure cross-node penalty
- Model L3 cache contention when many OSDs share a CPU
- Integrate memory bandwidth as a capacity constraint alongside CPU time

### Phase 4: I/O and Network CPU Cost

Model the CPU cost of async I/O handling, interrupt processing, and network packet handling for replication and recovery traffic.

**Work items:**
- Model BlueStore async I/O completion handling and context switching overhead
- Model network CPU cost for replication (serialize, checksum, transmit per replica)
- Model recovery network amplification (read N chunks, decode, re-encode, write)
- Integrate I/O and network CPU as capacity constraints

### Phase 5: Workload Pattern Library

Replace the simple read/write ratio with realistic workload profiles that model specific access patterns.

**Work items:**
- Define workload profiles: sequential large writes, random small reads, metadata-heavy (CephFS), mixed (RGW/S3), Zarr/HDF5 scientific data patterns
- Model how each workload profile maps to different CPU operation mixes
- Allow users to define custom workload profiles
- Validate each profile against real cluster measurements

### Phase 6: BlueStore Deep Modeling

Replace flat BlueStore overhead multipliers with realistic RocksDB compaction modeling, WAL/DB I/O patterns, and level amplification.

**Work items:**
- Model RocksDB compaction scheduling and CPU spikes
- Model WAL/DB write amplification under different write rates
- Model the relationship between object size, key density, and compaction frequency
- Validate against real BlueStore performance counters

### Future Considerations

- **Hardware database:** Community-contributed benchmark results for common CPU models, building a reference library of "CPU X can handle Y OSDs"
- **What-if analysis:** Model the impact of adding drives, changing EC profiles, enabling compression, etc. without re-running benchmarks
- **Integration with ceph-bench:** Cross-reference simulator predictions with live benchmark results from [ceph-bench](https://github.com/lancealot/ceph-bench)

## Validation Workflow

This is the process for validating simulator accuracy against real clusters:

1. **Define the test:** Hardware specs, Ceph configuration, workload description, duration.
2. **Run the workload** on a real cluster. Collect per-OSD CPU usage (e.g., `ceph daemon osd.N perf dump`, `top`, `pidstat`).
3. **Run the simulator** with matching configuration (same CPU, same drive count, same protection scheme, same object size).
4. **Compare:** Predicted CPU cost per IO vs. observed CPU cost per IO. Document the delta.
5. **Store the results** in `validation/` with a descriptive filename and metadata.
6. **If the delta is large:** Investigate which operation the model gets wrong. File an issue or adjust the cost model with justification.

## Origin

This project was originally part of [ceph-bench](https://github.com/lancealot/ceph-bench). The live OSD benchmarking tools (`ceph-bench.sh`, `ceph-analysis.py`) remain in that repository. The simulator was split out to give it room to grow into a comprehensive capacity planning framework.
