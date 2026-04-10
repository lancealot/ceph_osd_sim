# ceph-osd-cpu-simulator

A CPU capacity planning tool for Ceph storage clusters. Benchmarks local CPU performance on Ceph-specific operations (checksumming, compression, erasure coding, serialization, metadata ops) and estimates how many OSDs a given CPU can sustain at various drive speeds.

**What this tool does:** Runs real Ceph-like CPU operations on your hardware and models per-IO CPU cost to predict OSD capacity limits.

**What this tool does not do:** It does not benchmark actual Ceph clusters, measure network throughput, or test storage devices. It measures CPU in isolation. Use it as one input to a broader capacity planning process, not as a standalone predictor.

> **Status:** v1.0 — CPU micro-benchmark foundation. See [CLAUDE.md](CLAUDE.md) for the development roadmap toward comprehensive workload modeling.

## Quick Start

No external dependencies required. Runs on Python 3.6+ with stdlib only.

```bash
# Quick benchmark with defaults
./ceph-cpu-io-sim.py --quick

# HDD cluster: 24 drives, 3x replication
./ceph-cpu-io-sim.py --drive-type hdd --drive-count 24 --protection replicated:3

# NVMe cluster with erasure coding and compression
./ceph-cpu-io-sim.py --drive-type nvme --drive-count 4 --protection ec:4+2 --compress zstd

# Mixed media: 24 HDDs + 4 NVMe (2 OSDs each), WAL/DB offloaded
./ceph-cpu-io-sim.py --drives 24xhdd 4xnvme:0:2 --wal-db-separate

# Simulate recovery with 1 OSD failure
./ceph-cpu-io-sim.py --drive-type hdd --drive-count 36 --recovery-osds 1

# Benchmark under contention (simulate 32 OSD daemons competing for CPU)
./ceph-cpu-io-sim.py --drives 24xhdd 4xnvme:0:2 --parallel 32

# Run all scenarios (best/typical/worst) for capacity planning
./ceph-cpu-io-sim.py --drive-type ssd --drive-count 8 --scenario all
```

## What It Measures

The simulator benchmarks six categories of CPU work that Ceph OSD daemons perform:

| Operation | What It Models |
|---|---|
| **CRC32C** | Data integrity checksumming (every read and write) |
| **SHA256** | Authentication and scrub checksums |
| **Compression** | BlueStore inline compression (lz4, zstd, snappy, zlib) |
| **Erasure Coding** | EC encode/decode for erasure-coded pools |
| **Serialization** | OSD message framing and RADOS object headers |
| **RocksDB / Metadata** | BlueStore key-value operations and CRUSH placement |

Results are combined into a weighted per-IO CPU cost model that estimates how many OSDs the CPU can drive at a given IOPS rate.

## Report Sections

| Section | Description |
|---|---|
| Benchmark Results | Raw CPU micro-benchmark performance per operation and object size |
| CPU Cost Per IO | Weighted cost breakdown showing which operations dominate |
| OSD Capacity Estimate | Maximum OSDs the CPU can sustain per device class |
| Scale-Out Projection | Cluster performance extrapolated to 1–64 nodes |
| Recovery Impact | CPU overhead during OSD failure recovery (with `--recovery-osds`) |
| CPU Scaling Analysis | Whether to add more cores or get faster cores |

## Optional Dependencies

All benchmarks have stdlib fallbacks, but accuracy improves with native libraries:

| Library | Purpose | Install |
|---|---|---|
| `crcmod` or `crc32c` | Hardware-accelerated CRC32C | `pip install crcmod` |
| `pyeclib` | Real Reed-Solomon erasure coding | `pip install pyeclib` |
| `lz4` | Native LZ4 compression | `pip install lz4` |
| `pyzstd` or `zstandard` | Native Zstandard compression | `pip install pyzstd` |
| `python-snappy` | Native Snappy compression | `pip install python-snappy` |

The simulator auto-detects available libraries and reports which implementations are in use.

## Known Limitations

- **CPU-only model.** Does not account for memory bandwidth, cache effects, NUMA locality, or I/O scheduling overhead. Real workloads can be memory-bandwidth-bound.
- **No network CPU cost.** Replication and recovery network handling (packet processing, checksumming, queue management) is not modeled.
- **Simplified BlueStore model.** WAL/DB separation overhead is modeled as a flat multiplier. Real RocksDB compaction patterns, level amplification, and I/O-triggered CPU spikes are not captured.
- **Generic workload patterns.** Read/write ratio is configurable, but specific workload shapes (small metadata-heavy, large sequential, mixed random) are not yet modeled as distinct profiles.
- **No async I/O modeling.** Bluestore's asynchronous I/O patterns, interrupt handling, and context switching overhead are not represented.

These limitations are addressed in the [development roadmap](CLAUDE.md#roadmap).

## Running Tests

```bash
python3 -m pytest test_ceph_cpu_io_sim.py -v
```

## Related Tools

- **[ceph-bench](https://github.com/lancealot/ceph-bench)** — Live OSD benchmarking and analysis scripts for identifying slow drives in a running Ceph cluster. This simulator was originally developed within that project.
- **[ceph_primary_balancer](https://github.com/lancealot/ceph_primary_balancer)** — Equalizes primary OSD assignments across OSD, host, and pool dimensions using `pg-upmap-primary`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
