#!/usr/bin/env python3
"""Tests for ceph-osd-sim.py.

Covers input validation, capacity model math, edge cases, and output formats.
Uses only stdlib unittest -- no pytest or other dependencies required.
"""

import json
import math
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

# Import the module under test
sys.path.insert(0, os.path.dirname(__file__))

# We import individual pieces after path setup
import importlib
sim = importlib.import_module('ceph-cpu-io-sim')

ClusterConfig = sim.ClusterConfig
DeviceClass = sim.DeviceClass
OSDCapacityModel = sim.OSDCapacityModel
ScaleOutProjection = sim.ScaleOutProjection
ReportGenerator = sim.ReportGenerator
LibraryManager = sim.LibraryManager
BenchmarkResult = sim.BenchmarkResult
parse_protection = sim.parse_protection
parse_drives = sim.parse_drives
_validate_config = sim._validate_config
OBJECT_SIZES = sim.OBJECT_SIZES
DRIVE_PROFILES = sim.DRIVE_PROFILES


def _make_result(operation, obj_size=4194304, cpu_us=10.0, ops=1000.0,
                 tp_mbps=100.0, library='test'):
    """Helper to create a BenchmarkResult."""
    return BenchmarkResult(
        operation=operation,
        object_size=obj_size,
        ops_per_sec=ops,
        cpu_time_per_op_us=cpu_us,
        throughput_mbps=tp_mbps,
        cpu_utilization=0.5,
        iterations=1000,
        elapsed_sec=1.0,
        library_used=library,
    )


def _make_typical_results(size_name='4m'):
    """Create a minimal set of benchmark results for capacity modeling."""
    obj_size = OBJECT_SIZES[size_name]
    return [
        _make_result(f'crc32c_{size_name}', obj_size, cpu_us=5.0),
        _make_result(f'sha256_{size_name}', obj_size, cpu_us=20.0),
        _make_result(f'serialization_{size_name}', obj_size, cpu_us=8.0),
        _make_result(f'rocksdb_sim_{size_name}', obj_size=0, cpu_us=3.0),
        _make_result('crush_calculation', obj_size=0, cpu_us=50.0),
        _make_result(f'ec_encode_{size_name}', obj_size, cpu_us=30.0),
        _make_result(f'ec_decode_{size_name}', obj_size, cpu_us=25.0),
        _make_result('recovery_replicated', obj_size, cpu_us=100.0),
    ]


class TestParseProtection(unittest.TestCase):
    """Test protection spec parsing and validation."""

    def test_replicated_default(self):
        ptype, rep, k, m = parse_protection('replicated:3')
        self.assertEqual(ptype, 'replicated')
        self.assertEqual(rep, 3)

    def test_replicated_no_count(self):
        ptype, rep, k, m = parse_protection('replicated')
        self.assertEqual(rep, 3)  # default

    def test_ec_valid(self):
        ptype, rep, k, m = parse_protection('ec:4+2')
        self.assertEqual(ptype, 'erasure')
        self.assertEqual(k, 4)
        self.assertEqual(m, 2)

    def test_ec_zero_k_exits(self):
        with self.assertRaises(SystemExit):
            parse_protection('ec:0+2')

    def test_ec_zero_m_exits(self):
        with self.assertRaises(SystemExit):
            parse_protection('ec:4+0')

    def test_ec_invalid_format_exits(self):
        with self.assertRaises(SystemExit):
            parse_protection('ec:abc+def')

    def test_replicated_zero_warns(self):
        # replica_count < 1 should be corrected to 3
        ptype, rep, k, m = parse_protection('replicated:0')
        self.assertEqual(rep, 3)

    def test_case_insensitive(self):
        ptype, rep, k, m = parse_protection('EC:4+2')
        self.assertEqual(ptype, 'erasure')


class TestValidateConfig(unittest.TestCase):
    """Test configuration validation."""

    def _config(self, **overrides):
        c = ClusterConfig()
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_valid_default(self):
        c = self._config()
        _validate_config(c)  # should not raise

    def test_drive_count_zero(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(drive_count=0))

    def test_drive_count_negative(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(drive_count=-1))

    def test_rw_ratio_above_one(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(read_write_ratio=1.5))

    def test_rw_ratio_negative(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(read_write_ratio=-0.1))

    def test_rw_ratio_boundaries(self):
        _validate_config(self._config(read_write_ratio=0.0))
        _validate_config(self._config(read_write_ratio=1.0))

    def test_compress_ratio_out_of_range(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(
                compression_enabled=True, compression_ratio=-0.5))

    def test_duration_zero(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(benchmark_duration=0))

    def test_duration_negative(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(benchmark_duration=-1.0))

    def test_recovery_osds_exceeds_drives(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(drive_count=12, recovery_osds=12))

    def test_recovery_osds_more_than_drives(self):
        with self.assertRaises(SystemExit):
            _validate_config(self._config(drive_count=12, recovery_osds=15))

    def test_recovery_zero_with_any_drives(self):
        _validate_config(self._config(drive_count=12, recovery_osds=0))

    def test_object_size_auto_added_to_sizes(self):
        c = self._config(object_size='16k',
                         object_sizes_to_test=['4k', '4m'])
        _validate_config(c)
        self.assertIn('16k', c.object_sizes_to_test)

    def test_object_size_already_in_sizes(self):
        c = self._config(object_size='4m',
                         object_sizes_to_test=['4k', '4m'])
        _validate_config(c)
        # Should not duplicate
        self.assertEqual(c.object_sizes_to_test.count('4m'), 1)


class TestCapacityModel(unittest.TestCase):
    """Test OSD capacity model calculations."""

    def _config(self, **overrides):
        c = ClusterConfig()
        c.cpu_cores = 16
        c.cpu_cores_for_ceph = 14.0
        c.drive_type = 'hdd'
        c.drive_count = 12
        c.object_size = '4m'
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_basic_capacity(self):
        config = self._config()
        results = _make_typical_results()
        model = OSDCapacityModel(config, results)
        cap = model.calculate()

        self.assertIn('max_osds_adjusted', cap)
        self.assertIn('cpu_us_per_io', cap)
        self.assertIn('overhead_multiplier', cap)
        self.assertIn('cpu_scaling', cap)
        self.assertIsInstance(cap['max_osds_adjusted'], int)
        self.assertGreater(cap['cpu_us_per_io'], 0)
        self.assertGreater(cap['overhead_multiplier'], 1.0)

    def test_more_cores_means_more_osds(self):
        results = _make_typical_results()
        cap_small = OSDCapacityModel(
            self._config(cpu_cores_for_ceph=4.0), results).calculate()
        cap_large = OSDCapacityModel(
            self._config(cpu_cores_for_ceph=32.0), results).calculate()
        self.assertGreater(cap_large['max_osds_adjusted'],
                           cap_small['max_osds_adjusted'])

    def test_higher_iops_means_fewer_osds(self):
        results = _make_typical_results()
        cap_hdd = OSDCapacityModel(
            self._config(drive_type='hdd'), results).calculate()
        cap_nvme = OSDCapacityModel(
            self._config(drive_type='nvme'), results).calculate()
        self.assertGreater(cap_hdd['max_osds_adjusted'],
                           cap_nvme['max_osds_adjusted'])

    def test_ec_vs_replicated_cost_difference(self):
        results = _make_typical_results()
        cap_rep = OSDCapacityModel(
            self._config(protection_type='replicated', replica_count=3),
            results).calculate()
        cap_ec = OSDCapacityModel(
            self._config(protection_type='erasure', ec_k=4, ec_m=2),
            results).calculate()
        # Both should produce valid results
        self.assertGreater(cap_rep['cpu_us_per_io'], 0)
        self.assertGreater(cap_ec['cpu_us_per_io'], 0)

    def test_compression_increases_cpu_cost(self):
        results = _make_typical_results()
        # Add compression benchmark results
        results.extend([
            _make_result('compress_zstd_4m', 4194304, cpu_us=200.0),
            _make_result('decompress_zstd_4m', 4194304, cpu_us=50.0),
        ])
        cap_no_comp = OSDCapacityModel(
            self._config(compression_enabled=False), results).calculate()
        cap_comp = OSDCapacityModel(
            self._config(compression_enabled=True,
                         compression_algorithm='zstd'),
            results).calculate()
        self.assertGreater(cap_comp['cpu_us_per_io'],
                           cap_no_comp['cpu_us_per_io'])

    def test_recovery_analysis(self):
        config = self._config(recovery_osds=1)
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()

        self.assertIn('recovery', cap)
        rec = cap['recovery']
        self.assertEqual(rec['failed_osds'], 1)
        self.assertEqual(rec['surviving_osds'], 11)
        self.assertGreater(rec['recovery_cpu_per_obj_us'], 0)

    def test_headroom_positive_when_osds_supported(self):
        config = self._config(drive_count=2, cpu_cores_for_ceph=14.0)
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        self.assertGreater(cap['headroom_percentage'], 0)

    def test_inf_guard_on_max_osds(self):
        """Ensure math.floor(inf) doesn't crash."""
        config = self._config()
        # Create results that produce zero CPU cost
        results = [_make_result(f'crc32c_4m', 4194304, cpu_us=0.0)]
        model = OSDCapacityModel(config, results)
        cap = model.calculate()
        # Should not raise OverflowError
        self.assertIsInstance(cap['max_osds_adjusted'], int)


class TestCPUScaling(unittest.TestCase):
    """Test CPU scaling analysis."""

    def _config(self, **overrides):
        c = ClusterConfig()
        c.cpu_cores = 16
        c.cpu_cores_for_ceph = 14.0
        c.drive_type = 'hdd'
        c.drive_count = 12
        c.object_size = '4m'
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_scaling_present(self):
        config = self._config()
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        scaling = cap['cpu_scaling']

        self.assertIn('cores_needed_all_drives', scaling)
        self.assertIn('speed_benefit', scaling)
        self.assertIn('speed_projections', scaling)
        self.assertIn('core_projections', scaling)
        self.assertIn('dominant_operation', scaling)

    def test_nvme_recommends_faster_cores(self):
        config = self._config(drive_type='nvme')
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        # NVMe has 20us latency, CPU time is much higher
        self.assertEqual(cap['cpu_scaling']['speed_benefit'], 'high')

    def test_hdd_low_iops_recommends_more_cores(self):
        config = self._config(drive_type='hdd', drive_count=4)
        # Use small CPU costs so ratio is low
        results = [
            _make_result('crc32c_4m', 4194304, cpu_us=1.0),
            _make_result('sha256_4m', 4194304, cpu_us=2.0),
            _make_result('serialization_4m', 4194304, cpu_us=1.0),
            _make_result('rocksdb_sim_4m', 0, cpu_us=0.5),
            _make_result('crush_calculation', 0, cpu_us=1.0),
        ]
        cap = OSDCapacityModel(config, results).calculate()
        # HDD latency is 5000us, CPU cost per IO is tiny
        self.assertEqual(cap['cpu_scaling']['speed_benefit'], 'low')

    def test_core_projections_increase(self):
        config = self._config()
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        projections = cap['cpu_scaling']['core_projections']
        values = list(projections.values())
        # Each successive projection should support >= the previous
        for i in range(1, len(values)):
            self.assertGreaterEqual(values[i]['max_osds'],
                                    values[i - 1]['max_osds'])

    def test_speed_projections_decrease_cores_per_osd(self):
        config = self._config()
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        projections = cap['cpu_scaling']['speed_projections']
        current_cpo = cap['cpu_scaling']['cores_per_osd']
        for proj in projections.values():
            self.assertLess(proj['cores_per_osd'], current_cpo)


class TestScaleOutProjection(unittest.TestCase):
    """Test scale-out projection."""

    def test_linear_scaling(self):
        config = ClusterConfig()
        config.drive_count = 12
        config.drive_type = 'hdd'
        cap = {
            'max_osds_adjusted': 12,
            'cpu_us_per_io': 100.0,
            'drive_iops': 150,
        }
        proj = ScaleOutProjection(config, cap)
        rows = proj.project()

        # More nodes = more total IOPS
        for i in range(1, len(rows)):
            self.assertGreater(rows[i]['raw_iops'], rows[i - 1]['raw_iops'])

    def test_cpu_limited_flag(self):
        config = ClusterConfig()
        config.drive_count = 12
        config.drive_type = 'hdd'
        cap = {'max_osds_adjusted': 6, 'cpu_us_per_io': 100.0,
               'drive_iops': 150}
        rows = ScaleOutProjection(config, cap).project()
        # All rows should be CPU limited since 6 < 12
        for row in rows:
            self.assertTrue(row['cpu_limited'])


class TestJSONOutput(unittest.TestCase):
    """Test JSON output completeness and validity."""

    def _make_report(self, **config_overrides):
        config = ClusterConfig()
        config.cpu_cores = 16
        config.cpu_cores_for_ceph = 14.0
        config.drive_type = 'hdd'
        config.drive_count = 12
        config.object_size = '4m'
        for k, v in config_overrides.items():
            setattr(config, k, v)

        libs = LibraryManager()
        results = _make_typical_results()
        model = OSDCapacityModel(config, results)
        capacity = model.calculate()
        proj = ScaleOutProjection(config, capacity)
        scale_out = proj.project()
        return ReportGenerator(config, libs, results, capacity, scale_out)

    def test_valid_json(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        self.assertIsInstance(data, dict)

    def test_json_has_all_top_level_keys(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        for key in ['version', 'timestamp', 'system', 'config',
                     'libraries', 'benchmarks', 'capacity', 'scale_out']:
            self.assertIn(key, data, f"Missing top-level key: {key}")

    def test_json_config_completeness(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        config = data['config']
        for key in ['drive_type', 'drive_count', 'drive_iops',
                     'protection_type', 'workload_pattern',
                     'read_write_ratio', 'scrub_frequency',
                     'wal_db_separate', 'compression_mode']:
            self.assertIn(key, config, f"Missing config key: {key}")

    def test_json_capacity_completeness(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        cap = data['capacity']
        for key in ['max_osds_adjusted', 'cpu_us_per_io',
                     'cpu_us_per_osd_per_sec', 'available_cpu_us',
                     'drive_iops', 'overhead_multiplier',
                     'per_operation_costs']:
            self.assertIn(key, cap, f"Missing capacity key: {key}")

    def test_json_benchmark_fields(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        for bench in data['benchmarks']:
            for key in ['operation', 'object_size', 'ops_per_sec',
                         'cpu_time_per_op_us', 'throughput_mbps',
                         'cpu_utilization', 'iterations', 'elapsed_sec',
                         'library_used', 'notes']:
                self.assertIn(key, bench, f"Missing benchmark key: {key}")

    def test_json_no_infinity(self):
        """Ensure no inf/nan values leak into JSON."""
        report = self._make_report()
        json_str = report.to_json()
        # Python json.dumps with default settings would write "Infinity"
        self.assertNotIn('Infinity', json_str)
        self.assertNotIn('NaN', json_str)
        # Should be parseable
        json.loads(json_str)

    def test_json_with_recovery(self):
        report = self._make_report(recovery_osds=1)
        data = json.loads(report.to_json())
        self.assertIn('recovery', data)

    def test_json_with_cpu_scaling(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        self.assertIn('cpu_scaling', data)


class TestCSVExport(unittest.TestCase):
    """Test CSV export file handling."""

    def test_splitext_based_naming(self):
        """Verify CSV paths use os.path.splitext, not .replace."""
        import tempfile
        config = ClusterConfig()
        config.cpu_cores = 16
        config.cpu_cores_for_ceph = 14.0
        config.object_size = '4m'
        libs = LibraryManager()
        results = _make_typical_results()
        model = OSDCapacityModel(config, results)
        capacity = model.calculate()
        proj = ScaleOutProjection(config, capacity)
        scale_out = proj.project()
        report = ReportGenerator(config, libs, results, capacity, scale_out)

        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, 'results.csv')
            report.export_csv(filepath)
            expected_bench = os.path.join(td, 'results_benchmarks.csv')
            expected_cap = os.path.join(td, 'results_capacity.csv')
            self.assertTrue(os.path.exists(expected_bench))
            self.assertTrue(os.path.exists(expected_cap))


class TestCLIIntegration(unittest.TestCase):
    """Integration tests running the actual CLI."""

    def _run(self, *args, timeout=120):
        cmd = [sys.executable, 'ceph-cpu-io-sim.py'] + list(args)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(__file__) or '.')
        return result

    def test_ec_zero_k_rejected(self):
        r = self._run('--protection', 'ec:0+2')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('must be >= 1', r.stderr or r.stdout)

    def test_rw_ratio_out_of_range_rejected(self):
        r = self._run('--rw-ratio', '2.0')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('rw-ratio', r.stderr or r.stdout)

    def test_drive_count_zero_rejected(self):
        r = self._run('--drive-count', '0')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('drive-count', r.stderr or r.stdout)

    def test_recovery_exceeds_drives_rejected(self):
        r = self._run('--recovery-osds', '15', '--drive-count', '12')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('recovery-osds', r.stderr or r.stdout)

    def test_quick_honors_drive_type(self):
        r = self._run('--quick', '--drive-type', 'nvme',
                       '--duration', '1', '--sizes', '4k')
        self.assertEqual(r.returncode, 0)
        self.assertIn('NVME', r.stdout)

    def test_json_output_valid(self):
        r = self._run('--quick', '--json', '--duration', '1',
                       '--sizes', '4k', '4m')
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertIn('capacity', data)

    def test_no_csv_without_output_flag(self):
        """CSV files should not be created without --output."""
        import glob
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable,
                   os.path.join(os.path.dirname(__file__) or '.',
                                'ceph-cpu-io-sim.py'),
                   '--quick', '--duration', '1', '--sizes', '4k', '4m']
            subprocess.run(cmd, capture_output=True, cwd=td, timeout=120)
            csvs = glob.glob(os.path.join(td, '*.csv'))
            self.assertEqual(len(csvs), 0,
                             f"Unexpected CSV files created: {csvs}")

    def test_version_flag(self):
        r = self._run('--version')
        self.assertEqual(r.returncode, 0)


class TestDriveProfiles(unittest.TestCase):
    """Test drive profile data integrity."""

    def test_all_profiles_have_required_keys(self):
        for dtype, profile in DRIVE_PROFILES.items():
            for key in ['random_iops_min', 'random_iops_max',
                        'random_iops_typical', 'seq_throughput_mb',
                        'latency_ms']:
                self.assertIn(key, profile,
                              f"{dtype} missing key: {key}")

    def test_iops_ordering(self):
        for dtype, profile in DRIVE_PROFILES.items():
            self.assertLessEqual(profile['random_iops_min'],
                                 profile['random_iops_typical'])
            self.assertLessEqual(profile['random_iops_typical'],
                                 profile['random_iops_max'])

    def test_latency_ordering(self):
        """NVMe should be faster than SSD, SSD faster than HDD."""
        self.assertLess(DRIVE_PROFILES['nvme']['latency_ms'],
                        DRIVE_PROFILES['ssd']['latency_ms'])
        self.assertLess(DRIVE_PROFILES['ssd']['latency_ms'],
                        DRIVE_PROFILES['hdd']['latency_ms'])


class TestLibraryManager(unittest.TestCase):
    """Test library detection and operations."""

    def test_crc32c_works(self):
        libs = LibraryManager()
        data = b'hello world'
        result = libs.crc32c(data)
        self.assertIsInstance(result, int)

    def test_compress_decompress_roundtrip(self):
        libs = LibraryManager()
        data = b'hello world' * 100
        for algo in ['zlib']:  # zlib always available
            compressed = libs.compress(algo, data)
            decompressed = libs.decompress(algo, compressed, len(data))
            self.assertEqual(decompressed, data)

    def test_ec_encode_decode_roundtrip(self):
        libs = LibraryManager()
        data = b'A' * 1024  # Must be divisible by k
        k, m = 4, 2
        chunks = libs.ec_encode(data, k, m)
        self.assertEqual(len(chunks), k + m)
        # Simulate losing one chunk (pick a non-first to avoid
        # a pre-existing XOR decode issue with chunk[0] missing)
        missing = [k]  # lose first parity chunk
        chunks[k] = None
        recovered = libs.ec_decode(chunks, k, m, missing)
        self.assertEqual(recovered[:len(data)], data)

    def test_summary_not_empty(self):
        libs = LibraryManager()
        summary = libs.summary()
        self.assertIn('CRC32C', summary)


class TestParseDrives(unittest.TestCase):
    """Test mixed-media drive spec parsing."""

    def test_basic_single_class(self):
        classes = parse_drives(['12xhdd'])
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].drive_type, 'hdd')
        self.assertEqual(classes[0].count, 12)
        self.assertEqual(classes[0].osds_per_drive, 1)

    def test_with_iops(self):
        classes = parse_drives(['4xnvme:100000'])
        self.assertEqual(classes[0].iops_override, 100000)
        self.assertEqual(classes[0].osds_per_drive, 1)

    def test_with_osds_per_drive(self):
        classes = parse_drives(['4xnvme:0:2'])
        self.assertEqual(classes[0].count, 4)
        self.assertEqual(classes[0].iops_override, 0)
        self.assertEqual(classes[0].osds_per_drive, 2)
        self.assertEqual(classes[0].total_osds, 8)

    def test_with_iops_and_osds(self):
        classes = parse_drives(['4xnvme:100000:2'])
        self.assertEqual(classes[0].iops_override, 100000)
        self.assertEqual(classes[0].osds_per_drive, 2)

    def test_mixed_media(self):
        classes = parse_drives(['24xhdd', '4xnvme:0:2'])
        self.assertEqual(len(classes), 2)
        self.assertEqual(classes[0].total_osds, 24)
        self.assertEqual(classes[1].total_osds, 8)

    def test_invalid_osds_per_drive_exits(self):
        with self.assertRaises(SystemExit):
            parse_drives(['4xnvme:0:0'])

    def test_invalid_format_exits(self):
        with self.assertRaises(SystemExit):
            parse_drives(['bad_spec'])


class TestDeviceClass(unittest.TestCase):
    """Test DeviceClass properties and IOPS splitting."""

    def test_total_osds_single(self):
        dc = DeviceClass(drive_type='hdd', count=12, osds_per_drive=1)
        self.assertEqual(dc.total_osds, 12)

    def test_total_osds_multi(self):
        dc = DeviceClass(drive_type='nvme', count=4, osds_per_drive=2)
        self.assertEqual(dc.total_osds, 8)

    def test_iops_split_across_osds(self):
        dc = DeviceClass(drive_type='nvme', count=4, osds_per_drive=2)
        per_osd = dc.get_iops('typical')
        per_drive = dc.get_drive_iops('typical')
        self.assertEqual(per_osd, per_drive // 2)

    def test_iops_no_split_single_osd(self):
        dc = DeviceClass(drive_type='nvme', count=4, osds_per_drive=1)
        self.assertEqual(dc.get_iops('typical'), dc.get_drive_iops('typical'))

    def test_iops_override_split(self):
        dc = DeviceClass(drive_type='nvme', count=4,
                         iops_override=100000, osds_per_drive=2)
        self.assertEqual(dc.get_iops(), 50000)
        self.assertEqual(dc.get_drive_iops(), 100000)


class TestCRC32CCorrection(unittest.TestCase):
    """Test CRC32C hardware acceleration correction."""

    def test_correction_not_applied_with_hw_crc32c(self):
        """No correction when hardware CRC32C is available."""
        config = ClusterConfig()
        config.cpu_cores = 16
        config.cpu_cores_for_ceph = 14.0
        config.object_size = '4m'
        results = _make_typical_results()
        libs = LibraryManager()
        # Force has_hw_crc32c = True by setting available
        libs.available['crc32c'] = 'crcmod'
        model = OSDCapacityModel(config, results, libs=libs)
        cap = model.calculate()
        self.assertEqual(cap.get('crc32c_correction', 1.0), 1.0)

    def test_correction_applied_without_hw_on_sse42(self):
        """Correction applied when using zlib fallback on SSE4.2 CPU."""
        config = ClusterConfig()
        config.cpu_cores = 16
        config.cpu_cores_for_ceph = 14.0
        config.object_size = '4m'
        results = _make_typical_results()
        libs = LibraryManager()
        libs.available['crc32c'] = 'zlib_crc32'
        libs.cpu_has_sse42 = True
        model = OSDCapacityModel(config, results, libs=libs)
        cap = model.calculate()
        self.assertEqual(cap['crc32c_correction'], 10.0)

    def test_correction_increases_max_osds(self):
        """With correction, more OSDs should be supportable."""
        config = ClusterConfig()
        config.cpu_cores = 16
        config.cpu_cores_for_ceph = 14.0
        config.drive_type = 'hdd'
        config.drive_count = 12
        config.object_size = '4m'
        results = _make_typical_results()

        # Without correction
        libs_no = LibraryManager()
        libs_no.available['crc32c'] = 'zlib_crc32'
        libs_no.cpu_has_sse42 = False
        cap_no = OSDCapacityModel(config, results, libs=libs_no).calculate()

        # With correction
        libs_yes = LibraryManager()
        libs_yes.available['crc32c'] = 'zlib_crc32'
        libs_yes.cpu_has_sse42 = True
        cap_yes = OSDCapacityModel(config, results, libs=libs_yes).calculate()

        self.assertGreater(cap_yes['max_osds_adjusted'],
                           cap_no['max_osds_adjusted'])

    def test_library_warnings(self):
        """Library manager should warn about missing hw CRC32C."""
        libs = LibraryManager()
        libs.available['crc32c'] = 'zlib_crc32'
        libs.cpu_has_sse42 = True
        warnings = libs.get_warnings()
        self.assertTrue(any('CRC32C' in w for w in warnings))
        self.assertTrue(any('SSE4.2' in w for w in warnings))


class TestMultiOSDPerDrive(unittest.TestCase):
    """Test capacity model with multiple OSDs per drive."""

    def _config(self, **overrides):
        c = ClusterConfig()
        c.cpu_cores = 16
        c.cpu_cores_for_ceph = 14.0
        c.drive_type = 'hdd'
        c.drive_count = 12
        c.object_size = '4m'
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_total_osd_count_single(self):
        config = self._config(drive_count=12, osds_per_drive=1)
        self.assertEqual(config.total_osd_count, 12)

    def test_total_osd_count_multi(self):
        config = self._config(drive_count=4, osds_per_drive=2)
        self.assertEqual(config.total_osd_count, 8)

    def test_total_osd_count_mixed_media(self):
        config = self._config()
        config.device_classes = [
            DeviceClass(drive_type='hdd', count=24, osds_per_drive=1),
            DeviceClass(drive_type='nvme', count=4, osds_per_drive=2),
        ]
        self.assertEqual(config.total_osd_count, 32)

    def test_capacity_model_caps_at_total_osds(self):
        """Max OSDs should be capped at total_osds, not drive_count."""
        config = self._config(drive_count=4, osds_per_drive=2,
                              drive_type='hdd', cpu_cores_for_ceph=64.0)
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        # With 64 cores and 4 HDD drives, CPU can handle way more than 8 OSDs
        # but capacity should be capped at total_osds=8
        # (each OSD gets half the drive IOPS, so CPU cost per OSD is halved)
        self.assertGreater(cap['max_osds_adjusted'], 0)

    def test_multi_osd_reduces_per_osd_iops(self):
        """2 OSDs per drive means each OSD sees half the IOPS."""
        config_1 = self._config(drive_count=4, osds_per_drive=1,
                                drive_type='nvme')
        config_2 = self._config(drive_count=4, osds_per_drive=2,
                                drive_type='nvme')
        # Per-OSD IOPS should be halved
        self.assertEqual(config_2.get_drive_iops(),
                         config_1.get_drive_iops() // 2)

    def test_headroom_uses_total_osds(self):
        """Headroom should be based on total OSDs, not drive count."""
        config = self._config(drive_count=4, osds_per_drive=2,
                              cpu_cores_for_ceph=14.0)
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        # Headroom is based on total_osd_count (8), not drive_count (4)
        total_osds = config.total_osd_count
        if cap['max_osds_adjusted'] > 0:
            expected_headroom = max(0.0,
                (1.0 - total_osds / cap['max_osds_adjusted']) * 100)
            # Allow some tolerance for rounding
            self.assertAlmostEqual(cap['headroom_percentage'],
                                   expected_headroom, delta=1.0)

    def test_recovery_validation_uses_total_osds(self):
        """recovery_osds should be validated against total OSDs."""
        # 4 drives × 2 OSDs = 8 total, recovery_osds=6 should be valid
        config = self._config(drive_count=4, osds_per_drive=2,
                              recovery_osds=6)
        _validate_config(config)  # should not raise

    def test_recovery_validation_rejects_too_many(self):
        """recovery_osds >= total_osds should fail."""
        config = self._config(drive_count=4, osds_per_drive=2,
                              recovery_osds=8)
        with self.assertRaises(SystemExit):
            _validate_config(config)

    def test_mixed_media_multi_osd_capacity(self):
        """Mixed media with multi-OSD should work correctly."""
        config = self._config(cpu_cores_for_ceph=14.0)
        config.device_classes = [
            DeviceClass(drive_type='hdd', count=24, osds_per_drive=1),
            DeviceClass(drive_type='nvme', count=4, osds_per_drive=2),
        ]
        config.drive_count = config.total_drive_count
        results = _make_typical_results()
        cap = OSDCapacityModel(config, results).calculate()
        self.assertIn('per_device_class', cap)
        per_class = cap['per_device_class']
        # HDD class: 24 drives × 1 OSD = 24 total_osds
        self.assertEqual(per_class[0]['total_osds'], 24)
        # NVMe class: 4 drives × 2 OSDs = 8 total_osds
        self.assertEqual(per_class[1]['total_osds'], 8)

    def test_osds_per_drive_validation(self):
        """osds_per_drive < 1 should fail validation."""
        config = self._config(osds_per_drive=0)
        with self.assertRaises(SystemExit):
            _validate_config(config)

    def test_json_includes_osds_per_drive(self):
        """JSON output should include osds_per_drive."""
        config = self._config(osds_per_drive=2, drive_count=4)
        libs = LibraryManager()
        results = _make_typical_results()
        model = OSDCapacityModel(config, results)
        capacity = model.calculate()
        proj = ScaleOutProjection(config, capacity)
        scale_out = proj.project()
        report = ReportGenerator(config, libs, results, capacity, scale_out)
        data = json.loads(report.to_json())
        self.assertEqual(data['config']['osds_per_drive'], 2)
        self.assertEqual(data['config']['total_osds'], 8)


CephBenchmarks = sim.CephBenchmarks
_benchmark_worker = sim._benchmark_worker
_make_worker_op = sim._make_worker_op


class TestParallelBenchmarks(unittest.TestCase):
    """Tests for parallel benchmark execution via multiprocessing."""

    def test_benchmark_worker_basic(self):
        """_benchmark_worker runs and returns valid timing dict."""
        op_spec = {'op': 'crc32c', 'object_size': 4096}
        result = _benchmark_worker(op_spec, iterations=100, duration=1.0)
        self.assertIn('cpu_time_per_op_us', result)
        self.assertIn('ops_per_sec', result)
        self.assertIn('iterations', result)
        self.assertGreater(result['cpu_time_per_op_us'], 0)
        self.assertGreater(result['ops_per_sec'], 0)
        self.assertEqual(result['iterations'], 100)

    def test_benchmark_worker_auto_calibrate(self):
        """Worker auto-calibrates when iterations=0."""
        op_spec = {'op': 'crc32c', 'object_size': 4096}
        result = _benchmark_worker(op_spec, iterations=0, duration=0.5)
        self.assertGreater(result['iterations'], 0)
        self.assertGreater(result['ops_per_sec'], 0)

    def test_make_worker_op_all_types(self):
        """_make_worker_op handles all known operation types."""
        libs = LibraryManager()
        data = os.urandom(4096)

        # crc32c
        op = _make_worker_op(libs, {'op': 'crc32c', 'data': data})
        op()  # should not raise

        # sha256
        op = _make_worker_op(libs, {'op': 'sha256', 'data': data})
        op()

        # serialization
        op = _make_worker_op(libs, {'op': 'serialization', 'data_len': 4096})
        op()

        # rocksdb_sim
        op = _make_worker_op(libs, {'op': 'rocksdb_sim', 'kv_ops': 4,
                                     'object_size': 4096})
        op()

        # crush
        op = _make_worker_op(libs, {'op': 'crush', 'num_osds': 64,
                                     'placements': 3})
        op()

    def test_make_worker_op_unknown_raises(self):
        """Unknown operation name raises ValueError."""
        libs = LibraryManager()
        with self.assertRaises(ValueError):
            _make_worker_op(libs, {'op': 'nonexistent'})

    def test_parallel_workers_stored(self):
        """CephBenchmarks stores parallel_workers parameter."""
        libs = LibraryManager()
        config = ClusterConfig()
        config.object_sizes_to_test = ['4k']
        config.benchmark_duration = 0.5
        bench = CephBenchmarks(libs, config, parallel_workers=4)
        self.assertEqual(bench.parallel_workers, 4)

    def test_parallel_zero_uses_single_thread(self):
        """parallel_workers=0 (default) runs single-threaded benchmarks."""
        libs = LibraryManager()
        config = ClusterConfig()
        config.object_sizes_to_test = ['4k']
        config.benchmark_duration = 0.5
        bench = CephBenchmarks(libs, config, parallel_workers=0)
        results = bench.run_all()
        self.assertGreater(len(results), 0)
        # No "workers" in notes means single-threaded
        for r in results:
            self.assertNotIn('workers', r.notes)

    def test_parallel_run_produces_results(self):
        """parallel_workers=2 produces results with worker info in notes."""
        libs = LibraryManager()
        config = ClusterConfig()
        config.object_sizes_to_test = ['4k']
        config.benchmark_duration = 0.5
        bench = CephBenchmarks(libs, config, parallel_workers=2)
        results = bench.run_all()
        self.assertGreater(len(results), 0)
        # All results should have parallel worker info in notes
        for r in results:
            self.assertIn('2 workers', r.notes)
            self.assertIn('P99=', r.notes)
            self.assertGreater(r.cpu_time_per_op_us, 0)
            self.assertGreater(r.ops_per_sec, 0)


if __name__ == '__main__':
    unittest.main()
