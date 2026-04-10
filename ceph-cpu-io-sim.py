#!/usr/bin/env python3

# ceph-cpu-io-sim.py - Ceph CPU IO Simulator
# Copyright (C) 2026
# License: Apache-2.0 (see LICENSE)

"""
Ceph CPU IO Simulator - Benchmarks CPU capacity for Ceph OSD workloads.

Measures how many OSDs a given CPU can support by running real Ceph-like
CPU operations (checksumming, compression, erasure coding, serialization,
metadata ops) and modeling the per-IO CPU cost at various drive speeds.

Works with zero optional dependencies (stdlib only) but produces more
accurate results when Ceph libraries are available.
"""

import argparse
import binascii
import csv
import ctypes
import ctypes.util
import hashlib
import json
import math
import multiprocessing
import os
import platform
import statistics
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

VERSION = "1.0.0"

OBJECT_SIZES = {
    '4k': 4096,
    '8k': 8192,
    '16k': 16384,
    '32k': 32768,
    '64k': 65536,
    '128k': 131072,
    '256k': 262144,
    '512k': 524288,
    '1m': 1048576,
    '4m': 4194304,
    '16m': 16777216,
}

DRIVE_PROFILES = {
    'hdd': {
        'random_iops_min': 100,
        'random_iops_max': 200,
        'random_iops_typical': 150,
        'seq_throughput_mb': 175,
        'latency_ms': 5.0,
    },
    'ssd': {
        'random_iops_min': 10000,
        'random_iops_max': 100000,
        'random_iops_typical': 50000,
        'seq_throughput_mb': 550,
        'latency_ms': 0.1,
    },
    'nvme': {
        'random_iops_min': 100000,
        'random_iops_max': 1000000,
        'random_iops_typical': 500000,
        'seq_throughput_mb': 3500,
        'latency_ms': 0.02,
    },
}

BLUESTORE_OVERHEAD = {
    'wal_db_same_device': 1.15,
    'wal_db_separate': 1.05,
    'rocksdb_compaction': 1.10,
}

SCRUB_FREQUENCY_OVERHEAD = {
    'daily': 0.05,
    'weekly': 0.01,
    'disabled': 0.0,
}


# ---------------------------------------------------------------------------
# Library Detection and Wrapping
# ---------------------------------------------------------------------------

class LibraryManager:
    """Detects available Ceph-related libraries and provides unified wrappers."""

    def __init__(self):
        self.available = {}
        self._liblz4 = None
        self._libzstd = None
        self._libsnappy = None
        self._crc32c_fn = None
        self._ec_driver = None
        self._kv_store = {}
        self._detect_libraries()

    def _detect_libraries(self):
        self._detect_cpu_features()
        self._detect_crc32c()
        self._detect_lz4()
        self._detect_zstd()
        self._detect_snappy()
        self._detect_erasure_coding()
        self._detect_rocksdb()
        self.available['zlib'] = 'stdlib'
        self.available['sha256'] = 'hashlib'

    def _detect_cpu_features(self):
        """Detect CPU features relevant to Ceph performance."""
        self.cpu_has_sse42 = False
        self.cpu_has_avx2 = False
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('flags'):
                        flags = line.split(':')[1] if ':' in line else ''
                        self.cpu_has_sse42 = 'sse4_2' in flags
                        self.cpu_has_avx2 = 'avx2' in flags
                        break
        except (IOError, Exception):
            pass

    @property
    def has_hw_crc32c(self) -> bool:
        """Check if we're using hardware-accelerated CRC32C."""
        crc_impl = self.available.get('crc32c', '')
        return crc_impl in ('crcmod', 'crc32c', 'ctypes_isal', 'ctypes_crc32c')

    def get_warnings(self) -> List[str]:
        """Return warnings about suboptimal library detection."""
        warnings = []
        if not self.has_hw_crc32c:
            msg = ("CRC32C: Using zlib.crc32 fallback (IEEE polynomial, "
                   "NOT hardware-accelerated CRC32C). Results will "
                   "SIGNIFICANTLY overestimate CPU cost (~5-10x).")
            if self.cpu_has_sse42:
                msg += ("\n   Your CPU supports SSE4.2 hardware CRC32C! "
                        "Install one of:\n"
                        "     pip install crc32c    (recommended)\n"
                        "     pip install crcmod    (alternative)\n"
                        "     dnf install libisal   (ISA-L library)")
            warnings.append(msg)
        ec_impl = self.available.get('erasure_coding', '')
        if ec_impl == 'xor_simulation':
            warnings.append(
                "Erasure coding: Using XOR simulation. EC encode/decode "
                "benchmarks will not reflect real Reed-Solomon performance. "
                "Install pyeclib for accurate EC benchmarks.")
        return warnings

    # -- CRC32C --
    def _detect_crc32c(self):
        # Tier 1: Python packages (hardware-accelerated)
        try:
            import crcmod
            self._crc32c_fn = crcmod.predefined.mkCrcFun('crc-32c')
            self.available['crc32c'] = 'crcmod'
            return
        except (ImportError, Exception):
            pass
        try:
            import crc32c as _crc32c
            self._crc32c_fn = _crc32c.crc32c
            self.available['crc32c'] = 'crc32c'
            return
        except (ImportError, Exception):
            pass

        # Tier 2: ctypes - try system libraries that provide CRC32C
        # These use hardware SSE4.2/ARMv8 instructions when available
        for lib_name, candidates, setup_fn in [
            ('isal', ['libisal.so.2', 'libisal.so'], '_setup_isal_crc32c'),
            ('crc32c', ['libcrc32c.so.1', 'libcrc32c.so'], '_setup_libcrc32c'),
        ]:
            try:
                path = ctypes.util.find_library(lib_name)
                if path is None:
                    for candidate in candidates:
                        try:
                            ctypes.CDLL(candidate)
                            path = candidate
                            break
                        except OSError:
                            continue
                if path:
                    lib = ctypes.CDLL(path)
                    fn = getattr(self, setup_fn)(lib)
                    if fn:
                        self._crc32c_fn = fn
                        self.available['crc32c'] = f'ctypes_{lib_name}'
                        return
            except (OSError, Exception):
                pass

        # Tier 3: Fallback to zlib.crc32 (WARNING: different polynomial,
        # not hardware-accelerated -- results will overestimate CPU cost)
        self._crc32c_fn = zlib.crc32
        self.available['crc32c'] = 'zlib_crc32'

    def _setup_isal_crc32c(self, lib):
        """Setup Intel ISA-L crc32_iscsi (CRC32C Castagnoli)."""
        try:
            # isal provides crc32_iscsi which uses CRC32C polynomial
            fn = lib.crc32_iscsi
            fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
            fn.restype = ctypes.c_uint
            # Verify it works
            test = fn(b'hello', 5, 0)
            if isinstance(test, int):
                def crc32c_isal(data):
                    return fn(data, len(data), 0)
                return crc32c_isal
        except (AttributeError, Exception):
            pass
        return None

    def _setup_libcrc32c(self, lib):
        """Setup Google crc32c library."""
        try:
            # Try crc32c_extend (Google CRC32C library)
            fn = lib.crc32c_extend
            fn.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_size_t]
            fn.restype = ctypes.c_uint
            test = fn(0, b'hello', 5)
            if isinstance(test, int):
                def crc32c_google(data):
                    return fn(0, data, len(data))
                return crc32c_google
        except (AttributeError, Exception):
            pass
        try:
            # Alternative API: crc32c_value
            fn = lib.crc32c_value
            fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            fn.restype = ctypes.c_uint
            test = fn(b'hello', 5)
            if isinstance(test, int):
                def crc32c_val(data):
                    return fn(data, len(data))
                return crc32c_val
        except (AttributeError, Exception):
            pass
        return None

    # -- LZ4 --
    def _detect_lz4(self):
        try:
            import lz4.block
            self.available['lz4'] = 'python_lz4'
            return
        except (ImportError, Exception):
            pass
        try:
            path = ctypes.util.find_library('lz4')
            if path is None:
                for candidate in ['liblz4.so.1', 'liblz4.so', 'liblz4.dylib']:
                    try:
                        lib = ctypes.CDLL(candidate)
                        path = candidate
                        break
                    except OSError:
                        continue
            if path:
                lib = ctypes.CDLL(path)
                lib.LZ4_compress_default.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
                lib.LZ4_compress_default.restype = ctypes.c_int
                lib.LZ4_compressBound.argtypes = [ctypes.c_int]
                lib.LZ4_compressBound.restype = ctypes.c_int
                lib.LZ4_decompress_safe.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
                lib.LZ4_decompress_safe.restype = ctypes.c_int
                self._liblz4 = lib
                self.available['lz4'] = 'ctypes'
                return
        except (OSError, Exception):
            pass
        self.available['lz4'] = None

    # -- ZSTD --
    def _detect_zstd(self):
        try:
            import pyzstd
            self.available['zstd'] = 'pyzstd'
            return
        except (ImportError, Exception):
            pass
        try:
            import zstandard
            self.available['zstd'] = 'zstandard'
            return
        except (ImportError, Exception):
            pass
        try:
            path = ctypes.util.find_library('zstd')
            if path is None:
                for candidate in ['libzstd.so.1', 'libzstd.so', 'libzstd.dylib']:
                    try:
                        lib = ctypes.CDLL(candidate)
                        path = candidate
                        break
                    except OSError:
                        continue
            if path:
                lib = ctypes.CDLL(path)
                lib.ZSTD_compress.argtypes = [
                    ctypes.c_void_p, ctypes.c_size_t,
                    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                lib.ZSTD_compress.restype = ctypes.c_size_t
                lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
                lib.ZSTD_compressBound.restype = ctypes.c_size_t
                lib.ZSTD_decompress.argtypes = [
                    ctypes.c_void_p, ctypes.c_size_t,
                    ctypes.c_void_p, ctypes.c_size_t]
                lib.ZSTD_decompress.restype = ctypes.c_size_t
                lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
                lib.ZSTD_isError.restype = ctypes.c_uint
                self._libzstd = lib
                self.available['zstd'] = 'ctypes'
                return
        except (OSError, Exception):
            pass
        self.available['zstd'] = None

    # -- Snappy --
    def _detect_snappy(self):
        try:
            import snappy
            self.available['snappy'] = 'python_snappy'
            return
        except (ImportError, Exception):
            pass
        try:
            path = ctypes.util.find_library('snappy')
            if path is None:
                for candidate in ['libsnappy.so.1', 'libsnappy.so', 'libsnappy.dylib']:
                    try:
                        lib = ctypes.CDLL(candidate)
                        path = candidate
                        break
                    except OSError:
                        continue
            if path:
                lib = ctypes.CDLL(path)
                self._libsnappy = lib
                self.available['snappy'] = 'ctypes'
                return
        except (OSError, Exception):
            pass
        self.available['snappy'] = None

    # -- Erasure Coding --
    def _detect_erasure_coding(self):
        try:
            from pyeclib.ec_iface import ECDriver
            self.available['erasure_coding'] = 'pyeclib'
            return
        except (ImportError, Exception):
            pass
        try:
            import reedsolo
            self.available['erasure_coding'] = 'reedsolo'
            return
        except (ImportError, Exception):
            pass
        self.available['erasure_coding'] = 'xor_simulation'

    # -- RocksDB --
    def _detect_rocksdb(self):
        try:
            import rocksdb
            self.available['rocksdb'] = 'python_rocksdb'
            return
        except (ImportError, Exception):
            pass
        try:
            import plyvel
            self.available['rocksdb'] = 'plyvel'
            return
        except (ImportError, Exception):
            pass
        self.available['rocksdb'] = 'dict_simulation'

    def summary(self) -> str:
        lines = ["=== Library Detection ==="]
        crc_label = {
            'crcmod': 'crcmod (hardware-accelerated CRC32C)',
            'crc32c': 'crc32c package (Castagnoli)',
            'ctypes_isal': 'ISA-L (hardware CRC32C via ctypes)',
            'ctypes_crc32c': 'libcrc32c (hardware CRC32C via ctypes)',
            'zlib_crc32': ('zlib.crc32 (NOT CRC32C - ~10x slower than '
                           'hardware! Install crcmod or crc32c package)'),
        }
        lines.append(f"CRC32C:        {crc_label.get(self.available.get('crc32c', ''), 'unknown')}")

        comp_parts = []
        for algo in ['lz4', 'zstd', 'snappy', 'zlib']:
            status = self.available.get(algo)
            if status is None:
                comp_parts.append(f"{algo}: NOT AVAILABLE")
            elif status == 'stdlib':
                comp_parts.append(f"{algo} (stdlib)")
            else:
                comp_parts.append(f"{algo} ({status})")
        lines.append(f"Compression:   {', '.join(comp_parts)}")

        ec_label = {
            'pyeclib': 'pyeclib (ISA-L/Jerasure)',
            'reedsolo': 'reedsolo (pure Python Reed-Solomon)',
            'xor_simulation': 'XOR simulation (install pyeclib for real RS)',
        }
        lines.append(f"Erasure Code:  {ec_label.get(self.available.get('erasure_coding', ''), 'unknown')}")

        kv_label = {
            'python_rocksdb': 'python-rocksdb (native)',
            'plyvel': 'plyvel (LevelDB)',
            'dict_simulation': 'dict simulation',
        }
        lines.append(f"RocksDB:       {kv_label.get(self.available.get('rocksdb', ''), 'unknown')}")
        lines.append(f"SHA256:        hashlib (OpenSSL-backed)")
        return '\n'.join(lines)

    def crc32c(self, data: bytes) -> int:
        return self._crc32c_fn(data)

    def compress(self, algorithm: str, data: bytes, level: int = -1) -> bytes:
        if algorithm == 'zlib':
            return zlib.compress(data, level if level >= 0 else 5)

        if algorithm == 'lz4':
            impl = self.available.get('lz4')
            if impl == 'python_lz4':
                import lz4.block
                return lz4.block.compress(data)
            elif impl == 'ctypes' and self._liblz4:
                src = data
                bound = self._liblz4.LZ4_compressBound(len(src))
                dst = ctypes.create_string_buffer(bound)
                result = self._liblz4.LZ4_compress_default(
                    src, dst, len(src), bound)
                if result <= 0:
                    raise RuntimeError("LZ4 compression failed")
                return dst.raw[:result]
            else:
                raise RuntimeError("lz4 not available")

        if algorithm == 'zstd':
            impl = self.available.get('zstd')
            if impl == 'pyzstd':
                import pyzstd
                return pyzstd.compress(data, level if level >= 0 else 1)
            elif impl == 'zstandard':
                import zstandard
                cctx = zstandard.ZstdCompressor(level=level if level >= 0 else 1)
                return cctx.compress(data)
            elif impl == 'ctypes' and self._libzstd:
                src = data
                bound = self._libzstd.ZSTD_compressBound(len(src))
                dst = ctypes.create_string_buffer(bound)
                result = self._libzstd.ZSTD_compress(
                    dst, bound, src, len(src), level if level >= 0 else 1)
                if self._libzstd.ZSTD_isError(result):
                    raise RuntimeError("ZSTD compression failed")
                return bytes(dst.raw[:result])
            else:
                raise RuntimeError("zstd not available")

        if algorithm == 'snappy':
            impl = self.available.get('snappy')
            if impl == 'python_snappy':
                import snappy
                return snappy.compress(data)
            elif impl == 'ctypes' and self._libsnappy:
                raise RuntimeError("snappy ctypes compress not fully implemented")
            else:
                raise RuntimeError("snappy not available")

        raise ValueError(f"Unknown compression algorithm: {algorithm}")

    def decompress(self, algorithm: str, data: bytes,
                   original_size: int = 0) -> bytes:
        if algorithm == 'zlib':
            return zlib.decompress(data)

        if algorithm == 'lz4':
            impl = self.available.get('lz4')
            if impl == 'python_lz4':
                import lz4.block
                return lz4.block.decompress(data, uncompressed_size=original_size)
            elif impl == 'ctypes' and self._liblz4:
                dst = ctypes.create_string_buffer(original_size)
                result = self._liblz4.LZ4_decompress_safe(
                    data, dst, len(data), original_size)
                if result < 0:
                    raise RuntimeError("LZ4 decompression failed")
                return dst.raw[:result]
            else:
                raise RuntimeError("lz4 not available")

        if algorithm == 'zstd':
            impl = self.available.get('zstd')
            if impl == 'pyzstd':
                import pyzstd
                return pyzstd.decompress(data)
            elif impl == 'zstandard':
                import zstandard
                dctx = zstandard.ZstdDecompressor()
                return dctx.decompress(data, max_output_size=original_size or
                                       len(data) * 20)
            elif impl == 'ctypes' and self._libzstd:
                out_size = original_size or len(data) * 10
                dst = ctypes.create_string_buffer(out_size)
                result = self._libzstd.ZSTD_decompress(
                    dst, out_size, data, len(data))
                if self._libzstd.ZSTD_isError(result):
                    raise RuntimeError("ZSTD decompression failed")
                return bytes(dst.raw[:result])
            else:
                raise RuntimeError("zstd not available")

        if algorithm == 'snappy':
            impl = self.available.get('snappy')
            if impl == 'python_snappy':
                import snappy
                return snappy.decompress(data)
            else:
                raise RuntimeError("snappy not available")

        raise ValueError(f"Unknown decompression algorithm: {algorithm}")

    def ec_encode(self, data: bytes, k: int, m: int) -> List[bytes]:
        impl = self.available.get('erasure_coding')
        if impl == 'pyeclib':
            from pyeclib.ec_iface import ECDriver
            if self._ec_driver is None:
                self._ec_driver = ECDriver(k=k, m=m,
                                           ec_type='liberasurecode_rs_vand')
            return self._ec_driver.encode(data)
        elif impl == 'reedsolo':
            return self._ec_encode_reedsolo(data, k, m)
        else:
            return self._ec_encode_xor(data, k, m)

    def ec_decode(self, chunks: List[bytes], k: int, m: int,
                  missing: List[int]) -> bytes:
        impl = self.available.get('erasure_coding')
        if impl == 'pyeclib':
            from pyeclib.ec_iface import ECDriver
            if self._ec_driver is None:
                self._ec_driver = ECDriver(k=k, m=m,
                                           ec_type='liberasurecode_rs_vand')
            available = [c for i, c in enumerate(chunks) if i not in missing]
            return self._ec_driver.decode(available)
        elif impl == 'reedsolo':
            return self._ec_decode_reedsolo(chunks, k, m, missing)
        else:
            return self._ec_decode_xor(chunks, k, m, missing)

    @staticmethod
    def _xor_blocks(a: bytes, b: bytes) -> bytes:
        """XOR two byte strings efficiently using array module."""
        import array
        # Process 8 bytes at a time using unsigned long long
        padded_len = (len(a) + 7) & ~7
        a_padded = a.ljust(padded_len, b'\x00')
        b_padded = b.ljust(padded_len, b'\x00')
        a_arr = array.array('Q')
        a_arr.frombytes(a_padded)
        b_arr = array.array('Q')
        b_arr.frombytes(b_padded)
        result = array.array('Q', (x ^ y for x, y in zip(a_arr, b_arr)))
        return result.tobytes()[:len(a)]

    def _ec_encode_xor(self, data: bytes, k: int, m: int) -> List[bytes]:
        chunk_size = len(data) // k
        remainder = len(data) % k
        if remainder != 0:
            data += b'\x00' * (k - remainder)
            chunk_size = len(data) // k
        chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(k)]
        parity_chunks = []
        for p in range(m):
            parity = b'\x00' * chunk_size
            for chunk in chunks:
                parity = self._xor_blocks(parity, chunk)
            # For additional parity chunks, rotate data to simulate
            # Galois field multiplication cost
            if p > 0:
                for i, chunk in enumerate(chunks):
                    shift = (p * i * 8) % (chunk_size * 8)
                    shift_bytes = shift // 8
                    if shift_bytes > 0 and shift_bytes < chunk_size:
                        rotated = chunk[shift_bytes:] + chunk[:shift_bytes]
                        parity = self._xor_blocks(parity, rotated)
            parity_chunks.append(parity)
        return chunks + parity_chunks

    def _ec_decode_xor(self, chunks: List[bytes], k: int, m: int,
                       missing: List[int]) -> bytes:
        chunk_size = len(chunks[0]) if chunks else 0
        if len(missing) == 0:
            return b''.join(chunks[:k])
        if len(missing) == 1 and missing[0] < k:
            data_indices = [i for i in range(k) if i != missing[0]]
            parity = chunks[k] if k < len(chunks) else b'\x00' * chunk_size
            recovered = parity
            for idx in data_indices:
                if idx < len(chunks):
                    recovered = self._xor_blocks(recovered, chunks[idx])
            result_chunks = []
            for i in range(k):
                if i == missing[0]:
                    result_chunks.append(recovered)
                else:
                    result_chunks.append(chunks[i])
            return b''.join(result_chunks)
        return b''.join(chunks[:k])

    def _ec_encode_reedsolo(self, data: bytes, k: int, m: int) -> List[bytes]:
        import reedsolo
        rs = reedsolo.RSCodec(m)
        chunk_size = len(data) // k
        if len(data) % k != 0:
            data += b'\x00' * (k - len(data) % k)
            chunk_size = len(data) // k
        chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(k)]
        parity_chunks = []
        for p in range(m):
            parity = bytearray(chunk_size)
            for i, chunk in enumerate(chunks):
                encoded = rs.encode(chunk)
                parity_part = encoded[len(chunk):]
                if p < len(parity_part):
                    for j in range(chunk_size):
                        parity[j] ^= parity_part[j % len(parity_part)]
            parity_chunks.append(bytes(parity))
        return chunks + parity_chunks

    def _ec_decode_reedsolo(self, chunks: List[bytes], k: int, m: int,
                            missing: List[int]) -> bytes:
        return self._ec_decode_xor(chunks, k, m, missing)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    operation: str
    object_size: int
    ops_per_sec: float
    cpu_time_per_op_us: float
    throughput_mbps: float
    cpu_utilization: float
    iterations: int
    elapsed_sec: float
    library_used: str
    notes: str = ''


@dataclass
class DeviceClass:
    """Represents one class of drives in a mixed-media configuration."""
    drive_type: str = 'hdd'
    count: int = 12
    iops_override: int = 0
    osds_per_drive: int = 1

    @property
    def total_osds(self) -> int:
        """Total OSD daemons for this device class."""
        return self.count * self.osds_per_drive

    def get_iops(self, scenario: str = 'typical') -> int:
        """Get per-OSD IOPS (drive IOPS split across OSDs on same drive)."""
        if self.iops_override > 0:
            raw_iops = self.iops_override
        else:
            profile = DRIVE_PROFILES[self.drive_type]
            if scenario == 'best':
                raw_iops = profile['random_iops_min']
            elif scenario == 'worst':
                raw_iops = profile['random_iops_max']
            else:
                raw_iops = profile['random_iops_typical']
        # When multiple OSDs share a drive, each OSD gets a fraction
        # of the drive's IOPS capacity
        return raw_iops // self.osds_per_drive if self.osds_per_drive > 1 else raw_iops

    def get_drive_iops(self, scenario: str = 'typical') -> int:
        """Get raw per-drive IOPS (before splitting across OSDs)."""
        if self.iops_override > 0:
            return self.iops_override
        profile = DRIVE_PROFILES[self.drive_type]
        if scenario == 'best':
            return profile['random_iops_min']
        elif scenario == 'worst':
            return profile['random_iops_max']
        return profile['random_iops_typical']


@dataclass
class ClusterConfig:
    """Represents the user's theoretical cluster configuration."""
    cpu_cores: int = 0
    cpu_cores_for_ceph: float = 0.0
    cpu_model: str = ''

    drive_type: str = 'hdd'
    drive_count: int = 12
    drive_iops: int = 0
    drive_throughput_mb: int = 0
    osds_per_drive: int = 1

    # Mixed-media support: list of device classes.
    # When populated, overrides drive_type / drive_count / drive_iops.
    device_classes: List['DeviceClass'] = field(default_factory=list)

    protection_type: str = 'replicated'
    replica_count: int = 3
    ec_k: int = 4
    ec_m: int = 2

    compression_enabled: bool = False
    compression_algorithm: str = 'zstd'
    compression_ratio: float = 0.5
    compression_mode: str = 'passive'

    wal_db_separate: bool = False

    recovery_osds: int = 0

    object_size: str = '4m'
    workload_pattern: str = 'mixed'
    read_write_ratio: float = 0.7

    scrub_frequency: str = 'daily'

    scenario: str = 'typical'

    benchmark_duration: float = 5.0
    object_sizes_to_test: List[str] = field(
        default_factory=lambda: ['4k', '64k', '128k', '4m'])

    def get_object_size_bytes(self) -> int:
        return OBJECT_SIZES.get(self.object_size, 4194304)

    def get_drive_iops(self) -> int:
        """Get per-OSD IOPS (split across OSDs sharing a drive)."""
        if self.drive_iops > 0:
            raw = self.drive_iops
        else:
            profile = DRIVE_PROFILES[self.drive_type]
            if self.scenario == 'best':
                raw = profile['random_iops_min']
            elif self.scenario == 'worst':
                raw = profile['random_iops_max']
            else:
                raw = profile['random_iops_typical']
        if self.osds_per_drive > 1:
            return raw // self.osds_per_drive
        return raw

    def get_drive_throughput_mb(self) -> int:
        if self.drive_throughput_mb > 0:
            return self.drive_throughput_mb
        return DRIVE_PROFILES[self.drive_type]['seq_throughput_mb']

    @property
    def is_mixed_media(self) -> bool:
        return len(self.device_classes) > 1

    def get_effective_device_classes(self) -> List['DeviceClass']:
        """Return device classes (mixed-media or single-class fallback)."""
        if self.device_classes:
            return self.device_classes
        return [DeviceClass(
            drive_type=self.drive_type,
            count=self.drive_count,
            iops_override=self.drive_iops,
            osds_per_drive=self.osds_per_drive,
        )]

    @property
    def total_drive_count(self) -> int:
        if self.device_classes:
            return sum(dc.count for dc in self.device_classes)
        return self.drive_count

    @property
    def total_osd_count(self) -> int:
        """Total OSD daemons across all drives (accounts for multi-OSD)."""
        if self.device_classes:
            return sum(dc.total_osds for dc in self.device_classes)
        return self.drive_count * self.osds_per_drive


# ---------------------------------------------------------------------------
# Parallel Benchmark Worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _make_worker_op(libs: LibraryManager, op_spec: Dict[str, Any]):
    """Reconstruct a benchmark closure from a serializable spec.

    Each worker process calls this to build the callable that _run_timed
    would normally create inline.  The spec dict carries the operation
    name and the parameters needed to reconstruct it.
    """
    name = op_spec['op']
    data = op_spec.get('data', b'')
    obj_size = op_spec.get('object_size', 0)

    if name == 'crc32c':
        return lambda: libs.crc32c(data)

    if name == 'sha256':
        return lambda: hashlib.sha256(data).digest()

    if name == 'compress':
        algo = op_spec['algo']
        level = op_spec['level']
        return lambda: libs.compress(algo, data, level)

    if name == 'decompress':
        algo = op_spec['algo']
        compressed = op_spec['compressed']
        orig_size = op_spec['orig_size']
        return lambda: libs.decompress(algo, compressed, orig_size)

    if name == 'ec_encode':
        k, m = op_spec['k'], op_spec['m']
        return lambda: libs.ec_encode(data, k, m)

    if name == 'ec_decode':
        k, m = op_spec['k'], op_spec['m']
        chunks = op_spec['chunks']
        missing = op_spec['missing']
        return lambda: libs.ec_decode(chunks, k, m, missing)

    if name == 'serialization':
        crc_fn = libs.crc32c
        data_len = op_spec['data_len']
        def ser_op():
            header = struct.pack('<IIQQQII',
                                 0x0001, data_len, 0x12345678,
                                 0, data_len, 42, 0)
            crc_fn(header)
            frame_hdr = struct.pack('<IIQI', 0x0002, 0, data_len, 0)
            crc_fn(frame_hdr)
        return ser_op

    if name == 'rocksdb_sim':
        store = {}
        counter = [0]
        kv_ops = op_spec.get('kv_ops', 4)
        def rdb_op():
            for i in range(kv_ops):
                k = struct.pack('>QQ', counter[0], i)
                v = struct.pack('<QQII',
                                counter[0] * obj_size, obj_size,
                                0, zlib.crc32(k))
                store[k] = v
            counter[0] += 1
        return rdb_op

    if name == 'crush':
        num_osds = op_spec['num_osds']
        placements = op_spec['placements']
        seed_data = os.urandom(4096)
        seed_offset = [0]
        def crush_op():
            off = seed_offset[0] % (len(seed_data) - 4)
            seed_offset[0] += 4
            pg_id = struct.unpack_from('<I', seed_data, off)[0]
            for rep in range(placements):
                best_osd = -1
                best_hash = -1
                for osd in range(num_osds):
                    h = zlib.crc32(struct.pack('<III', pg_id, osd, rep))
                    if h > best_hash:
                        best_hash = h
                        best_osd = osd
        return crush_op

    if name == 'recovery_replicated':
        crc_fn = libs.crc32c
        replica_count = op_spec['replica_count']
        total_osds = op_spec['total_osds']
        store = {}
        counter = [0]
        def rec_rep_op():
            crc_fn(data)
            header = struct.pack('<IIQQQII',
                                 0x0002, len(data), 0x12345678,
                                 0, len(data), 42, 0)
            crc_fn(header)
            crc_fn(data)
            crc_fn(data)
            for i in range(6):
                k = struct.pack('>QQ', counter[0], i)
                v = struct.pack('<QQII', counter[0] * len(data),
                                len(data), 0, zlib.crc32(k))
                store[k] = v
            counter[0] += 1
            pg_id = struct.unpack('<I', data[:4])[0]
            for rep in range(replica_count):
                best = -1
                for osd in range(max(total_osds * 8, 64)):
                    h = zlib.crc32(struct.pack('<III', pg_id, osd, rep))
                    if h > best:
                        best = h
        return rec_rep_op

    if name == 'recovery_ec':
        crc_fn = libs.crc32c
        k, m = op_spec['k'], op_spec['m']
        chunks = op_spec['chunks']
        missing = op_spec['missing']
        total_osds = op_spec['total_osds']
        store = {}
        counter = [0]
        def rec_ec_op():
            for i in range(k + m):
                if i not in missing and i < len(chunks):
                    crc_fn(chunks[i])
            libs.ec_decode(chunks, k, m, missing)
            libs.ec_encode(data, k, m)
            crc_fn(chunks[0])
            for i in range(8):
                key = struct.pack('>QQ', counter[0], i)
                val = struct.pack('<QQII', counter[0] * len(data),
                                  len(data), 0, zlib.crc32(key))
                store[key] = val
            counter[0] += 1
            pg_id = struct.unpack('<I', data[:4])[0]
            for rep in range(k + m):
                best = -1
                for osd in range(max(total_osds * 8, 64)):
                    h = zlib.crc32(struct.pack('<III', pg_id, osd, rep))
                    if h > best:
                        best = h
        return rec_ec_op

    raise ValueError(f"Unknown operation: {name}")


def _benchmark_worker(op_spec: Dict[str, Any], iterations: int,
                      duration: float) -> Dict[str, Any]:
    """Worker process: runs one benchmark and returns timing results.

    Each worker creates its own LibraryManager (separate process =
    separate GIL, separate library handles). This is a module-level
    function so multiprocessing can pickle it.
    """
    libs = LibraryManager()

    # Reconstruct random data if needed (can't share across processes)
    obj_size = op_spec.get('object_size', 0)
    if obj_size > 0 and 'data' not in op_spec:
        op_spec['data'] = os.urandom(obj_size)

    op = _make_worker_op(libs, op_spec)

    # Warm up
    for _ in range(3):
        op()

    # If no pre-calibrated iteration count, calibrate locally
    if iterations <= 0:
        start = time.perf_counter()
        for _ in range(10):
            op()
        elapsed = time.perf_counter() - start
        per_op = elapsed / 10
        iterations = max(100, int(duration / per_op)) if per_op > 0 else 100000

    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    for _ in range(iterations):
        op()
    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    elapsed = max(end_wall - start_wall, 1e-9)
    cpu_time = max(end_cpu - start_cpu, 1e-9)

    return {
        'iterations': iterations,
        'elapsed_sec': elapsed,
        'cpu_time_sec': cpu_time,
        'ops_per_sec': iterations / elapsed,
        'cpu_time_per_op_us': (cpu_time / iterations) * 1e6,
        'throughput_mbps': (obj_size * iterations / elapsed / (1024 * 1024)
                            if obj_size > 0 else 0.0),
        'cpu_utilization': cpu_time / elapsed,
    }


# ---------------------------------------------------------------------------
# CPU Micro-Benchmarks
# ---------------------------------------------------------------------------

class CephBenchmarks:
    """Runs CPU micro-benchmarks simulating Ceph OSD operations."""

    def __init__(self, libs: LibraryManager, config: ClusterConfig,
                 verbose: bool = False, parallel_workers: int = 0):
        self.libs = libs
        self.config = config
        self.verbose = verbose
        self.parallel_workers = parallel_workers
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> List[BenchmarkResult]:
        self.results = []
        for size_name in self.config.object_sizes_to_test:
            size_bytes = OBJECT_SIZES[size_name]
            data = os.urandom(size_bytes)

            self._print_progress(f"  CRC32C @ {size_name}...")
            self._bench_crc32c(data, size_name)

            self._print_progress(f"  SHA256 @ {size_name}...")
            self._bench_sha256(data, size_name)

            if self.config.compression_enabled:
                for algo in self._get_available_compression_algos():
                    self._print_progress(f"  Compress {algo} @ {size_name}...")
                    self._bench_compress(data, size_name, algo)
                    self._print_progress(f"  Decompress {algo} @ {size_name}...")
                    self._bench_decompress(data, size_name, algo)

            if self.config.protection_type == 'erasure':
                self._print_progress(
                    f"  EC encode {self.config.ec_k}+{self.config.ec_m} "
                    f"@ {size_name}...")
                self._bench_ec_encode(data, size_name)
                self._print_progress(
                    f"  EC decode {self.config.ec_k}+{self.config.ec_m} "
                    f"@ {size_name}...")
                self._bench_ec_decode(data, size_name)

            self._print_progress(f"  Serialization @ {size_name}...")
            self._bench_serialization(data, size_name)

            self._print_progress(f"  RocksDB sim @ {size_name}...")
            self._bench_rocksdb_sim(size_name)

        self._print_progress("  CRUSH calculation...")
        self._bench_crush_calculation()

        if self.config.recovery_osds > 0:
            # Recovery always operates on full RADOS objects (typically 4M),
            # not at the IO size used for normal operations
            recovery_obj_size = 4194304  # 4M RADOS default
            recovery_data = os.urandom(recovery_obj_size)
            self._print_progress(
                f"  Recovery (full pipeline) @ 4m (RADOS object size)...")
            if self.config.protection_type == 'erasure':
                self._bench_recovery_ec(recovery_data)
            else:
                self._bench_recovery_replicated(recovery_data)

        return self.results

    def _print_progress(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _get_available_compression_algos(self) -> List[str]:
        algo = self.config.compression_algorithm
        if self.libs.available.get(algo) is not None:
            return [algo]
        available = []
        for a in ['lz4', 'zstd', 'snappy', 'zlib']:
            if self.libs.available.get(a) is not None:
                available.append(a)
        if available:
            print(f"Warning: {algo} not available, using: {available[0]}")
            return [available[0]]
        return ['zlib']

    def _calibrate_iterations(self, func, target_seconds=None) -> int:
        target = target_seconds or self.config.benchmark_duration
        warmup = 3
        for _ in range(warmup):
            func()
        start = time.perf_counter()
        for _ in range(10):
            func()
        elapsed = time.perf_counter() - start
        per_op = elapsed / 10
        if per_op <= 0:
            return 1000000
        return max(100, int(target / per_op))

    def _run_timed(self, name: str, func, object_size: int,
                   library_used: str, notes: str = '') -> BenchmarkResult:
        iterations = self._calibrate_iterations(func)

        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        for _ in range(iterations):
            func()
        end_wall = time.perf_counter()
        end_cpu = time.process_time()

        elapsed = end_wall - start_wall
        cpu_time = end_cpu - start_cpu

        if elapsed <= 0:
            elapsed = 1e-9
        if cpu_time <= 0:
            cpu_time = 1e-9

        ops_per_sec = iterations / elapsed
        cpu_time_per_op = (cpu_time / iterations) * 1e6
        throughput = (object_size * iterations / elapsed / (1024 * 1024)
                      if object_size > 0 else 0.0)
        cpu_util = cpu_time / elapsed

        result = BenchmarkResult(
            operation=name,
            object_size=object_size,
            ops_per_sec=ops_per_sec,
            cpu_time_per_op_us=cpu_time_per_op,
            throughput_mbps=throughput,
            cpu_utilization=cpu_util,
            iterations=iterations,
            elapsed_sec=elapsed,
            library_used=library_used,
            notes=notes,
        )
        self.results.append(result)
        return result

    def _run_parallel(self, name: str, op_spec: Dict[str, Any],
                      object_size: int, library_used: str,
                      notes: str = '') -> BenchmarkResult:
        """Run benchmark across parallel workers to measure contention.

        Each worker simulates one OSD running the same operation.
        Results are averaged to get per-OSD cost under contention.
        """
        n = self.parallel_workers

        # Calibrate in parent process, broadcast iteration count
        iterations = 0
        if object_size > 0 and 'data' not in op_spec:
            op_spec['data'] = os.urandom(object_size)
        try:
            op = _make_worker_op(self.libs, op_spec)
            iterations = self._calibrate_iterations(op)
        except Exception:
            pass

        # Build per-worker args (each worker regenerates its own data)
        worker_specs = []
        for _ in range(n):
            spec = dict(op_spec)
            # Remove data so each worker generates its own random buffer
            spec.pop('data', None)
            spec['object_size'] = object_size
            worker_specs.append(
                (spec, iterations, self.config.benchmark_duration))

        with multiprocessing.Pool(n) as pool:
            worker_results = pool.starmap(_benchmark_worker, worker_specs)

        # Aggregate: mean and P99
        cpu_times = [r['cpu_time_per_op_us'] for r in worker_results]
        ops_rates = [r['ops_per_sec'] for r in worker_results]
        throughputs = [r['throughput_mbps'] for r in worker_results]
        cpu_utils = [r['cpu_utilization'] for r in worker_results]
        iters = [r['iterations'] for r in worker_results]

        avg_cpu = statistics.mean(cpu_times)
        avg_ops = statistics.mean(ops_rates)
        avg_tp = statistics.mean(throughputs)
        avg_util = statistics.mean(cpu_utils)
        total_iters = sum(iters)

        p99_cpu = sorted(cpu_times)[int(len(cpu_times) * 0.99)] if len(cpu_times) > 1 else cpu_times[0]

        parallel_notes = (f'{notes}; ' if notes else '') + f'{n} workers, P99={p99_cpu:.2f}us'

        elapsed = statistics.mean([r['elapsed_sec'] for r in worker_results])

        result = BenchmarkResult(
            operation=name,
            object_size=object_size,
            ops_per_sec=avg_ops,
            cpu_time_per_op_us=avg_cpu,
            throughput_mbps=avg_tp,
            cpu_utilization=avg_util,
            iterations=total_iters,
            elapsed_sec=elapsed,
            library_used=library_used,
            notes=parallel_notes,
        )
        self.results.append(result)
        return result

    def _run_op(self, name: str, op_spec: Dict[str, Any],
                object_size: int, library_used: str,
                func=None, notes: str = '') -> BenchmarkResult:
        """Dispatch to parallel or single-threaded benchmark."""
        if self.parallel_workers >= 2:
            return self._run_parallel(
                name, op_spec, object_size, library_used, notes)
        else:
            return self._run_timed(name, func, object_size, library_used, notes)

    def _bench_crc32c(self, data: bytes, size_name: str):
        def op():
            self.libs.crc32c(data)
        op_spec = {'op': 'crc32c', 'object_size': len(data)}
        self._run_op(
            f'crc32c_{size_name}', op_spec, len(data),
            self.libs.available.get('crc32c', 'zlib'), func=op)

    def _bench_sha256(self, data: bytes, size_name: str):
        def op():
            hashlib.sha256(data).digest()
        op_spec = {'op': 'sha256', 'object_size': len(data)}
        self._run_op(
            f'sha256_{size_name}', op_spec, len(data), 'hashlib',
            func=op, notes='deep scrub verification')

    def _bench_compress(self, data: bytes, size_name: str, algo: str):
        comp_data = self._generate_compressible_data(
            len(data), self.config.compression_ratio)
        level = {'zstd': 1, 'zlib': 5, 'lz4': -1, 'snappy': -1}.get(algo, -1)

        def op():
            self.libs.compress(algo, comp_data, level)

        op_spec = {'op': 'compress', 'algo': algo, 'level': level,
                   'object_size': len(comp_data), 'data': comp_data}
        self._run_op(
            f'compress_{algo}_{size_name}', op_spec, len(comp_data),
            self.libs.available.get(algo, 'unknown'),
            func=op, notes=f'ratio={self.config.compression_ratio}')

    def _bench_decompress(self, data: bytes, size_name: str, algo: str):
        comp_data = self._generate_compressible_data(
            len(data), self.config.compression_ratio)
        level = {'zstd': 1, 'zlib': 5, 'lz4': -1, 'snappy': -1}.get(algo, -1)
        try:
            compressed = self.libs.compress(algo, comp_data, level)
        except Exception:
            return
        original_size = len(comp_data)

        def op():
            self.libs.decompress(algo, compressed, original_size)

        op_spec = {'op': 'decompress', 'algo': algo,
                   'compressed': compressed, 'orig_size': original_size,
                   'object_size': original_size}
        self._run_op(
            f'decompress_{algo}_{size_name}', op_spec, original_size,
            self.libs.available.get(algo, 'unknown'), func=op)

    def _bench_ec_encode(self, data: bytes, size_name: str):
        k, m = self.config.ec_k, self.config.ec_m

        def op():
            self.libs.ec_encode(data, k, m)

        op_spec = {'op': 'ec_encode', 'k': k, 'm': m,
                   'object_size': len(data)}
        self._run_op(
            f'ec_encode_{k}_{m}_{size_name}', op_spec, len(data),
            self.libs.available.get('erasure_coding', 'xor'),
            func=op, notes=f'k={k} m={m}')

    def _bench_ec_decode(self, data: bytes, size_name: str):
        k, m = self.config.ec_k, self.config.ec_m
        chunks = self.libs.ec_encode(data, k, m)
        missing = [k - 1]

        def op():
            self.libs.ec_decode(chunks, k, m, missing)

        op_spec = {'op': 'ec_decode', 'k': k, 'm': m,
                   'chunks': chunks, 'missing': missing,
                   'object_size': len(data)}
        self._run_op(
            f'ec_decode_{k}_{m}_{size_name}', op_spec, len(data),
            self.libs.available.get('erasure_coding', 'xor'),
            func=op, notes=f'k={k} m={m}, 1 missing chunk')

    def _bench_serialization(self, data: bytes, size_name: str):
        """Benchmark OSD replication message framing.

        Models the per-replica cost on the PRIMARY OSD for replication.
        In real Ceph, the primary:
        - Encodes an OSD op message header (~100-200 bytes)
        - CRC32Cs the header (not the data payload -- data goes via
          zero-copy sendmsg scatter-gather)
        - The RECEIVING replica CRC32Cs the full data, but that happens
          on the replica's CPU, not the primary's.

        So we benchmark: header encode + header CRC + small framing CRC.
        This replaces the old benchmark that incorrectly CRC32C'd the
        full data payload per replica.
        """
        crc_fn = self.libs.crc32c

        def op():
            # Encode OSD op message header (MOSDOp-like)
            header = struct.pack('<IIQQQII',
                                 0x0001, len(data), 0x12345678,
                                 0, len(data), 42, 0)
            crc_fn(header)
            # Messenger v2 frame: CRC32C of a ~60-byte frame header
            frame_hdr = struct.pack('<IIQI', 0x0002, 0, len(data), 0)
            crc_fn(frame_hdr)

        op_spec = {'op': 'serialization', 'data_len': len(data),
                   'object_size': len(data)}
        self._run_op(
            f'serialization_{size_name}', op_spec, len(data),
            self.libs.available.get('crc32c', 'zlib'),
            func=op, notes='per-replica message framing (header CRC only)')

    def _bench_rocksdb_sim(self, size_name: str):
        KV_OPS_PER_IO = 4
        obj_size = OBJECT_SIZES[size_name]
        store = {}
        counter = [0]

        def op():
            for i in range(KV_OPS_PER_IO):
                k = struct.pack('>QQ', counter[0], i)
                v = struct.pack('<QQII',
                                counter[0] * obj_size, obj_size,
                                0, zlib.crc32(k))
                store[k] = v
            counter[0] += 1

        op_spec = {'op': 'rocksdb_sim', 'kv_ops': KV_OPS_PER_IO,
                   'object_size': obj_size}
        self._run_op(
            f'rocksdb_sim_{size_name}', op_spec, 0,
            self.libs.available.get('rocksdb', 'dict_simulation'),
            func=op, notes=f'{KV_OPS_PER_IO} KV ops per IO')

    def _bench_crush_calculation(self):
        num_osds = max(self.config.total_osd_count * 8, 64)
        if self.config.protection_type == 'replicated':
            placements = self.config.replica_count
        else:
            placements = self.config.ec_k + self.config.ec_m

        seed_data = os.urandom(4096)
        seed_offset = [0]

        def op():
            off = seed_offset[0] % (len(seed_data) - 4)
            seed_offset[0] += 4
            pg_id = struct.unpack_from('<I', seed_data, off)[0]
            selected = []
            for rep in range(placements):
                best_osd = -1
                best_hash = -1
                for osd in range(num_osds):
                    h = zlib.crc32(struct.pack('<III', pg_id, osd, rep))
                    if h > best_hash:
                        best_hash = h
                        best_osd = osd
                selected.append(best_osd)

        op_spec = {'op': 'crush', 'num_osds': num_osds,
                   'placements': placements}
        self._run_op(
            'crush_calculation', op_spec, 0, 'simulation',
            func=op, notes=f'{placements} placements across {num_osds} OSDs')

    def _bench_recovery_replicated(self, data: bytes):
        """Benchmark full replicated recovery pipeline per object.

        Recovery for a replicated pool requires:
        1. Read object from surviving replica (CRC32C verify)
        2. Serialize for network (CRC32C header + payload)
        3. Write to new OSD location (CRC32C for BlueStore)
        4. RocksDB metadata updates on source and destination OSDs
        5. CRUSH recalculation for new PG mapping
        """
        crc_fn = self.libs.crc32c
        obj_size = len(data)
        store = {}
        counter = [0]

        def op():
            # 1. Read from surviving OSD: verify CRC
            crc_fn(data)
            # 2. Serialize replication message: header + CRC of payload
            header = struct.pack('<IIQQQII',
                                 0x0002, obj_size, 0x12345678,
                                 0, obj_size, 42, 0)
            crc_fn(header)
            crc_fn(data)
            # 3. Write to new OSD: CRC for BlueStore
            crc_fn(data)
            # 4. RocksDB metadata on source (mark PG migrating) +
            #    destination (new object entry) = ~6 KV ops total
            for i in range(6):
                k = struct.pack('>QQ', counter[0], i)
                v = struct.pack('<QQII', counter[0] * obj_size,
                                obj_size, 0, zlib.crc32(k))
                store[k] = v
            counter[0] += 1
            # 5. CRUSH lookup for new placement
            pg_id = struct.unpack('<I', data[:4])[0]
            for rep in range(self.config.replica_count):
                best = -1
                for osd in range(max(self.config.total_osd_count * 8, 64)):
                    h = zlib.crc32(struct.pack('<III', pg_id, osd, rep))
                    if h > best:
                        best = h

        op_spec = {'op': 'recovery_replicated',
                   'replica_count': self.config.replica_count,
                   'total_osds': self.config.total_osd_count,
                   'object_size': obj_size}
        self._run_op(
            'recovery_replicated', op_spec, obj_size,
            self.libs.available.get('crc32c', 'zlib'),
            func=op,
            notes=f'full pipeline: read+verify+serialize+write+metadata+CRUSH')

    def _bench_recovery_ec(self, data: bytes):
        """Benchmark full EC recovery pipeline per object.

        Recovery for an EC pool requires:
        1. Read k chunks from surviving OSDs (CRC32C each)
        2. EC decode to reconstruct missing chunk(s)
        3. EC re-encode to generate new parity if needed
        4. CRC32C the recovered chunk for BlueStore write
        5. RocksDB metadata updates (~8 KV ops: source PG states +
           destination extent maps)
        6. CRUSH recalculation
        """
        crc_fn = self.libs.crc32c
        k, m = self.config.ec_k, self.config.ec_m
        chunks = self.libs.ec_encode(data, k, m)
        missing = [0]
        obj_size = len(data)
        store = {}
        counter = [0]

        def op():
            # 1. Read k chunks from surviving OSDs, CRC each
            for i in range(k + m):
                if i not in missing and i < len(chunks):
                    crc_fn(chunks[i])
            # 2. EC decode (reconstruct missing)
            self.libs.ec_decode(chunks, k, m, missing)
            # 3. EC re-encode (rebuild parity)
            self.libs.ec_encode(data, k, m)
            # 4. CRC the recovered chunk for BlueStore
            crc_fn(chunks[0])
            # 5. RocksDB metadata (~8 KV ops for recovery)
            for i in range(8):
                key = struct.pack('>QQ', counter[0], i)
                val = struct.pack('<QQII', counter[0] * obj_size,
                                  obj_size, 0, zlib.crc32(key))
                store[key] = val
            counter[0] += 1
            # 6. CRUSH for new placement
            pg_id = struct.unpack('<I', data[:4])[0]
            for rep in range(k + m):
                best = -1
                for osd in range(max(self.config.total_osd_count * 8, 64)):
                    h = zlib.crc32(struct.pack('<III', pg_id, osd, rep))
                    if h > best:
                        best = h

        op_spec = {'op': 'recovery_ec',
                   'k': k, 'm': m,
                   'chunks': chunks, 'missing': missing,
                   'total_osds': self.config.total_osd_count,
                   'object_size': obj_size}
        self._run_op(
            'recovery_ec', op_spec, obj_size,
            self.libs.available.get('erasure_coding', 'xor'),
            func=op,
            notes=f'full pipeline: read_k+decode+re-encode+verify+metadata+CRUSH')

    @staticmethod
    def _generate_compressible_data(size: int, ratio: float) -> bytes:
        random_fraction = min(ratio * 1.5, 1.0)
        random_bytes = int(size * random_fraction)
        pattern_bytes = size - random_bytes
        return os.urandom(random_bytes) + (b'\x00' * pattern_bytes)


# ---------------------------------------------------------------------------
# OSD Capacity Model
# ---------------------------------------------------------------------------

class OSDCapacityModel:
    """Calculates how many OSDs a CPU can support based on benchmark results."""

    def __init__(self, config: ClusterConfig, results: List[BenchmarkResult],
                 libs: Optional[LibraryManager] = None):
        self.config = config
        self.results = results
        self.libs = libs
        self._cpu_costs: Dict[str, float] = {}
        self._crc32c_correction: float = 1.0

    def calculate(self) -> Dict[str, Any]:
        self._compute_per_io_cpu_cost()

        total_cpu_us_per_io = self._total_cpu_cost_per_io()
        available_cpu_us = self.config.cpu_cores_for_ceph * 1_000_000

        if total_cpu_us_per_io <= 0:
            total_cpu_us_per_io = 1.0

        overhead = self._compute_overhead_multiplier()

        # ---- Mixed-media per-class analysis ----
        device_classes = self.config.get_effective_device_classes()
        per_class = []
        total_cpu_used_us = 0.0

        for dc in device_classes:
            # Per-OSD IOPS (split across OSDs sharing a drive)
            dc_iops = dc.get_iops(self.config.scenario)
            cpu_per_osd_sec = total_cpu_us_per_io * dc_iops
            # Total CPU for ALL OSDs in this class
            cpu_total_class = dc.total_osds * cpu_per_osd_sec * overhead

            if cpu_per_osd_sec > 0:
                max_osds_raw = available_cpu_us / (cpu_per_osd_sec * overhead)
            else:
                max_osds_raw = float('inf')

            # Cap at total OSD daemons (drives × OSDs-per-drive)
            max_osds_adj = (min(math.floor(max_osds_raw), dc.total_osds)
                            if not math.isinf(max_osds_raw)
                            else dc.total_osds)

            entry = {
                'drive_type': dc.drive_type,
                'drive_count': dc.count,
                'osds_per_drive': dc.osds_per_drive,
                'total_osds': dc.total_osds,
                'drive_iops': dc_iops,
                'drive_iops_raw': dc.get_drive_iops(self.config.scenario),
                'cpu_us_per_osd_per_sec': cpu_per_osd_sec,
                'cpu_total_us_sec': cpu_total_class,
                'max_osds_standalone': max_osds_adj,
                'latency_ms': DRIVE_PROFILES[dc.drive_type]['latency_ms'],
            }
            per_class.append(entry)
            total_cpu_used_us += cpu_total_class

        # How many OSDs of each class can share the CPU simultaneously?
        #
        # Strategy: fulfill cheaper classes first (HDDs use much less CPU
        # per OSD than NVMe), then allocate remaining CPU to expensive
        # classes. This reflects how real clusters work -- you wouldn't
        # run NVMe OSDs at full random-IOPS on a node with 36 HDDs.
        #
        # Sort by CPU cost per OSD (ascending) so cheap classes get
        # their full allocation first.
        sorted_classes = sorted(
            enumerate(per_class), key=lambda x: x[1]['cpu_us_per_osd_per_sec'])

        remaining_cpu = available_cpu_us
        for idx, entry in sorted_classes:
            cpu_per_osd = entry['cpu_us_per_osd_per_sec'] * overhead
            if cpu_per_osd > 0:
                max_from_budget = remaining_cpu / cpu_per_osd
                osds = min(entry['total_osds'], math.floor(max_from_budget))
                osds = max(0, osds)
            else:
                osds = entry['total_osds']
            entry['max_osds_shared'] = osds
            entry['cpu_limited'] = osds < entry['total_osds']
            remaining_cpu -= osds * cpu_per_osd * (1 if cpu_per_osd > 0 else 0)

        total_shared_osds = sum(e['max_osds_shared'] for e in per_class)
        total_drives = sum(e['drive_count'] for e in per_class)
        total_osds_wanted = sum(e['total_osds'] for e in per_class)

        # For single-class backward compat, also compute simple numbers
        drive_iops = self.config.get_drive_iops()
        cpu_us_per_osd_per_sec = total_cpu_us_per_io * drive_iops
        if cpu_us_per_osd_per_sec > 0:
            max_osds = available_cpu_us / cpu_us_per_osd_per_sec
        else:
            max_osds = float('inf')
        max_osds_adjusted = max_osds / overhead
        if math.isinf(max_osds_adjusted):
            max_osds_adjusted_int = 999999
        else:
            max_osds_adjusted_int = math.floor(max_osds_adjusted)

        headroom = self._compute_headroom(max_osds_adjusted)
        if self.config.is_mixed_media:
            # Headroom based on combined usage
            if total_cpu_used_us > 0:
                headroom = ((available_cpu_us - total_cpu_used_us)
                            / available_cpu_us * 100)
            else:
                headroom = 100.0

        # Guard against inf from zero IOPS edge case
        if math.isinf(max_osds_adjusted):
            max_osds_adjusted_int = 999999
        else:
            max_osds_adjusted_int = math.floor(max_osds_adjusted)

        result = {
            'max_osds_raw': max_osds,
            'max_osds_adjusted': max_osds_adjusted_int,
            'cpu_us_per_io': total_cpu_us_per_io,
            'cpu_us_per_osd_per_sec': cpu_us_per_osd_per_sec,
            'overhead_multiplier': overhead,
            'drive_iops': drive_iops,
            'available_cpu_us': available_cpu_us,
            'per_operation_costs': dict(self._cpu_costs),
            'headroom_percentage': headroom,
            'crc32c_correction': self._crc32c_correction,
        }

        if self.config.is_mixed_media:
            result['per_device_class'] = per_class
            result['total_shared_osds'] = total_shared_osds
            result['total_drives'] = total_drives
            result['total_osds_wanted'] = total_osds_wanted

        if self.config.recovery_osds > 0:
            result['recovery'] = self._compute_recovery_impact(
                total_cpu_us_per_io, available_cpu_us, drive_iops, overhead)

        result['cpu_scaling'] = self._compute_cpu_scaling(
            total_cpu_us_per_io, drive_iops, overhead)

        return result

    def _compute_recovery_impact(self, normal_cpu_per_io: float,
                                  available_cpu_us: float,
                                  drive_iops: int,
                                  overhead: float) -> Dict[str, Any]:
        """Model CPU impact during OSD recovery.

        When an OSD fails, its PGs are redistributed. The surviving OSDs
        must:
        - Continue serving normal client IO
        - Additionally perform recovery IO (read + reconstruct + write)

        Recovery IO competes with client IO for CPU. The recovery CPU cost
        per object is higher than normal IO because it involves the full
        pipeline (read all replicas/chunks, verify, reconstruct, write,
        update metadata).
        """
        recovery_osds = self.config.recovery_osds
        total_osds = self.config.total_osd_count

        # Get the recovery benchmark result
        recovery_key = ('recovery_ec' if self.config.protection_type == 'erasure'
                        else 'recovery_replicated')
        recovery_cpu_per_obj = self._get_cost(recovery_key)
        if recovery_cpu_per_obj <= 0:
            recovery_cpu_per_obj = normal_cpu_per_io * 3  # rough estimate

        # PGs affected by the failed OSD(s)
        # In a cluster with N OSDs and 3x replication, each OSD holds
        # ~1/N of all PGs. When an OSD dies, those PGs need recovery.
        # The recovery work is spread across the remaining OSDs.
        surviving_osds = max(total_osds - recovery_osds, 1)

        # Each surviving OSD gets a share of the recovery work.
        # With replication, each PG recovery involves replica_count-1
        # surviving OSDs (one reads, others write). With EC, k OSDs read
        # and 1+ write.
        if self.config.protection_type == 'replicated':
            # Each OSD participates in recovery for PGs it hosts
            # Recovery load factor: fraction of PGs each surviving OSD
            # must help recover
            recovery_participation = (recovery_osds / surviving_osds)
        else:
            # EC: recovery requires reading k chunks from surviving OSDs
            # and writing m new parity/data chunks.  The load is spread
            # across survivors but each operation touches k+m OSDs.
            ec_width = self.config.ec_k + self.config.ec_m
            participation_ratio = min(ec_width / surviving_osds, 1.0)
            recovery_participation = (
                (recovery_osds / surviving_osds) * participation_ratio
                * (ec_width / max(self.config.ec_k, 1)))

        # Recovery IOPS per surviving OSD (Ceph rate-limits recovery,
        # but default osd_recovery_max_active=3 and
        # osd_max_backfills=1 per OSD)
        # At typical settings, each OSD does ~50-200 recovery ops/sec
        # for HDD, more for SSD/NVMe
        recovery_ops_per_sec = {
            'hdd': 50,
            'ssd': 200,
            'nvme': 500,
        }.get(self.config.drive_type, 50)

        # Scale by how many failed OSDs' PGs this OSD must help with
        effective_recovery_ops = recovery_ops_per_sec * recovery_participation

        # Total CPU cost per surviving OSD per second during recovery:
        # normal IO + recovery IO
        normal_cpu_per_osd = normal_cpu_per_io * drive_iops
        recovery_cpu_per_osd = recovery_cpu_per_obj * effective_recovery_ops
        total_cpu_per_osd = normal_cpu_per_osd + recovery_cpu_per_osd

        # Max OSDs during recovery
        if total_cpu_per_osd > 0:
            max_osds_recovery = available_cpu_us / (total_cpu_per_osd * overhead)
        else:
            max_osds_recovery = float('inf')

        # Client IO degradation: what fraction of CPU is left for client IO
        # after recovery takes its share
        recovery_cpu_fraction = (
            recovery_cpu_per_osd / (normal_cpu_per_osd + recovery_cpu_per_osd)
            if (normal_cpu_per_osd + recovery_cpu_per_osd) > 0 else 0)

        client_iops_fraction = 1.0 - recovery_cpu_fraction

        # Time to recover (rough estimate)
        # Recovery operates on full RADOS objects (4M default), not IO size.
        # Object count depends on drive capacity and RADOS object size.
        # A 4TB HDD with 4M RADOS objects at 70% usage: ~750K objects
        drive_capacity_tb = {'hdd': 4, 'ssd': 2, 'nvme': 2}.get(
            self.config.drive_type, 4)
        rados_obj_size = 4194304  # 4M RADOS default
        objects_per_osd = int(
            drive_capacity_tb * 1024 * 1024 * 1024 * 1024 * 0.7
            / max(rados_obj_size, 1))
        total_objects = objects_per_osd * recovery_osds

        # Recovery ops across all surviving OSDs
        cluster_recovery_ops = recovery_ops_per_sec * surviving_osds
        if cluster_recovery_ops > 0:
            recovery_time_sec = total_objects / cluster_recovery_ops
            recovery_time_hours = recovery_time_sec / 3600
        else:
            recovery_time_hours = float('inf')

        return {
            'failed_osds': recovery_osds,
            'surviving_osds': surviving_osds,
            'recovery_cpu_per_obj_us': recovery_cpu_per_obj,
            'recovery_ops_per_osd_sec': effective_recovery_ops,
            'normal_cpu_per_osd_us': normal_cpu_per_osd,
            'recovery_cpu_per_osd_us': recovery_cpu_per_osd,
            'total_cpu_per_osd_us': total_cpu_per_osd,
            'max_osds_during_recovery': math.floor(max_osds_recovery),
            'client_io_fraction': client_iops_fraction,
            'client_iops_degraded': int(drive_iops * client_iops_fraction),
            'est_recovery_time_hours': recovery_time_hours,
        }

    def _compute_cpu_scaling(self, cpu_us_per_io: float,
                             drive_iops: int,
                             overhead: float) -> Dict[str, Any]:
        """Analyze whether more cores or faster cores would help more.

        Determines the dominant CPU cost (serialized single-threaded work vs
        parallelizable work) and computes how many cores are needed at
        various target configurations.
        """
        drives = self.config.total_osd_count
        costs = dict(self._cpu_costs)

        # Cores needed to support all configured OSDs
        cpu_per_osd_per_sec = cpu_us_per_io * drive_iops
        if cpu_per_osd_per_sec > 0:
            cores_needed_all_drives = (
                drives * cpu_per_osd_per_sec * overhead / 1_000_000)
        else:
            cores_needed_all_drives = 0.0

        # Cores needed with 20% headroom (recommended minimum)
        cores_with_headroom = cores_needed_all_drives * 1.20

        # Cores needed during recovery (if recovery data available)
        cores_recovery = 0.0
        recovery_data = None
        for r in self.results:
            if r.operation.startswith('recovery_'):
                recovery_data = r
                break
        if recovery_data and self.config.recovery_osds > 0:
            recovery_ops_per_osd = {
                'hdd': 50, 'ssd': 200, 'nvme': 500,
            }.get(self.config.drive_type, 50)
            surviving = max(drives - self.config.recovery_osds, 1)
            recovery_participation = self.config.recovery_osds / surviving
            recovery_cpu_per_osd = (recovery_data.cpu_time_per_op_us *
                                    recovery_ops_per_osd *
                                    recovery_participation)
            total_per_osd_recovery = cpu_per_osd_per_sec + recovery_cpu_per_osd
            cores_recovery = (
                surviving * total_per_osd_recovery * overhead / 1_000_000)

        # Identify the dominant cost component
        weighted_costs = {
            'checksumming': costs.get('_crc32c_weighted', 0),
            'compression': costs.get('_compression_weighted', 0),
            'data_protection': costs.get('_protection_weighted', 0),
            'metadata': costs.get('_rocksdb_weighted', 0),
            'crush': costs.get('_crush_weighted', 0),
            'scrub': costs.get('_scrub_weighted', 0),
        }
        total_weighted = sum(weighted_costs.values())
        if total_weighted <= 0:
            total_weighted = 1.0

        dominant_op = max(weighted_costs, key=weighted_costs.get)
        dominant_pct = weighted_costs[dominant_op] / total_weighted * 100

        # Determine if workload benefits more from clock speed or core count
        #
        # Ceph OSD threads are largely independent per-OSD. Within a single
        # OSD, the IO path is pipelined but individual operations (CRC,
        # compress, EC encode) run on a single core. So:
        #
        # - More cores: helps when you have many OSDs per node and each OSD
        #   needs its own CPU budget. The total cluster throughput scales
        #   linearly with core count.
        #
        # - Faster cores: helps when per-IO latency is the bottleneck.
        #   Operations like CRC32C, compression, and EC encode are
        #   single-threaded within an OSD. Faster cores reduce per-IO
        #   latency, which matters for latency-sensitive workloads and
        #   for NVMe drives where the CPU can't keep up with drive speed.
        #
        # Key heuristic: if cpu_us_per_io is high relative to drive latency,
        # the CPU adds significant latency and faster cores help. If the
        # node just needs more total throughput across many OSDs, more
        # cores help.

        drive_latency_us = DRIVE_PROFILES[self.config.drive_type][
            'latency_ms'] * 1000
        cpu_latency_ratio = cpu_us_per_io / drive_latency_us if drive_latency_us > 0 else 0

        # Classify the bottleneck
        # If CPU time per IO exceeds drive latency, CPU adds meaningful
        # latency to every IO -> faster cores help
        # If we just need more total throughput -> more cores help
        if cpu_latency_ratio > 1.0:
            speed_benefit = 'high'
        elif cpu_latency_ratio > 0.3:
            speed_benefit = 'moderate'
        else:
            speed_benefit = 'low'

        current_cores = self.config.cpu_cores_for_ceph
        cores_per_osd = (cpu_per_osd_per_sec * overhead / 1_000_000
                         if cpu_per_osd_per_sec > 0 else 0)

        # Projected core counts at different clock speed improvements
        # (faster clock linearly reduces cpu_us_per_io)
        speed_projections = {}
        for label, factor in [('1.25x faster', 1.25),
                              ('1.5x faster', 1.5),
                              ('2x faster', 2.0)]:
            scaled_cpu = cpu_us_per_io / factor
            scaled_per_osd = scaled_cpu * drive_iops * overhead / 1_000_000
            if scaled_per_osd > 0:
                max_osds_scaled = current_cores / scaled_per_osd
            else:
                max_osds_scaled = float('inf')
            speed_projections[label] = {
                'cores_per_osd': scaled_per_osd,
                'max_osds': math.floor(max_osds_scaled),
                'cores_for_all_drives': drives * scaled_per_osd,
            }

        # Projected capacity at different core counts
        core_projections = {}
        for extra in [4, 8, 16, 32]:
            total = current_cores + extra
            if cores_per_osd > 0:
                max_osds_at_count = total / cores_per_osd
            else:
                max_osds_at_count = float('inf')
            core_projections[f'+{extra} cores ({total:.0f} total)'] = {
                'max_osds': math.floor(max_osds_at_count),
                'supports_all_drives': math.floor(max_osds_at_count) >= drives,
            }

        return {
            'cores_needed_all_drives': cores_needed_all_drives,
            'cores_with_headroom': cores_with_headroom,
            'cores_recovery': cores_recovery,
            'cores_per_osd': cores_per_osd,
            'dominant_operation': dominant_op,
            'dominant_percentage': dominant_pct,
            'cpu_latency_ratio': cpu_latency_ratio,
            'speed_benefit': speed_benefit,
            'speed_projections': speed_projections,
            'core_projections': core_projections,
            'drive_latency_us': drive_latency_us,
        }

    def _compute_per_io_cpu_cost(self):
        obj_size = self.config.get_object_size_bytes()
        size_name = self.config.object_size
        for r in self.results:
            key = r.operation
            if size_name in key or r.object_size == obj_size:
                base = key
                for sn in OBJECT_SIZES:
                    base = base.replace(f'_{sn}', '')
                if base not in self._cpu_costs:
                    self._cpu_costs[base] = r.cpu_time_per_op_us
            elif r.object_size == 0:
                self._cpu_costs[r.operation] = r.cpu_time_per_op_us

    def _get_cost(self, prefix: str) -> float:
        for key, val in self._cpu_costs.items():
            if key.startswith(prefix):
                return val
        return 0.0

    def _total_cpu_cost_per_io(self) -> float:
        crc_cost = self._get_cost('crc32c')

        # Apply CRC32C correction when using slow fallback on SSE4.2 CPU.
        # Real Ceph uses hardware CRC32C (SSE4.2 crc32 instruction) which
        # is ~8-12x faster than zlib.crc32 (software, different polynomial).
        # Since we're modeling what real Ceph would do on this CPU, we
        # correct the CRC32C cost to reflect hardware acceleration.
        # The serialization benchmark also uses CRC32C internally, so
        # it benefits from the same correction.
        if (self.libs and not self.libs.has_hw_crc32c
                and self.libs.cpu_has_sse42 and crc_cost > 0):
            # Hardware CRC32C is typically 8-12x faster than software.
            # Use 10x as a conservative middle estimate.
            self._crc32c_correction = 10.0
            crc_cost /= self._crc32c_correction

        comp_cost = 0.0
        if self.config.compression_enabled:
            comp_prob = {'passive': 0.3, 'aggressive': 0.7, 'force': 1.0}
            p = comp_prob.get(self.config.compression_mode, 0.3)
            rw = self.config.read_write_ratio
            algo = self.config.compression_algorithm
            compress_c = self._get_cost(f'compress_{algo}')
            decompress_c = self._get_cost(f'decompress_{algo}')
            # If the configured algorithm wasn't available and we fell back
            # to a different one, find whatever compression benchmark ran
            if compress_c == 0.0:
                compress_c = self._get_cost('compress_')
            if decompress_c == 0.0:
                decompress_c = self._get_cost('decompress_')
            comp_cost = ((1 - rw) * p * compress_c +
                         rw * p * decompress_c)

        params = self._get_scenario_params()

        if self.config.protection_type == 'erasure':
            rw = self.config.read_write_ratio
            ec_encode_c = self._get_cost('ec_encode')
            ec_decode_c = self._get_cost('ec_decode')
            ec_cost = ((1 - rw) * ec_encode_c +
                       ec_decode_c * params['ec_recovery_fraction'])
        else:
            serial_c = self._get_cost('serialization')
            # Apply CRC32C correction to serialization (uses CRC32C internally)
            if self._crc32c_correction > 1.0:
                serial_c /= self._crc32c_correction
            ec_cost = (self.config.replica_count - 1) * serial_c

        rocksdb_cost = self._get_cost('rocksdb_sim')
        rocksdb_cost *= params['kv_ops_per_io'] / 4.0

        crush_cost = self._get_cost('crush_calculation')

        scrub_overhead = SCRUB_FREQUENCY_OVERHEAD.get(
            self.config.scrub_frequency, 0.01)
        sha256_cost = self._get_cost('sha256') * scrub_overhead

        total = (crc_cost + comp_cost + ec_cost + rocksdb_cost +
                 crush_cost + sha256_cost)

        self._cpu_costs['_total'] = total
        self._cpu_costs['_crc32c_weighted'] = crc_cost
        self._cpu_costs['_compression_weighted'] = comp_cost
        self._cpu_costs['_protection_weighted'] = ec_cost
        self._cpu_costs['_rocksdb_weighted'] = rocksdb_cost
        self._cpu_costs['_crush_weighted'] = crush_cost
        self._cpu_costs['_scrub_weighted'] = sha256_cost

        return total

    def _get_scenario_params(self) -> dict:
        if self.config.scenario == 'best':
            return {
                'kv_ops_per_io': 3,
                'context_switch_overhead': 1.05,
                'safety_margin': 1.00,
                'ec_recovery_fraction': 0.001,
            }
        elif self.config.scenario == 'worst':
            return {
                'kv_ops_per_io': 6,
                'context_switch_overhead': 1.20,
                'safety_margin': 1.30,
                'ec_recovery_fraction': 0.05,
            }
        else:
            return {
                'kv_ops_per_io': 4,
                'context_switch_overhead': 1.10,
                'safety_margin': 1.15,
                'ec_recovery_fraction': 0.01,
            }

    def _compute_overhead_multiplier(self) -> float:
        params = self._get_scenario_params()
        overhead = 1.0

        if self.config.wal_db_separate:
            overhead *= BLUESTORE_OVERHEAD['wal_db_separate']
        else:
            overhead *= BLUESTORE_OVERHEAD['wal_db_same_device']

        overhead *= BLUESTORE_OVERHEAD['rocksdb_compaction']
        overhead *= params['context_switch_overhead']
        overhead *= params['safety_margin']
        return overhead

    def _compute_headroom(self, max_osds: float) -> float:
        if max_osds <= 0:
            return 0.0
        used_fraction = self.config.total_osd_count / max_osds
        return max(0.0, (1.0 - used_fraction) * 100)


# ---------------------------------------------------------------------------
# Scale-Out Projection
# ---------------------------------------------------------------------------

class ScaleOutProjection:
    """Projects performance across multiple nodes."""

    NODE_COUNTS = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, config: ClusterConfig, capacity: Dict[str, Any]):
        self.config = config
        self.capacity = capacity

    def project(self) -> List[Dict[str, Any]]:
        rows = []
        for n in self.NODE_COUNTS:
            rows.append(self._project_for_nodes(n))
        return rows

    def _project_for_nodes(self, node_count: int) -> Dict[str, Any]:
        # Mixed-media: aggregate across device classes
        per_class_data = self.capacity.get('per_device_class')
        if per_class_data:
            total_osds = 0
            total_iops = 0
            any_limited = False
            for entry in per_class_data:
                osds = entry['max_osds_shared']
                total_osds += osds * node_count
                total_iops += osds * node_count * entry['drive_iops']
                if entry['cpu_limited']:
                    any_limited = True
            osds_per_node = sum(e['max_osds_shared'] for e in per_class_data)
        else:
            osds_per_node = min(self.config.total_osd_count,
                                max(self.capacity['max_osds_adjusted'], 0))
            total_osds = osds_per_node * node_count
            drive_iops = self.config.get_drive_iops()
            total_iops = total_osds * drive_iops
            any_limited = (self.config.total_osd_count >
                           self.capacity['max_osds_adjusted'])

        if self.config.protection_type == 'replicated':
            write_amplification = self.config.replica_count
        else:
            write_amplification = ((self.config.ec_k + self.config.ec_m) /
                                   max(self.config.ec_k, 1))

        rw = self.config.read_write_ratio
        denominator = rw * 1.0 + (1 - rw) * write_amplification
        if denominator <= 0:
            denominator = 1.0
        effective_iops = total_iops / denominator

        network_efficiency = 1.0 - (0.02 * math.log2(max(node_count, 1)))
        network_efficiency = max(network_efficiency, 0.80)
        if node_count <= 1:
            network_efficiency = 1.0

        effective_iops *= network_efficiency

        obj_size = self.config.get_object_size_bytes()
        throughput_mbps = effective_iops * obj_size / (1024 * 1024)

        return {
            'nodes': node_count,
            'osds_per_node': osds_per_node,
            'total_osds': total_osds,
            'raw_iops': total_iops,
            'effective_iops': int(effective_iops),
            'throughput_mbps': throughput_mbps,
            'network_efficiency': network_efficiency,
            'cpu_limited': any_limited,
        }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def bytes_to_human_readable(bytes_value: float) -> str:
    """Convert bytes/sec to human readable format."""
    units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
    index = 0
    while bytes_value >= 1024 and index < len(units) - 1:
        bytes_value /= 1024
        index += 1
    return f"{bytes_value:.2f} {units[index]}"


class ReportGenerator:
    """Formats and outputs results in text and CSV."""

    def __init__(self, config: ClusterConfig, libs: LibraryManager,
                 bench_results: List[BenchmarkResult],
                 capacity: Dict[str, Any],
                 scale_out: List[Dict[str, Any]]):
        self.config = config
        self.libs = libs
        self.bench_results = bench_results
        self.capacity = capacity
        self.scale_out = scale_out

    def print_report(self):
        self._print_header()
        self._print_system_info()
        print()
        print(self.libs.summary())
        # Print library warnings (missing hardware CRC32C, etc.)
        warnings = self.libs.get_warnings()
        if warnings:
            print()
            for w in warnings:
                for line in w.split('\n'):
                    print(f"!! WARNING: {line}" if not line.startswith(' ')
                          else f"  {line}")
        print()
        self._print_config_summary()
        self._print_benchmark_results()
        self._print_cpu_cost_breakdown()
        self._print_capacity_estimate()
        self._print_scale_out_table()
        if 'recovery' in self.capacity:
            self._print_recovery_analysis()
        self._print_cpu_scaling_advice()
        self._print_recommendations()

    def _print_header(self):
        print()
        print("=" * 60)
        print("  Ceph CPU IO Simulator Report")
        print("=" * 60)
        print(f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Version: {VERSION}")

    def _print_system_info(self):
        print()
        print("=== System Information ===")
        print(f"CPU Model:    {self.config.cpu_model or 'Unknown'}")
        print(f"CPU Cores:    {self.config.cpu_cores} "
              f"({self.config.cpu_cores_for_ceph:.0f} allocated to Ceph)")
        print(f"Architecture: {platform.machine()}")
        print(f"Python:       {platform.python_version()}")
        print(f"OS:           {platform.system()} {platform.release()}")

    def _print_config_summary(self):
        print("=== Cluster Configuration ===")
        if self.config.is_mixed_media:
            print("Drive Config: Mixed media")
            for dc in self.config.device_classes:
                iops = dc.get_iops(self.config.scenario)
                osd_info = (f" ({dc.osds_per_drive} OSDs/drive)"
                            if dc.osds_per_drive > 1 else "")
                print(f"  {dc.count}x {dc.drive_type.upper():>4} @ "
                      f"{iops:>10,} IOPS/OSD ({self.config.scenario})"
                      f"{osd_info}")
            print(f"Total Drives: {self.config.total_drive_count} per node")
            if self.config.total_osd_count != self.config.total_drive_count:
                print(f"Total OSDs:   {self.config.total_osd_count} per node")
        else:
            print(f"Drive Type:   {self.config.drive_type.upper()}")
            print(f"Drive Count:  {self.config.drive_count} per node")
            if self.config.osds_per_drive > 1:
                print(f"OSDs/Drive:   {self.config.osds_per_drive}")
                print(f"Total OSDs:   {self.config.total_osd_count} per node")
            print(f"Drive IOPS:   {self.config.get_drive_iops():,} "
                  f"({self.config.scenario} profile)")

        if self.config.protection_type == 'replicated':
            print(f"Protection:   Replicated x{self.config.replica_count}")
        else:
            print(f"Protection:   EC {self.config.ec_k}+{self.config.ec_m}")

        if self.config.compression_enabled:
            print(f"Compression:  {self.config.compression_algorithm} "
                  f"({self.config.compression_mode}, "
                  f"est. ratio {self.config.compression_ratio:.2f})")
        else:
            print("Compression:  Disabled")

        print(f"WAL/DB:       {'Separate device' if self.config.wal_db_separate else 'Same device'}")
        print(f"Object Size:  {self.config.object_size}")
        rw = self.config.read_write_ratio
        print(f"Workload:     {self.config.workload_pattern.capitalize()} "
              f"({rw*100:.0f}% read / {(1-rw)*100:.0f}% write)")
        print(f"Scrub:        {self.config.scrub_frequency.capitalize()}")
        print(f"Scenario:     {self.config.scenario.capitalize()}")
        if self.config.recovery_osds > 0:
            print(f"Recovery:     Simulating {self.config.recovery_osds} "
                  f"OSD failure(s)")

    def _print_benchmark_results(self):
        print()
        # Detect parallel mode from result notes
        parallel_n = 0
        for r in self.bench_results:
            if 'workers' in r.notes:
                import re
                m = re.search(r'(\d+) workers', r.notes)
                if m:
                    parallel_n = int(m.group(1))
                    break

        if parallel_n > 0:
            print(f"=== Benchmark Results ({parallel_n} parallel workers) ===")
        else:
            print("=== Benchmark Results ===")
        header = (f"{'Operation':<35} {'Size':>6} {'Ops/sec':>12} "
                  f"{'CPU us/op':>12} {'Throughput':>14} {'Library':<18}")
        print(header)
        print("-" * len(header))

        for r in self.bench_results:
            size_str = _format_size(r.object_size) if r.object_size > 0 else '--'
            tp_str = (f"{r.throughput_mbps:,.1f} MB/s"
                      if r.throughput_mbps > 0 else 'N/A')
            print(f"{r.operation:<35} {size_str:>6} {r.ops_per_sec:>12,.0f} "
                  f"{r.cpu_time_per_op_us:>12,.2f} {tp_str:>14} "
                  f"{r.library_used:<18}")

    def _print_cpu_cost_breakdown(self):
        print()
        print(f"=== CPU Cost Per IO "
              f"(at {self.config.object_size} object size, "
              f"{self.config.scenario} scenario) ===")
        costs = self.capacity.get('per_operation_costs', {})

        rows = [
            ('CRC32C', costs.get('_crc32c_weighted', 0)),
            ('Compression', costs.get('_compression_weighted', 0)),
            ('Data Protection', costs.get('_protection_weighted', 0)),
            ('RocksDB metadata', costs.get('_rocksdb_weighted', 0)),
            ('CRUSH lookup', costs.get('_crush_weighted', 0)),
            ('Scrub (amortized)', costs.get('_scrub_weighted', 0)),
        ]

        header = f"{'Component':<25} {'Weighted CPU us/IO':>20}"
        print(header)
        print("-" * len(header))
        total = 0.0
        for name, val in rows:
            print(f"{name:<25} {val:>20,.2f}")
            total += val
        print("-" * len(header))
        print(f"{'TOTAL':<25} {total:>20,.2f}")

        # Show correction notice if applied
        correction = self.capacity.get('crc32c_correction', 1.0)
        if correction > 1.0:
            print()
            print(f"Note: CRC32C and serialization costs above reflect "
                  f"estimated {correction:.0f}x")
            print(f"hardware acceleration correction (SSE4.2 CRC32C "
                  f"detected but not benchmarked).")
            print(f"Install 'pip install crc32c' for accurate measurement "
                  f"instead of estimation.")

    def _print_capacity_estimate(self):
        cap = self.capacity
        print()
        print("=== OSD Capacity Estimate ===")
        print(f"Available CPU:         "
              f"{cap['available_cpu_us']:,.0f} us/sec "
              f"({self.config.cpu_cores_for_ceph:.0f} cores)")

        if self.config.is_mixed_media:
            self._print_mixed_media_capacity()
        else:
            print(f"Drive IOPS:            "
                  f"{cap['drive_iops']:,} "
                  f"({self.config.drive_type.upper()} "
                  f"{self.config.scenario})")
            print(f"CPU per IO:            "
                  f"{cap['cpu_us_per_io']:,.2f} us")
            print(f"CPU per OSD per sec:   "
                  f"{cap['cpu_us_per_osd_per_sec']:,.0f} us")
            print(f"Overhead multiplier:   "
                  f"{cap['overhead_multiplier']:.2f}x")
            print(f"Max OSDs (raw):        "
                  f"{cap['max_osds_raw']:.2f}")
            print(f"Max OSDs (adjusted):   "
                  f"{cap['max_osds_adjusted']}")
            print(f"Drives configured:     "
                  f"{self.config.drive_count}")
            if self.config.osds_per_drive > 1:
                print(f"OSDs per drive:        "
                      f"{self.config.osds_per_drive}")
                print(f"Total OSDs:            "
                      f"{self.config.total_osd_count}")

            headroom = cap['headroom_percentage']
            if headroom > 0:
                print(f"CPU headroom:          {headroom:.1f}%")
            else:
                over = ((self.config.total_osd_count /
                         max(cap['max_osds_adjusted'], 0.01) - 1) * 100)
                print(f"CPU headroom:          NEGATIVE "
                      f"({over:.0f}% overprovisioned)")

            if cap['max_osds_adjusted'] < self.config.total_osd_count:
                print()
                print("!! WARNING: CPU cannot sustain all configured "
                      "drives at expected IOPS.")
                print("   Drives will be throttled by CPU capacity.")

    def _print_mixed_media_capacity(self):
        cap = self.capacity
        per_class = cap.get('per_device_class', [])
        overhead = cap['overhead_multiplier']

        print(f"CPU per IO:            "
              f"{cap['cpu_us_per_io']:,.2f} us "
              f"(same for all device classes)")
        print(f"Overhead multiplier:   {overhead:.2f}x")
        print()

        # Check if any class has multi-OSD-per-drive
        any_multi_osd = any(e.get('osds_per_drive', 1) > 1 for e in per_class)

        if any_multi_osd:
            header = (f"{'Type':>6} {'Drives':>6} {'OSD/drv':>7} "
                      f"{'OSDs':>5} {'IOPS/OSD':>10} "
                      f"{'CPU/OSD/s':>12} {'Alone':>6} {'Shared':>7} "
                      f"{'CPU Ltd':>8}")
        else:
            header = (f"{'Type':>6} {'Count':>6} {'IOPS/OSD':>10} "
                      f"{'CPU/OSD/s':>12} {'Alone':>6} {'Shared':>7} "
                      f"{'CPU Ltd':>8}")
        print(header)
        print("-" * len(header))
        for entry in per_class:
            dtype = entry['drive_type'].upper()
            count = entry['drive_count']
            opd = entry.get('osds_per_drive', 1)
            total = entry.get('total_osds', count)
            iops = entry['drive_iops']
            cpu_sec = entry['cpu_us_per_osd_per_sec']
            alone = entry['max_osds_standalone']
            shared = entry['max_osds_shared']
            limited = 'YES' if entry['cpu_limited'] else ''
            if any_multi_osd:
                print(f"{dtype:>6} {count:>6} {opd:>7} "
                      f"{total:>5} {iops:>10,} "
                      f"{cpu_sec:>12,.0f} {alone:>6} {shared:>7} "
                      f"{limited:>8}")
            else:
                print(f"{dtype:>6} {count:>6} {iops:>10,} "
                      f"{cpu_sec:>12,.0f} {alone:>6} {shared:>7} "
                      f"{limited:>8}")

        total_drives = cap.get('total_drives', 0)
        total_osds_wanted = cap.get('total_osds_wanted', total_drives)
        total_shared = cap.get('total_shared_osds', 0)
        print("-" * len(header))
        if any_multi_osd:
            print(f"{'TOTAL':>6} {total_drives:>6} {'':>7} "
                  f"{total_osds_wanted:>5} {'':>10} {'':>12} "
                  f"{'':>6} {total_shared:>7}")
        else:
            print(f"{'TOTAL':>6} {total_drives:>6} {'':>10} {'':>12} "
                  f"{'':>6} {total_shared:>7}")

        headroom = cap['headroom_percentage']
        print()
        if headroom > 0:
            print(f"CPU headroom:          {headroom:.1f}%")
        else:
            print(f"CPU headroom:          NEGATIVE "
                  f"({-headroom:.0f}% overprovisioned)")

        # Identify which class is the biggest CPU consumer
        if per_class:
            heaviest = max(per_class, key=lambda e: e['cpu_total_us_sec'])
            total_cpu = sum(e['cpu_total_us_sec'] for e in per_class)
            if total_cpu > 0:
                pct = heaviest['cpu_total_us_sec'] / total_cpu * 100
                print(f"\nDominant consumer: "
                      f"{heaviest['drive_count']}x "
                      f"{heaviest['drive_type'].upper()} "
                      f"({pct:.0f}% of total CPU demand)")

        # Warning if any class is throttled
        any_limited = any(e['cpu_limited'] for e in per_class)
        if any_limited:
            print()
            print("!! WARNING: CPU cannot sustain all drives at "
                  "expected IOPS.")
            for entry in per_class:
                if entry['cpu_limited']:
                    print(f"   {entry['drive_type'].upper()}: "
                          f"{entry['max_osds_shared']} of "
                          f"{entry['drive_count']} drives active")

    def _print_scale_out_table(self):
        print()
        print("=== Scale-Out Projection ===")
        header = (f"{'Nodes':>5} {'OSDs/Node':>10} {'Total OSDs':>11} "
                  f"{'Raw IOPS':>14} {'Eff. IOPS':>14} "
                  f"{'Throughput':>14} {'CPU Ltd':>8}")
        print(header)
        print("-" * len(header))

        for row in self.scale_out:
            tp = _format_throughput(row['throughput_mbps'])
            cpu_ltd = 'YES' if row['cpu_limited'] else 'no'
            print(f"{row['nodes']:>5} {row['osds_per_node']:>10} "
                  f"{row['total_osds']:>11} {row['raw_iops']:>14,} "
                  f"{row['effective_iops']:>14,} {tp:>14} {cpu_ltd:>8}")

    def _print_recovery_analysis(self):
        rec = self.capacity.get('recovery', {})
        if not rec:
            return

        print()
        print("=" * 60)
        print("  Recovery Impact Analysis")
        print("=" * 60)
        print(f"Failed OSDs:           {rec['failed_osds']}")
        print(f"Surviving OSDs:        {rec['surviving_osds']}")
        print()

        print("--- Per-Object Recovery Cost ---")
        print(f"Recovery CPU/object:   {rec['recovery_cpu_per_obj_us']:,.2f} us")
        normal_io_cost = self.capacity['cpu_us_per_io']
        multiplier = (rec['recovery_cpu_per_obj_us'] / normal_io_cost
                      if normal_io_cost > 0 else 0)
        print(f"Normal IO CPU/op:      {normal_io_cost:,.2f} us")
        print(f"Recovery cost:         {multiplier:.1f}x normal IO")
        print()

        print("--- Per-OSD CPU Budget During Recovery ---")
        print(f"Normal client IO:      "
              f"{rec['normal_cpu_per_osd_us']:,.0f} us/sec")
        print(f"Recovery overhead:     "
              f"{rec['recovery_cpu_per_osd_us']:,.0f} us/sec "
              f"({rec['recovery_ops_per_osd_sec']:.0f} recovery ops/sec)")
        print(f"Combined load:         "
              f"{rec['total_cpu_per_osd_us']:,.0f} us/sec")
        print()

        print("--- Cluster Impact ---")
        max_normal = self.capacity['max_osds_adjusted']
        max_recovery = rec['max_osds_during_recovery']
        print(f"Max OSDs (normal):     {max_normal}")
        print(f"Max OSDs (recovery):   {max_recovery}")
        if max_normal > 0:
            reduction = ((max_normal - max_recovery) / max_normal * 100)
            print(f"Capacity reduction:    {reduction:.0f}%")

        client_pct = rec['client_io_fraction'] * 100
        print(f"Client IO capacity:    {client_pct:.0f}% of normal")
        print(f"Client IOPS/OSD:       {rec['client_iops_degraded']:,} "
              f"(was {self.capacity['drive_iops']:,})")
        hours = rec['est_recovery_time_hours']
        if hours < 1.0:
            print(f"Est. recovery time:    {hours * 60:.0f} minutes")
        else:
            print(f"Est. recovery time:    {hours:.1f} hours")

        if max_recovery < max_normal:
            print()
            if max_recovery < self.config.total_osd_count - rec['failed_osds']:
                print("!! CRITICAL: Recovery overhead pushes CPU beyond "
                      "capacity.")
                print("   CPU cannot sustain the surviving OSDs during "
                      "recovery.")
                print("   This can cause cascading slowdowns and "
                      "client timeouts.")
            else:
                print("!! WARNING: Recovery reduces max supportable OSDs "
                      f"from {max_normal} to {max_recovery}.")

        if rec['client_io_fraction'] < 0.5:
            print()
            print("!! WARNING: Client IO drops below 50% during recovery.")
            print("   Applications will experience significant latency "
                  "increases.")
        elif rec['client_io_fraction'] < 0.8:
            print()
            print("!! CAUTION: Client IO reduced to "
                  f"{client_pct:.0f}% during recovery.")

    def _print_cpu_scaling_advice(self):
        scaling = self.capacity.get('cpu_scaling')
        if not scaling:
            return

        cap = self.capacity
        drives = self.config.total_osd_count
        current_cores = self.config.cpu_cores_for_ceph
        max_osds = cap['max_osds_adjusted']

        print()
        print("=" * 60)
        print("  CPU Scaling Analysis")
        print("=" * 60)

        # Current state
        print(f"Current CPU:           "
              f"{current_cores:.0f} cores for Ceph")
        print(f"CPU per OSD:           "
              f"{scaling['cores_per_osd']:.2f} cores/OSD")
        print(f"Cores for {drives} OSDs:    "
              f"{scaling['cores_needed_all_drives']:.1f} cores "
              f"(+20% headroom: {scaling['cores_with_headroom']:.1f})")
        if scaling['cores_recovery'] > 0:
            print(f"Cores during recovery: "
                  f"{scaling['cores_recovery']:.1f} cores")

        # Dominant operation analysis
        print()
        print("--- Cost Breakdown ---")
        dom_op = scaling['dominant_operation']
        dom_pct = scaling['dominant_percentage']
        dom_labels = {
            'checksumming': 'Checksumming (CRC32C)',
            'compression': 'Compression/decompression',
            'data_protection': 'Data protection (replication/EC)',
            'metadata': 'Metadata (RocksDB)',
            'crush': 'CRUSH placement',
            'scrub': 'Scrub',
        }
        print(f"Dominant CPU cost:     {dom_labels.get(dom_op, dom_op)} "
              f"({dom_pct:.0f}% of per-IO cost)")

        # More cores vs faster cores
        print()
        print("--- More Cores vs Faster Cores ---")
        ratio = scaling['cpu_latency_ratio']
        drive_lat = scaling['drive_latency_us']
        cpu_per_io = cap['cpu_us_per_io']
        benefit = scaling['speed_benefit']

        print(f"CPU time per IO:       {cpu_per_io:,.1f} us")
        print(f"Drive latency:         {drive_lat:,.1f} us "
              f"({self.config.drive_type.upper()})")
        print(f"CPU/drive ratio:       {ratio:.2f}x")

        if benefit == 'high':
            print()
            print(">> FASTER CORES recommended.")
            print("   CPU time per IO exceeds drive latency -- the CPU adds")
            print("   significant latency to every operation. Higher clock")
            print("   speed directly reduces per-IO latency.")
            if dom_op == 'data_protection':
                print("   Data protection (replication/EC serialization) is the")
                print("   dominant cost. This scales linearly with clock speed.")
            elif dom_op == 'checksumming':
                print("   CRC32C checksumming dominates. CPUs with dedicated")
                print("   CRC32C instructions (SSE4.2) give ~10x improvement.")
            elif dom_op == 'compression':
                print("   Compression dominates. Faster cores and/or switching")
                print("   to a lighter algorithm (lz4 vs zstd) would help.")
        elif benefit == 'moderate':
            print()
            print(">> BOTH faster cores and more cores would help.")
            print("   CPU time is a meaningful fraction of drive latency.")
            print("   More cores lets you run more OSDs; faster cores")
            print("   reduce per-IO latency for each OSD.")
        else:
            print()
            print(">> MORE CORES recommended over faster cores.")
            print("   CPU time per IO is small relative to drive latency.")
            print("   The drive is the latency bottleneck, not CPU.")
            print("   Adding cores lets you run more OSDs per node.")

        # Core count projections table
        print()
        print("--- Adding More Cores ---")
        header = f"{'Configuration':<30} {'Max OSDs':>10} {'Supports All':>13}"
        print(header)
        print("-" * len(header))
        print(f"{'Current (' + f'{current_cores:.0f} cores)':<30} "
              f"{max_osds:>10} "
              f"{'YES' if max_osds >= drives else 'no':>13}")
        for label, proj in scaling['core_projections'].items():
            support = 'YES' if proj['supports_all_drives'] else 'no'
            print(f"{label:<30} {proj['max_osds']:>10} {support:>13}")

        # Clock speed projections table
        print()
        print("--- Faster Clock Speed ---")
        header = (f"{'Clock Speed':<20} {'Cores/OSD':>10} "
                  f"{'Max OSDs':>10} {'For All OSDs':>15}")
        print(header)
        print("-" * len(header))
        print(f"{'Current':<20} {scaling['cores_per_osd']:>10.2f} "
              f"{max_osds:>10} "
              f"{scaling['cores_needed_all_drives']:>14.1f}")
        for label, proj in scaling['speed_projections'].items():
            print(f"{label:<20} {proj['cores_per_osd']:>10.2f} "
                  f"{proj['max_osds']:>10} "
                  f"{proj['cores_for_all_drives']:>14.1f}")

    def _print_recommendations(self):
        cap = self.capacity
        print()
        print("=== Recommendations ===")

        max_osds = cap['max_osds_adjusted']
        total_osds = self.config.total_osd_count

        if max_osds >= total_osds * 2:
            print("CPU has ample headroom for this configuration.")
            print("Consider adding more drives or enabling compression "
                  "for better utilization.")
        elif max_osds >= total_osds:
            headroom = cap['headroom_percentage']
            print(f"CPU can support all {total_osds} OSDs with "
                  f"{headroom:.0f}% headroom.")
            if headroom < 20:
                print("Headroom is limited. Monitor CPU usage under load.")
        elif max_osds > 0:
            print(f"CPU can only support {max_osds} of {total_osds} OSDs.")
            print("Consider:")
            if self.config.compression_enabled:
                print("  - Disabling or reducing compression")
            if self.config.drive_type == 'nvme':
                print("  - Using fewer NVMe drives per node")
                print("  - Adding more CPU cores")
            if not self.config.wal_db_separate:
                print("  - Moving WAL/DB to a separate fast device")
            print("  - Using larger RADOS object sizes to reduce per-IO "
                  "overhead")
        else:
            print(f"CPU BOTTLENECK: Cannot sustain even 1 OSD at "
                  f"{cap['drive_iops']:,} IOPS.")
            print("This typically occurs with NVMe drives at full speed.")
            print("Real-world Ceph deployments throttle NVMe to match "
                  "CPU capacity.")
            print("Consider:")
            print("  - Reducing drive IOPS expectation "
                  "(use --drive-iops to set realistic target)")
            print("  - Adding significantly more CPU cores")
            print("  - Using larger RADOS object sizes")

    def export_csv(self, filepath: str):
        base, ext = os.path.splitext(filepath)
        if not ext:
            ext = '.csv'
        bench_path = f"{base}_benchmarks{ext}"
        cap_path = f"{base}_capacity{ext}"

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(bench_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'operation', 'object_size', 'ops_per_sec',
                'cpu_time_per_op_us', 'throughput_mbps', 'cpu_utilization',
                'library_used', 'scenario', 'notes'])
            for r in self.bench_results:
                writer.writerow([
                    ts, r.operation, r.object_size, f'{r.ops_per_sec:.2f}',
                    f'{r.cpu_time_per_op_us:.2f}', f'{r.throughput_mbps:.2f}',
                    f'{r.cpu_utilization:.4f}', r.library_used,
                    self.config.scenario, r.notes])
        print(f"\nBenchmark results saved to: {bench_path}")

        with open(cap_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'scenario', 'drive_type', 'drive_count',
                'osds_per_drive', 'total_osds',
                'drive_iops', 'protection', 'compression', 'object_size',
                'cpu_us_per_io', 'max_osds_raw', 'max_osds_adjusted',
                'overhead_multiplier', 'cpu_headroom_pct'])
            prot = (f'replicated:{self.config.replica_count}'
                    if self.config.protection_type == 'replicated'
                    else f'ec:{self.config.ec_k}+{self.config.ec_m}')
            comp = (self.config.compression_algorithm
                    if self.config.compression_enabled else 'none')
            cap = self.capacity
            writer.writerow([
                ts, self.config.scenario, self.config.drive_type,
                self.config.drive_count, self.config.osds_per_drive,
                self.config.total_osd_count,
                cap['drive_iops'], prot, comp,
                self.config.object_size, f"{cap['cpu_us_per_io']:.2f}",
                f"{cap['max_osds_raw']:.2f}", cap['max_osds_adjusted'],
                f"{cap['overhead_multiplier']:.2f}",
                f"{cap['headroom_percentage']:.1f}"])
        print(f"Capacity estimates saved to: {cap_path}")

    @staticmethod
    def _json_safe(obj):
        """Replace inf/nan with JSON-safe values."""
        if isinstance(obj, float):
            if math.isinf(obj):
                return None
            if math.isnan(obj):
                return None
        return obj

    def to_json(self) -> str:
        data = {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_model': self.config.cpu_model,
                'cpu_cores': self.config.cpu_cores,
                'cpu_cores_for_ceph': self.config.cpu_cores_for_ceph,
                'architecture': platform.machine(),
                'python_version': platform.python_version(),
                'os': f'{platform.system()} {platform.release()}',
            },
            'config': {
                'drive_type': self.config.drive_type,
                'drive_count': self.config.drive_count,
                'osds_per_drive': self.config.osds_per_drive,
                'total_osds': self.config.total_osd_count,
                'drive_iops': self.config.get_drive_iops(),
                'protection_type': self.config.protection_type,
                'replica_count': self.config.replica_count,
                'ec_k': self.config.ec_k,
                'ec_m': self.config.ec_m,
                'compression_enabled': self.config.compression_enabled,
                'compression_algorithm': self.config.compression_algorithm,
                'compression_mode': self.config.compression_mode,
                'compression_ratio': self.config.compression_ratio,
                'object_size': self.config.object_size,
                'scenario': self.config.scenario,
                'workload_pattern': self.config.workload_pattern,
                'read_write_ratio': self.config.read_write_ratio,
                'scrub_frequency': self.config.scrub_frequency,
                'wal_db_separate': self.config.wal_db_separate,
            },
            'libraries': {
                'crc32c': self.libs.available.get('crc32c', 'N/A'),
                'lz4': self.libs.available.get('lz4', 'N/A'),
                'zstd': self.libs.available.get('zstd', 'N/A'),
                'snappy': self.libs.available.get('snappy', 'N/A'),
                'erasure_coding': self.libs.available.get(
                    'erasure_coding', 'N/A'),
                'rocksdb': self.libs.available.get('rocksdb', 'N/A'),
            },
            'benchmarks': [
                {
                    'operation': r.operation,
                    'object_size': r.object_size,
                    'ops_per_sec': round(r.ops_per_sec, 2),
                    'cpu_time_per_op_us': round(r.cpu_time_per_op_us, 2),
                    'throughput_mbps': round(r.throughput_mbps, 2),
                    'cpu_utilization': round(r.cpu_utilization, 4),
                    'iterations': r.iterations,
                    'elapsed_sec': round(r.elapsed_sec, 3),
                    'library_used': r.library_used,
                    'notes': r.notes,
                }
                for r in self.bench_results
            ],
            'capacity': {
                'max_osds_raw': round(self.capacity['max_osds_raw'], 2),
                'max_osds_adjusted': self.capacity['max_osds_adjusted'],
                'cpu_us_per_io': round(self.capacity['cpu_us_per_io'], 2),
                'cpu_us_per_osd_per_sec': round(
                    self.capacity['cpu_us_per_osd_per_sec'], 2),
                'available_cpu_us': self.capacity['available_cpu_us'],
                'drive_iops': self.capacity['drive_iops'],
                'overhead_multiplier': round(
                    self.capacity['overhead_multiplier'], 2),
                'headroom_percentage': round(
                    self.capacity['headroom_percentage'], 1),
                'per_operation_costs': {
                    k: round(v, 2)
                    for k, v in self.capacity.get(
                        'per_operation_costs', {}).items()
                },
            },
            'scale_out': self.scale_out,
        }
        if self.config.is_mixed_media:
            data['config']['device_classes'] = [
                {
                    'drive_type': dc.drive_type,
                    'count': dc.count,
                    'osds_per_drive': dc.osds_per_drive,
                    'total_osds': dc.total_osds,
                    'iops_per_osd': dc.get_iops(self.config.scenario),
                    'iops_per_drive': dc.get_drive_iops(self.config.scenario),
                }
                for dc in self.config.device_classes
            ]
        if 'per_device_class' in self.capacity:
            data['capacity']['per_device_class'] = (
                self.capacity['per_device_class'])
            data['capacity']['total_shared_osds'] = (
                self.capacity.get('total_shared_osds', 0))
        if 'recovery' in self.capacity:
            data['recovery'] = self.capacity['recovery']
        if 'cpu_scaling' in self.capacity:
            data['cpu_scaling'] = self.capacity['cpu_scaling']

        # Replace inf/nan with null for valid JSON
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            if isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return None
            return obj

        return json.dumps(sanitize(data), indent=2)


# ---------------------------------------------------------------------------
# Comparison with Real Benchmarks
# ---------------------------------------------------------------------------

def compare_with_real(csv_path: str, config: ClusterConfig,
                      capacity: Dict[str, Any]):
    """Compare simulation with real ceph-bench.sh results."""
    print()
    print("=" * 60)
    print("  Comparison with Real Benchmark Results")
    print("=" * 60)
    print(f"Source: {csv_path}")

    try:
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            print("No data found in CSV file.")
            return

        device_classes = {}
        for row in rows:
            dc = row.get('device_class', 'unknown')
            if dc not in device_classes:
                device_classes[dc] = {'iops': [], 'throughput': [],
                                      'osds': set()}
            try:
                device_classes[dc]['iops'].append(float(row.get('iops', 0)))
                device_classes[dc]['throughput'].append(
                    float(row.get('bytes_per_sec', 0)))
                device_classes[dc]['osds'].add(row.get('osd_id', ''))
            except (ValueError, TypeError):
                continue

        print()
        print(f"{'Device Class':<15} {'Avg IOPS':>12} "
              f"{'Avg Throughput':>16} {'OSD Count':>10}")
        print("-" * 55)

        for dc, data in sorted(device_classes.items()):
            if not data['iops']:
                continue
            avg_iops = sum(data['iops']) / len(data['iops'])
            avg_tp = sum(data['throughput']) / len(data['throughput'])
            osd_count = len(data['osds'])
            print(f"{dc:<15} {avg_iops:>12,.0f} "
                  f"{bytes_to_human_readable(avg_tp):>16} {osd_count:>10}")

        print()
        print("Simulated CPU capacity at real IOPS levels:")
        print(f"{'Device Class':<15} {'Real IOPS/OSD':>14} "
              f"{'CPU us/IO (sim)':>16} {'Max OSDs (sim)':>15}")
        print("-" * 62)

        cpu_us_per_io = capacity['cpu_us_per_io']
        available_cpu_us = capacity['available_cpu_us']
        overhead = capacity.get('overhead_multiplier', 1.0)

        for dc, data in sorted(device_classes.items()):
            if not data['iops']:
                continue
            avg_iops = sum(data['iops']) / len(data['iops'])
            osd_count = len(data['osds'])
            real_cpu_per_osd = cpu_us_per_io * avg_iops * overhead
            if real_cpu_per_osd > 0:
                real_max_osds = available_cpu_us / real_cpu_per_osd
            else:
                real_max_osds = float('inf')

            print(f"{dc:<15} {avg_iops:>14,.0f} "
                  f"{cpu_us_per_io:>16,.2f} {real_max_osds:>15,.1f}")

            utilization = (osd_count / real_max_osds * 100
                           if real_max_osds > 0 else float('inf'))
            if utilization > 100:
                print(f"  >> CPU BOTTLENECK: ~{utilization:.0f}% utilization "
                      "suggests throttling")
            elif utilization > 80:
                print(f"  >> WARNING: CPU near saturation at "
                      f"~{utilization:.0f}%")
            else:
                print(f"  >> OK: CPU has ~{100-utilization:.0f}% headroom")

    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.0f}M"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}K"
    return f"{size_bytes}B"


def _format_throughput(mbps: float) -> str:
    if mbps >= 1024:
        return f"{mbps / 1024:,.1f} GB/s"
    return f"{mbps:,.1f} MB/s"


def _detect_cpu_model() -> str:
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (OSError, IOError):
        pass
    try:
        return platform.processor() or 'Unknown'
    except Exception:
        return 'Unknown'


def _prompt_input(prompt: str, default: str = '') -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _prompt_int(prompt: str, default: int) -> int:
    val = _prompt_input(prompt, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def _prompt_float(prompt: str, default: float) -> float:
    val = _prompt_input(prompt, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def _prompt_choice(prompt: str, choices: List[str], default: str) -> str:
    choices_str = '/'.join(choices)
    val = _prompt_input(f"{prompt} [{choices_str}] ({default}): ", default)
    if val in choices:
        return val
    return default


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------

def interactive_config() -> ClusterConfig:
    """Guide user through cluster configuration interactively."""
    config = ClusterConfig()

    print()
    print("=== Ceph CPU IO Simulator - Interactive Configuration ===")
    print()

    # CPU
    cores = os.cpu_count() or 4
    print(f"Detected {cores} CPU cores.")
    config.cpu_cores = _prompt_int(f"CPU cores available [{cores}]: ", cores)
    default_ceph = max(1, config.cpu_cores - 2)
    config.cpu_cores_for_ceph = _prompt_float(
        f"CPU cores for Ceph [{default_ceph}]: ", float(default_ceph))

    # Drives
    print()
    config.drive_type = _prompt_choice("Drive type", ['hdd', 'ssd', 'nvme'],
                                       'hdd')
    config.drive_count = _prompt_int("Drives per node [12]: ", 12)

    config.osds_per_drive = _prompt_int(
        "OSD daemons per drive [1]: ", 1)

    custom_iops = _prompt_input(
        "Custom drive IOPS (0=use profile default) [0]: ", "0")
    try:
        config.drive_iops = int(custom_iops)
    except ValueError:
        config.drive_iops = 0

    # Data protection
    print()
    prot = _prompt_choice("Data protection", ['replicated', 'erasure'],
                          'replicated')
    config.protection_type = prot
    if prot == 'replicated':
        config.replica_count = _prompt_int("Replica count [3]: ", 3)
    else:
        config.ec_k = _prompt_int("EC data chunks (k) [4]: ", 4)
        config.ec_m = _prompt_int("EC parity chunks (m) [2]: ", 2)

    # Compression
    print()
    comp = _prompt_choice("Enable compression?", ['yes', 'no'], 'no')
    config.compression_enabled = (comp == 'yes')
    if config.compression_enabled:
        config.compression_algorithm = _prompt_choice(
            "Compression algorithm", ['snappy', 'zstd', 'lz4', 'zlib'],
            'zstd')
        config.compression_mode = _prompt_choice(
            "Compression mode", ['passive', 'aggressive', 'force'], 'passive')
        config.compression_ratio = _prompt_float(
            "Expected compression ratio (0.0-1.0) [0.5]: ", 0.5)

    # WAL/DB
    print()
    config.wal_db_separate = (_prompt_choice(
        "WAL/DB on separate device?", ['yes', 'no'], 'no') == 'yes')

    # Object size
    config.object_size = _prompt_choice(
        "RADOS object size", list(OBJECT_SIZES.keys()), '4m')

    # Workload
    print()
    config.workload_pattern = _prompt_choice(
        "Workload pattern", ['sequential', 'random', 'mixed'], 'mixed')
    config.read_write_ratio = _prompt_float(
        "Read/write ratio (0.0=all writes, 1.0=all reads) [0.7]: ", 0.7)

    # Scrub
    config.scrub_frequency = _prompt_choice(
        "Scrub frequency", ['daily', 'weekly', 'disabled'], 'daily')

    # Scenario
    print()
    config.scenario = _prompt_choice(
        "Scenario", ['best', 'worst', 'typical', 'all'], 'typical')

    # Duration
    config.benchmark_duration = _prompt_float(
        "Benchmark duration per test (seconds) [5.0]: ", 5.0)

    return config


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Ceph CPU IO Simulator - Benchmark CPU capacity '
                    'for OSD workloads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --drive-type hdd --drive-count 12 --protection replicated:3
  %(prog)s --drive-type nvme --drive-count 4 --protection ec:4+2 --compress zstd
  %(prog)s --drives 36xhdd 4xnvme --wal-db-separate --recovery-osds 1
  %(prog)s --drives 36xhdd:150 4xnvme:100000     # with IOPS overrides
  %(prog)s --interactive
  %(prog)s --quick
  %(prog)s --compare real_bench_results.csv
        """)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--interactive', '-i', action='store_true',
                      help='Interactive guided configuration')
    mode.add_argument('--quick', action='store_true',
                      help='Quick benchmark (2s duration, fewer sizes). '
                           'Other CLI flags are still honored')

    cpu = parser.add_argument_group('CPU Configuration')
    cpu.add_argument('--cpu-cores', type=int, default=0,
                     help='Total CPU cores (0=auto-detect)')
    cpu.add_argument('--cpu-cores-ceph', type=float, default=0,
                     help='CPU cores reserved for Ceph '
                          '(0=auto: total minus 2)')

    drive = parser.add_argument_group('Drive Configuration')
    drive.add_argument('--drive-type', '-t',
                       choices=['hdd', 'ssd', 'nvme'], default='hdd',
                       help='Drive type for single-class config '
                            '(default: hdd). Use --drives for mixed media')
    drive.add_argument('--drive-count', '-d', type=int, default=12,
                       help='Drives per node for single-class config '
                            '(default: 12). Use --drives for mixed media')
    drive.add_argument('--drive-iops', type=int, default=0,
                       help='Override drive IOPS (0=use profile: '
                            'HDD=150, SSD=50K, NVMe=500K typical)')
    drive.add_argument('--osds-per-drive', type=int, default=1,
                       help='OSD daemons per drive for single-class config '
                            '(default: 1). Use --drives NxTYPE:IOPS:OSDS '
                            'for per-class control in mixed media')
    drive.add_argument('--drives', nargs='+', default=None,
                       metavar='NxTYPE',
                       help='Mixed media: specify multiple device classes. '
                            'Format: COUNTxTYPE[:IOPS[:OSDS_PER_DRIVE]]. '
                            'Example: --drives 24xhdd 4xnvme:0:2 '
                            'or --drives 36xhdd:150 4xnvme:100000:2. '
                            'Use IOPS=0 for profile defaults. '
                            'Overrides --drive-type/--drive-count')

    prot = parser.add_argument_group('Data Protection')
    prot.add_argument('--protection', '-p', default='replicated:3',
                      help='Protection: replicated:N (N=replica count) or '
                           'ec:K+M (K=data, M=parity chunks) '
                           '(default: replicated:3)')

    comp = parser.add_argument_group('Compression')
    comp.add_argument('--compress', '-c', default=None,
                      choices=['snappy', 'zstd', 'lz4', 'zlib'],
                      help='Enable compression with specified algorithm')
    comp.add_argument('--compress-ratio', type=float, default=0.5,
                      help='Expected compression ratio: 0.5 means data '
                           'compresses to 50%% of original (default: 0.5)')
    comp.add_argument('--compress-mode', default='passive',
                      choices=['passive', 'aggressive', 'force'],
                      help='Compression mode: passive=only if beneficial, '
                           'aggressive=try more objects, force=always '
                           '(default: passive)')

    bs = parser.add_argument_group('BlueStore')
    bs.add_argument('--wal-db-separate', action='store_true',
                    help='WAL/DB on separate fast device')
    bs.add_argument('--object-size', default='4m',
                    choices=list(OBJECT_SIZES.keys()),
                    help='RADOS object size (default: 4m)')

    wl = parser.add_argument_group('Workload')
    wl.add_argument('--workload', default='mixed',
                    choices=['sequential', 'random', 'mixed'],
                    help='Workload pattern (default: mixed). Note: CPU cost '
                         'model uses random IOPS; sequential workloads have '
                         'lower CPU cost per byte in practice')
    wl.add_argument('--rw-ratio', type=float, default=0.7,
                    help='Read/write ratio: 0.0=all writes, 1.0=all reads '
                         '(default: 0.7)')
    wl.add_argument('--scrub', default='daily',
                    choices=['daily', 'weekly', 'disabled'],
                    help='Scrub frequency (default: daily)')

    parser.add_argument('--scenario', '-s', default='typical',
                        choices=['best', 'worst', 'typical', 'all'],
                        help='Scenario: best=lowest IOPS/overhead, '
                             'worst=highest IOPS/overhead, '
                             'all=run all three (default: typical)')

    rec = parser.add_argument_group('Recovery Simulation')
    rec.add_argument('--recovery-osds', type=int, default=0,
                     help='Simulate N OSD failures to model recovery '
                          'CPU impact (default: 0 = no recovery sim)')

    out = parser.add_argument_group('Output')
    out.add_argument('--output', '-o', default=None,
                     help='CSV output base file (creates '
                          '*_benchmarks.csv and *_capacity.csv)')
    out.add_argument('--compare', default=None,
                     help='Compare with ceph-bench.sh CSV results')
    out.add_argument('--json', action='store_true',
                     help='Output results as JSON')

    bench = parser.add_argument_group('Benchmark Control')
    bench.add_argument('--duration', type=float, default=5.0,
                       help='Seconds per benchmark (default: 5.0)')
    bench.add_argument('--sizes', nargs='+',
                       default=['4k', '64k', '128k', '4m'],
                       choices=list(OBJECT_SIZES.keys()),
                       help='Object sizes for micro-benchmarks. '
                            'The --object-size value is auto-added '
                            '(default: 4k 64k 128k 4m)')
    bench.add_argument('--parallel', type=int, default=0,
                       metavar='N',
                       help='Run benchmarks across N parallel workers to '
                            'measure CPU contention effects (default: 0 = '
                            'single-threaded). Simulates N OSD daemons '
                            'competing for CPU. Recommended: set to your '
                            'total OSD count')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {VERSION}')

    return parser.parse_args()


def parse_drives(specs: List[str]) -> List[DeviceClass]:
    """Parse mixed-media drive specs.

    Format: COUNTxTYPE[:IOPS[:OSDS_PER_DRIVE]]
    Examples:
      36xhdd              - 36 HDDs, 1 OSD each
      4xnvme:100000       - 4 NVMe, custom IOPS, 1 OSD each
      4xnvme:0:2          - 4 NVMe, default IOPS, 2 OSDs each
      4xnvme:100000:2     - 4 NVMe, custom IOPS, 2 OSDs each
    """
    classes = []
    valid_types = set(DRIVE_PROFILES.keys())
    for spec in specs:
        spec = spec.strip().lower()
        # Format: COUNTxTYPE[:IOPS[:OSDS_PER_DRIVE]]
        if 'x' not in spec:
            print(f"Error: Invalid drive spec '{spec}'. "
                  f"Use format COUNTxTYPE[:IOPS[:OSDS]] (e.g., 36xhdd, "
                  f"4xnvme:0:2)")
            sys.exit(1)
        count_str, rest = spec.split('x', 1)
        try:
            count = int(count_str)
        except ValueError:
            print(f"Error: Invalid drive count in '{spec}'")
            sys.exit(1)
        if count < 1:
            print(f"Error: Drive count must be >= 1 in '{spec}'")
            sys.exit(1)

        iops_override = 0
        osds_per_drive = 1
        if ':' in rest:
            parts = rest.split(':')
            dtype = parts[0]
            if len(parts) >= 2 and parts[1]:
                try:
                    iops_override = int(parts[1])
                except ValueError:
                    print(f"Error: Invalid IOPS in '{spec}'")
                    sys.exit(1)
            if len(parts) >= 3 and parts[2]:
                try:
                    osds_per_drive = int(parts[2])
                except ValueError:
                    print(f"Error: Invalid OSDs-per-drive in '{spec}'")
                    sys.exit(1)
                if osds_per_drive < 1:
                    print(f"Error: OSDs-per-drive must be >= 1 in '{spec}'")
                    sys.exit(1)
        else:
            dtype = rest

        if dtype not in valid_types:
            print(f"Error: Unknown drive type '{dtype}' in '{spec}'. "
                  f"Valid types: {', '.join(sorted(valid_types))}")
            sys.exit(1)

        classes.append(DeviceClass(
            drive_type=dtype, count=count, iops_override=iops_override,
            osds_per_drive=osds_per_drive))
    return classes


def parse_protection(spec: str) -> Tuple[str, int, int, int]:
    """Parse protection spec like 'replicated:3' or 'ec:4+2'."""
    spec = spec.strip().lower()
    try:
        if spec.startswith('replicated'):
            parts = spec.split(':')
            count = int(parts[1]) if len(parts) > 1 else 3
            if count < 1:
                print(f"Warning: replica count must be >= 1, using 3")
                count = 3
            return 'replicated', count, 0, 0
        elif spec.startswith('ec'):
            parts = spec.split(':')
            if len(parts) > 1:
                km = parts[1].split('+')
                k = int(km[0])
                m = int(km[1]) if len(km) > 1 else 2
                if k < 1:
                    print(f"Error: EC k (data chunks) must be >= 1, got {k}")
                    sys.exit(1)
                if m < 1:
                    print(f"Error: EC m (parity chunks) must be >= 1, got {m}")
                    sys.exit(1)
                return 'erasure', 0, k, m
    except (ValueError, IndexError) as e:
        print(f"Error: Invalid protection spec '{spec}': {e}")
        sys.exit(1)
    return 'replicated', 3, 0, 0


def _validate_config(config: ClusterConfig):
    """Validate configuration values and exit with helpful errors."""
    errors = []

    if config.drive_count < 1:
        errors.append("--drive-count must be >= 1")

    if config.osds_per_drive < 1:
        errors.append("--osds-per-drive must be >= 1")

    if config.drive_iops < 0:
        errors.append("--drive-iops must be >= 0")

    if not 0.0 <= config.read_write_ratio <= 1.0:
        errors.append(
            f"--rw-ratio must be between 0.0 and 1.0, got "
            f"{config.read_write_ratio}")

    if config.compression_enabled:
        if not 0.0 <= config.compression_ratio <= 1.0:
            errors.append(
                f"--compress-ratio must be between 0.0 and 1.0, got "
                f"{config.compression_ratio}")

    if config.benchmark_duration <= 0:
        errors.append("--duration must be > 0")

    if config.recovery_osds < 0:
        errors.append("--recovery-osds must be >= 0")

    total_osds = config.total_osd_count
    if config.recovery_osds > 0 and config.recovery_osds >= total_osds:
        errors.append(
            f"--recovery-osds ({config.recovery_osds}) must be less than "
            f"total OSD count ({total_osds}); "
            f"losing all OSDs means the cluster is down")

    # Ensure --object-size is included in --sizes for accurate capacity modeling
    if config.object_size not in config.object_sizes_to_test:
        config.object_sizes_to_test.append(config.object_size)
        print(f"Note: Added {config.object_size} to benchmark sizes "
              f"(required for capacity modeling)")

    if errors:
        for e in errors:
            print(f"Error: {e}")
        sys.exit(1)

    # Non-fatal warnings (only after passing validation)
    if config.recovery_osds > 0:
        if (config.protection_type == 'replicated' and
                config.recovery_osds >= config.replica_count):
            print(f"Warning: losing {config.recovery_osds} OSDs with "
                  f"replicated:{config.replica_count} means some data is "
                  f"unrecoverable. Modeling recovery of salvageable PGs.")
        elif (config.protection_type == 'erasure' and
              config.recovery_osds > config.ec_m):
            print(f"Warning: losing {config.recovery_osds} OSDs with "
                  f"EC {config.ec_k}+{config.ec_m} exceeds parity tolerance. "
                  f"Some data is unrecoverable. Modeling recovery of "
                  f"salvageable PGs.")


def build_config_from_args(args) -> ClusterConfig:
    config = ClusterConfig()
    config.cpu_cores = args.cpu_cores
    config.cpu_cores_for_ceph = args.cpu_cores_ceph

    # Mixed-media: --drives overrides --drive-type/--drive-count
    if args.drives:
        config.device_classes = parse_drives(args.drives)
        # Set primary type to the first class for backward compat
        config.drive_type = config.device_classes[0].drive_type
        config.drive_count = config.total_drive_count
        config.osds_per_drive = 1  # per-class control via --drives
    else:
        config.drive_type = args.drive_type
        config.drive_count = args.drive_count
        config.drive_iops = args.drive_iops
        config.osds_per_drive = args.osds_per_drive

    ptype, rep_count, ec_k, ec_m = parse_protection(args.protection)
    config.protection_type = ptype
    config.replica_count = rep_count
    config.ec_k = ec_k
    config.ec_m = ec_m

    if args.compress:
        config.compression_enabled = True
        config.compression_algorithm = args.compress
        config.compression_ratio = args.compress_ratio
        config.compression_mode = args.compress_mode

    config.wal_db_separate = args.wal_db_separate
    config.recovery_osds = args.recovery_osds
    config.object_size = args.object_size
    config.workload_pattern = args.workload
    config.read_write_ratio = args.rw_ratio
    config.scrub_frequency = args.scrub
    config.scenario = args.scenario
    config.benchmark_duration = args.duration
    config.object_sizes_to_test = args.sizes

    _validate_config(config)

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    libs = LibraryManager()

    if args.interactive:
        config = interactive_config()
    elif args.quick:
        config = build_config_from_args(args)
        config.benchmark_duration = 2.0
        config.object_sizes_to_test = ['4k', '4m']
        # Ensure capacity modeling size is included
        if config.object_size not in config.object_sizes_to_test:
            config.object_sizes_to_test.append(config.object_size)
    else:
        config = build_config_from_args(args)

    if config.cpu_cores == 0:
        config.cpu_cores = os.cpu_count() or 4
    if config.cpu_cores_for_ceph <= 0:
        config.cpu_cores_for_ceph = float(max(1, config.cpu_cores - 2))

    config.cpu_model = _detect_cpu_model()

    scenarios = (['best', 'worst', 'typical'] if config.scenario == 'all'
                 else [config.scenario])

    # Run benchmarks once -- scenario only affects the capacity model, not
    # the raw CPU micro-benchmarks.  Reuse results across scenarios.
    if not args.json:
        print(f"\n{'=' * 60}")
        print(f"  Running benchmarks...")
        print(f"{'=' * 60}")
    else:
        print("Running benchmarks...", file=sys.stderr)

    parallel = getattr(args, 'parallel', 0)
    benchmarks = CephBenchmarks(libs, config, verbose=args.verbose,
                                parallel_workers=parallel)
    if parallel >= 2 and not args.json:
        print(f"  Parallel mode: {parallel} workers")
    results = benchmarks.run_all()

    all_capacities = []
    json_documents = []

    for scenario in scenarios:
        config.scenario = scenario

        model = OSDCapacityModel(config, results, libs=libs)
        capacity = model.calculate()

        projection = ScaleOutProjection(config, capacity)
        scale_out = projection.project()

        report = ReportGenerator(config, libs, results, capacity, scale_out)

        if args.json:
            json_documents.append(json.loads(report.to_json()))
        else:
            report.print_report()

        if args.output:
            if len(scenarios) > 1:
                base, ext = os.path.splitext(args.output)
                output_file = f"{base}_{scenario}{ext}"
            else:
                output_file = args.output
            report.export_csv(output_file)

        all_capacities.append(capacity)

    # Emit valid JSON: single object or array depending on scenario count
    if args.json:
        if len(json_documents) == 1:
            print(json.dumps(json_documents[0], indent=2))
        else:
            print(json.dumps(json_documents, indent=2))

    if args.compare and all_capacities:
        compare_with_real(args.compare, config, all_capacities[-1])


if __name__ == '__main__':
    main()
