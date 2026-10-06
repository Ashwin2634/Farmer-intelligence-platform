"""
Hashing and checksum utilities for dataset integrity and deduplication.
"""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes the SHA-256 hash of a file efficiently using chunking.

    Args:
        file_path (Union[str, Path]): Path to file.
        chunk_size (int): Chunk size in bytes.

    Returns:
        str: Hexadecimal SHA-256 string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_md5(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes MD5 hash of a file.

    Args:
        file_path (Union[str, Path]): Path to file.
        chunk_size (int): Chunk size in bytes.

    Returns:
        str: Hexadecimal MD5 string.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()
