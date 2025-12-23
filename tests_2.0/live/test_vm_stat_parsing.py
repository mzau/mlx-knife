"""Unit tests for vm_stat parsing helpers."""

from .test_utils import parse_vm_stat_page_size


def test_parse_vm_stat_page_size_apple_silicon():
    output = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 12345."
    assert parse_vm_stat_page_size(output) == 16384


def test_parse_vm_stat_page_size_fallback():
    output = "Pages free: 12345."
    assert parse_vm_stat_page_size(output) == 4096
