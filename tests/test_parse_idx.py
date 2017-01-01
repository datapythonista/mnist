import io
import unittest
try:
    from unittest import mock
except ImportError:
    import mock  # py2
import numpy as np
import mnist


class TestParseIdx(unittest.TestCase):
    """Test that IDX files are parsed correctly"""

    def test_empty_file_raises_exception(self):
        fd = io.BytesIO(b'')
        with self.assertRaises(mnist.IdxDecodeError):
            mnist.parse_idx(fd)

    def test_missing_header_raises_exception(self):
        fd = io.BytesIO(b'\x00')
        with self.assertRaises(mnist.IdxDecodeError):
            mnist.parse_idx(fd)

    def test_missing_initial_zeros_raises_exception(self):
        fd = io.BytesIO(b'\xff\xff\x08\x00')
        with self.assertRaises(mnist.IdxDecodeError):
            mnist.parse_idx(fd)

    def test_unknown_data_type_raises_error(self):
        fd = io.BytesIO(b'\x00\x00\xff\x00')
        with self.assertRaises(mnist.IdxDecodeError):
            mnist.parse_idx(fd)

    def test_missing_items_raises_error(self):
        fd = io.BytesIO(b'\x00\x00\x08\x01'
                        b'\x00\x00\x00\x02'
                        b'\xff')
        with self.assertRaises(mnist.IdxDecodeError):
            mnist.parse_idx(fd)

    def test_unexpected_items_raises_error(self):
        fd = io.BytesIO(b'\x00\x00\x08\x01'
                        b'\x00\x00\x00\x02'
                        b'\x00\x01\x02')
        with self.assertRaises(mnist.IdxDecodeError):
            mnist.parse_idx(fd)

    def test_file_with_one_dimension_returns_correct_values(self):
        fd = io.BytesIO(b'\x00\x00\x08\x01'
                        b'\x00\x00\x00\x02'
                        b'\xff\x00')
        actual = mnist.parse_idx(fd)
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual([255, 0], actual.tolist())

    def test_file_with_two_dimensions_returns_correct_values(self):
        fd = io.BytesIO(b'\x00\x00\x08\x02'
                        b'\x00\x00\x00\x02'
                        b'\x00\x00\x00\x03'
                        b'\x00\x01\x02\x03\x04\x05')
        actual = mnist.parse_idx(fd)
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual([[0, 1, 2], [3, 4, 5]], actual.tolist())

    def test_file_with_int_type_returns_correct_values(self):
        fd = io.BytesIO(b'\x00\x00\x0c\x01'
                        b'\x00\x00\x00\x01'
                        b'\x00\x00\x00\xff')  # two's complement of 255
        actual = mnist.parse_idx(fd)
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual([255], actual.tolist())

    def test_file_with_negative_int_returns_correct_values(self):
        fd = io.BytesIO(b'\x00\x00\x0c\x01'
                        b'\x00\x00\x00\x01'
                        b'\xff\xff\xff\xff')  # two's complement of -1
        actual = mnist.parse_idx(fd)
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual([-1], actual.tolist())
