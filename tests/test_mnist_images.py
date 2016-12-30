import sys
import os
import struct
import tempfile
import gzip
import array
import unittest
from unittest import mock

sys.path.append('..')
import pymnist


SAMPLE_MAGIC_NUMBER = (0x08 << 8) + 0x01
SAMPLE_NUM_IMAGES = 3
SAMPLE_IMAGE_ROWS = 1
SAMPLE_IMAGE_COLS = 2
SAMPLE_NUM_PIXELS = SAMPLE_NUM_IMAGES * SAMPLE_IMAGE_ROWS * SAMPLE_IMAGE_COLS
SAMPLE_PIXELS = (0, 255) + (255, 0) + (128, 128)
SAMPLE_DATA = struct.pack('>' + 'I' * 4 + 'B' * SAMPLE_NUM_PIXELS,
                          SAMPLE_MAGIC_NUMBER,
                          SAMPLE_NUM_IMAGES,
                          SAMPLE_IMAGE_ROWS,
                          SAMPLE_IMAGE_COLS,
                          *SAMPLE_PIXELS)


class TestMnistImages(unittest.TestCase):
    """Test that mnist_images function returns the right data and
    types, and make the right calls."""

    def setUp(self):
        self.sample_fname = tempfile.mktemp(prefix='test_pymnist_')
        with gzip.open(self.sample_fname, 'wb') as f:
            f.write(SAMPLE_DATA)

    def tearDown(self):
        try:
            os.remove(self.sample_fname)
        except FileNotFoundError:  # not saved when download_mnist_file is mocked
            pass

    def test_passing_file_returns_correct_pixels(self):
        actual_labels = tuple(pymnist.mnist_images(self.sample_fname)[0].flatten())
        self.assertEqual(actual_labels, SAMPLE_PIXELS)

    def test_returns_numpy_array_by_default(self):
        import numpy as np
        actual_pixels, rows, cols = pymnist.mnist_images(self.sample_fname)
        self.assertIsInstance(actual_pixels, np.ndarray)

    def test_numpy_array_has_correct_rows(self):
        import numpy as np
        actual_num_rows = pymnist.mnist_images(self.sample_fname)[0].shape[0]
        self.assertEqual(actual_num_rows, SAMPLE_NUM_IMAGES)

    def test_numpy_array_has_correct_cols(self):
        import numpy as np
        actual_num_cols = pymnist.mnist_images(self.sample_fname)[0].shape[1]
        self.assertEqual(actual_num_cols, SAMPLE_IMAGE_ROWS * SAMPLE_IMAGE_COLS)

    def test_returns_python_array(self):
        actual_pixels, rows, cols = pymnist.mnist_images(self.sample_fname,
                                                         use_numpy=False)
        self.assertIsInstance(actual_pixels, array.array)

    def test_returned_python_array_has_correct_pixels(self):
        actual_pixels, rows, cols = pymnist.mnist_images(self.sample_fname,
                                                         use_numpy=False)
        self.assertEqual(len(actual_pixels), SAMPLE_NUM_PIXELS)

    def test_returns_correct_number_of_rows(self):
        actual_pixels, rows, cols = pymnist.mnist_images(self.sample_fname,
                                                         use_numpy=False)
        self.assertEqual(rows, SAMPLE_IMAGE_ROWS)

    def test_returns_correct_number_of_cols(self):
        actual_pixels, rows, cols = pymnist.mnist_images(self.sample_fname,
                                                         use_numpy=False)
        self.assertEqual(cols, SAMPLE_IMAGE_COLS)

    def test_no_fname_calls_download(self):
        with mock.patch('pymnist.main.download_mnist_file',
                        return_value=self.sample_fname) as download_mnist_file:
            pymnist.main.mnist_images()
            download_mnist_file.assert_called_once_with(test=False)

    def test_no_fname_calls_download_for_test(self):
        with mock.patch('pymnist.main.download_mnist_file', return_value=self.sample_fname) as download_mnist_file:
            pymnist.mnist_images(test=True)
            download_mnist_file.assert_called_once_with(test=True)

    def test_passing_file_does_not_remove_file(self):
        with mock.patch('os.remove') as os_remove:
            pymnist.mnist_images(self.sample_fname)
        self.assertFalse(os_remove.called)

    def test_no_fname_removes_downloaded_file_by_default(self):
        with mock.patch('pymnist.main.download_mnist_file', return_value=self.sample_fname):
            with mock.patch('os.remove') as os_remove:
                pymnist.main.mnist_images()
        os_remove.assert_called_once_with(self.sample_fname)

    def xtest_no_fname_does_not_remove_file(self):
        with mock.patch('pymnist.main.download_mnist_file', self.sample_fname):
            with mock.patch('os.remove') as os_remove:
                pymnist.mnist_images(delete_tempfile=False)
        self.assertFalse(os_remove.called)
