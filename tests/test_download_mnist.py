import sys
import os
import glob
import tempfile
import gzip
import unittest
sys.path.append('..')
import pymnist


TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
IMAGE_COLS = 28
IMAGE_ROWS = 28
IMAGE_SIZE = IMAGE_COLS * IMAGE_ROWS
HEADER_IMAGES_SIZE = 4 * 4
HEADER_LABELS_SIZE = 2 * 4
PIXEL_BYTES = 1
LABEL_BYTES = 1


class TestDownloadMNIST(): #unittest.TestCase):
    """Test that files have been successfully downloaded, and that they
    didn't change, by testing they have the expected sizes."""

    def tearDown(self):
        fname_pattern = os.path.join(tempfile.gettempdir(),
                                     'pymnist_*')
        for fname in glob.glob(fname_pattern):
            os.remove(fname)

    @staticmethod
    def _gzip_file_size(fname):
        with gzip.open(fname, 'rb') as f:
            return len(f.read())

    def test_train_images_has_right_size(self):
        fname = pymnist.download_mnist_file()
        expected_size = HEADER_IMAGES_SIZE + TRAIN_SAMPLES * IMAGE_SIZE * PIXEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    def test_test_images_has_right_size(self):
        fname = pymnist.download_mnist_file(test=True)
        expected_size = HEADER_IMAGES_SIZE + TEST_SAMPLES * IMAGE_SIZE * PIXEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    def test_train_labels_has_right_size(self):
        fname = pymnist.download_mnist_file(labels=True)
        expected_size = HEADER_LABELS_SIZE + TRAIN_SAMPLES * LABEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    def test_test_labels_has_right_size(self):
        fname = pymnist.download_mnist_file(labels=True, test=True)
        expected_size = HEADER_LABELS_SIZE + TEST_SAMPLES * LABEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)
