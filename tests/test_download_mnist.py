import os
import glob
import tempfile
import gzip
import unittest
try:
    from unittest import mock
except ImportError:
    import mock  # py2
import mnist


TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
IMAGE_COLS = 28
IMAGE_ROWS = 28
IMAGE_SIZE = IMAGE_COLS * IMAGE_ROWS
HEADER_IMAGES_SIZE = 4 * 4
HEADER_LABELS_SIZE = 2 * 4
PIXEL_BYTES = 1
LABEL_BYTES = 1


class TestDownloadMNIST(unittest.TestCase):
    """Test that files have been successfully downloaded, and that they
    didn't change, by testing they have the expected sizes."""

    def setUp(self):
        self.original_mnist_datasets_url = mnist.datasets_url
        self.downloaded_fname = os.path.join(tempfile.gettempdir(),
                                             'test_mnist_downloaded')
        with open(self.downloaded_fname, 'wb'):
            pass

    def tearDown(self):
        mnist.datasets_url = self.original_mnist_datasets_url
        os.remove(self.downloaded_fname)
        fname_pattern = os.path.join(tempfile.gettempdir(),
                                     '*-*-idx*-ubyte.gz')
        for fname in glob.glob(fname_pattern):
            os.remove(fname)

    @staticmethod
    def _gzip_file_size(fname):
        with gzip.open(fname, 'rb') as f:
            return len(f.read())

    def test_train_images_has_right_size(self):
        fname = 'train-images-idx3-ubyte.gz'
        fname = mnist.download_file(fname, force=True)
        expected_size = HEADER_IMAGES_SIZE + TRAIN_SAMPLES * IMAGE_SIZE * PIXEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    def test_test_images_has_right_size(self):
        fname = 't10k-images-idx3-ubyte.gz'
        fname = mnist.download_file(fname, force=True)
        expected_size = HEADER_IMAGES_SIZE + TEST_SAMPLES * IMAGE_SIZE * PIXEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    def test_train_labels_has_right_size(self):
        fname = 'train-labels-idx1-ubyte.gz'
        fname = mnist.download_file(fname, force=True)
        expected_size = HEADER_LABELS_SIZE + TRAIN_SAMPLES * LABEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    def test_test_labels_has_right_size(self):
        fname = 't10k-labels-idx1-ubyte.gz'
        fname = mnist.download_file(fname, force=True)
        expected_size = HEADER_LABELS_SIZE + TEST_SAMPLES * LABEL_BYTES
        actual_size = self._gzip_file_size(fname)
        self.assertEqual(expected_size, actual_size)

    @mock.patch('mnist.urlretrieve')
    def test_file_is_downloaded_to_target_dir(self, urlretrieve):
        fname = mnist.download_file('test', target_dir='/tmp/mnist_test/')
        urlretrieve.assert_called_once_with(mnist.datasets_url + 'test',
                                            '/tmp/mnist_test/test')
        self.assertEqual(fname, '/tmp/mnist_test/test')

    @mock.patch('mnist.urlretrieve')
    def test_file_is_not_downloaded_when_force_is_false(self, urlretrieve):
        mnist.download_file(self.downloaded_fname, force=False)
        self.assertFalse(urlretrieve.called)

    @mock.patch('mnist.urlretrieve')
    def test_file_is_downloaded_when_exists_and_force_is_true(self, urlretrieve):
        mnist.download_file('test', force=True)
        urlretrieve.assert_called_once_with(mnist.datasets_url + 'test',
                                            os.path.join(tempfile.gettempdir(), 'test'))

    @mock.patch('mnist.urlretrieve')
    def test_datasets_url_is_used(self, urlretrieve):
        mnist.datasets_url = 'http://aaa.com/'
        mnist.download_file('mnist_datasets_url.gz')
        fname = os.path.join(tempfile.gettempdir(), 'mnist_datasets_url.gz')
        urlretrieve.assert_called_once_with('http://aaa.com/mnist_datasets_url.gz',
                                            fname)
