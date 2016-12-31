import os
import gzip
import struct
import array
import tempfile
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2


BASE_URL = 'http://yann.lecun.com/exdb/mnist/'


def test_mock():
    raise NameError('not your business')
    pass


def download_mnist_file(labels=False, test=False):
    """Download a MNIST file from Yann LeCun site, and return
    the file name where it has been saved.
    
    Parameters
    ----------
    labels : bool
        Whether to download images (default) or labels file

    test : bool
        Whether to download train (default) or test file

    Examples
    --------
    >>> fname = download_mnist_file()  # download train images
    >>> fname = download_mnist_file(labels=True, test=True)  # download test labels
    """
    if labels:
        fname = 't10k-labels-idx1-ubyte.gz' if test else 'train-labels-idx1-ubyte.gz'
    else:
        fname = 't10k-images-idx3-ubyte.gz' if test else 'train-images-idx3-ubyte.gz'
        
    tmp_fname = tempfile.mktemp(prefix='pymnist_' + fname)
    urlretrieve(BASE_URL + fname, tmp_fname)
    return tmp_fname


def mnist_images(fname=None, test=False, use_numpy=True, delete_tempfile=True):
    """Open a MNIST file from http://yann.lecun.com/exdb/mnist/
    and return it as a numpy array.

    Parameters
    ----------
    fname : str
        Path to the labels file to process (if None, file is downloaded)

    test : bool
        If fname is None, specify whether to download train (default) or test file

    use_numpy : bool
        If True, return the result as a numpy array

    delete_tempfile : bool
        If True, delete the downloaded file after loading it to memory

    Returns
    -------
    labels : array or numpy.ndarray
        Array of int with the labels

    Examples
    --------
    >>> train_x, rows, cols  = mnist_images()
    >>> train_x,  = mnist_images(use_numpy=False)
    >>> train_x,  = mnist_images('train-images-idx3-ubyte.gz')
    >>> test_x,  = mnist_images(train=False)
    """
    if fname is None:
        fname = download_mnist_file(test=test)
    else:
        delete_tempfile = False

    with gzip.open(fname, 'rb') as f:
        (magic_number,
         num_images,
         image_rows,
         image_cols) = struct.unpack('>IIII', f.read(16))
        pixels = array.array('B', f.read())

    if delete_tempfile:
        os.remove(fname)
    
    expected_pixels = num_images * image_rows * image_cols
    if len(pixels) != expected_pixels:
        raise ValueError('Expected %d pixels, found %d' % (expected_pixels,
                                                           len(pixels)))

    if use_numpy:
        import numpy as np
        pixels = np.array(pixels).reshape((num_images,
                                           image_rows * image_cols))

    return pixels, image_rows, image_cols


def mnist_labels(fname=None, test=False, use_numpy=True, delete_tempfile=True):
    """Open a MNIST labels file (or downloads it if fname is None),
    and returns it as an array.

    Parameters
    ----------
    fname : str
        Path to the labels file to process (if None, file is downloaded)

    test : bool
        If fname is None, specify whether to download train (default) or test file

    use_numpy : bool
        If True, return the result as a numpy array

    delete_tempfile : bool
        If True, delete the downloaded file after loading it to memory

    Returns
    -------
    labels : array or numpy.ndarray
        Array of int with the labels

    Examples
    --------
    >>> train_y = mnist_labels()
    >>> train_y = mnist_labels(use_numpy=False)
    >>> train_y = mnist_labels('train-labels-idx1-ubyte.gz')
    >>> test_y = mnist_labels(test=True)

    """
    if fname is None:
        fname = download_mnist_file(labels=True, test=test)
    else:
        delete_tempfile = False

    with gzip.open(fname, 'rb') as f:
        magic_number, num_images = struct.unpack('>II', f.read(8))
        labels = array.array('B', f.read())

    if len(labels) != num_images:
        raise ValueError('Expected %d images, found %d' % (num_images,
                                                           len(labels)))

    if delete_tempfile:
        os.remove(fname)
        
    if use_numpy:
        import numpy as np
        labels = np.array(labels)

    return labels
