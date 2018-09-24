from setuptools import setup


LONG_DESCRIPTION = '''
The MNIST database is available at http://yann.lecun.com/exdb/mnist/

The MNIST database is a dataset of handwritten digits. It has 60,000
training samples, and 10,000 test samples. Each image is represented
by 28x28 pixels, each containing a value 0 - 255 with its grayscale value.

It is a subset of a larger set available from NIST. The digits have been
size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and
pattern recognition methods on real-world data while spending minimal
efforts on preprocessing and formatting.

There are four files available, which contain separately train and test,
and images and labels.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges.

mnist makes it easier to download and parse MNIST files.
'''

setup(name='mnist',
      description='Python utilities to download and parse the MNIST dataset',
      long_description=LONG_DESCRIPTION,
      url='https://github.com/datapythonista/mnist',
      version='0.2.2',
      author='Marc Garcia',
      author_email='garcia.marc@gmail.com',
      license='BSD',
      packages=['mnist'],
      install_requires=['numpy'],
      extras_require={':python_version=="2.7"': ['mock']},
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Operating System :: OS Independent',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering'])
