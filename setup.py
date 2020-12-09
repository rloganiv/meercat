import os
import setuptools
import sys


# Load README to get long description.
with open('README.md') as f:
    _LONG_DESCRIPTION = f.read()


# Don't import library while setting up.
version_path = os.path.join(os.path.dirname(__file__), 'meercat')
sys.path.append(version_path)
from version import __version__


setuptools.setup(
    name='meercat',
    version=__version__,
    description='MEERCAT Model + Experiments',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Robert L. Logan IV (@rloganiv)',
    url='https://github.com/rloganiv/meercat',
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers>=4.0.0',
    ],
    extras_require={
        'test': ['pytest']
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='text nlp machinelearining',
)
