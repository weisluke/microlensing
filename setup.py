from setuptools import setup, find_namespace_packages

long_description = '''This package provides code for

1. generating microlensing magnification maps,
2. locating microlensing critical curves and caustics, and
3. determining the number of microlensing caustic crossings,

all on GPUs. It relies on python wrappers around C++/CUDA libraries which have
been precompiled and included with this package for linux_x86_64 architectures.
Compilation used the GNU compiler v12.2.0 and the CUDA HPC SDK v23.9.
Further details can be found at https://github.com/weisluke/microlensing'''

setup(
    name='microlensing',
    version='0.1.0',
    description='A package for microlensing simulations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luke Weisenbach',
    author_email='weisluke@alum.mit.edu',
    url="https://github.com/weisluke/microlensing/",
    packages=find_namespace_packages(),
    license="GNU AFFERO GENERAL PUBLIC LICENSE",
    platforms=['linux_x86_64'],
    package_data={"microlensing.lib": ["*.so"]},
    install_requires=['numpy', 'scipy', 'astropy', 'matplotlib', 'sncosmo']
)
