from setuptools import setup, find_namespace_packages

long_description = '''This package provides code for

1. generating microlensing magnification maps,
2. locating microlensing critical curves and caustics,
3. determining the number of microlensing caustic crossings, and
4. locating the positions of microimages,

on GPUs (1-3) and CPUs (4). It relies on python wrappers around C++/CUDA
libraries which have been precompiled and included with this package for
Linux and Windows x86-64 architectures. Compilation for Linux used the
GNU compiler v12.2.0 and the CUDA compiler v12.6.85. Libraries *should* work
for Linux distributions that have GLIBC >= 2.34 and GLIBCXX >= 3.4.29 and CUDA
compute capabilities >=8.0, but no promises. Compilation for Windows used the
MSVC compiler v19.44.35226 and the CUDA compiler v12.5.82. Libraries *should*
work for Windows 11 and CUDA compute capabilities >=7.5, but no promises.
Further details can be found at https://github.com/weisluke/microlensing'''

setup(
    name='microlensing',
    version='0.1.8',
    description='A package for microlensing simulations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luke Weisenbach',
    author_email='weisluke@alum.mit.edu',
    url="https://github.com/weisluke/microlensing/",
    packages=find_namespace_packages(),
    license="GNU AFFERO GENERAL PUBLIC LICENSE",
    platforms=['Linux', 'Windows'],
    package_data={"microlensing.lib": ["*.so", "*.dll"]},
    python_requires='>=3.10',
    install_requires=['numpy', 'scipy', 'astropy', 'matplotlib', 'sncosmo>=2.10', 'shapely']
)
