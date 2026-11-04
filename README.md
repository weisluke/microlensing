# Introduction
This repository contains source code for 

1. generating microlensing magnification maps,
2. locating microlensing critical curves and caustics,
3. determining the number of microlensing caustic crossings, and
4. locating the positions of microimages,

on GPUs (1-3) and CPUs (4).

## Methods
We utilize Inverse Polygon Mapping (IPM, Mediavilla et al. [2006](https://ui.adsabs.harvard.edu/abs/2006ApJ...653..942M/abstract), [2011](https://ui.adsabs.harvard.edu/abs/2011ApJ...741...42M/abstract)), combined with the Fast Multipole Method (FMM, Greengard and Rokhlin [1987](https://ui.adsabs.harvard.edu/abs/1987JCoPh..73..325G/abstract)) as suggested and used by Jiménez-Vicente and Mediavilla ([2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...941...80J/abstract)) for CPUs, in order to generate magnification maps. The area of each IPM cell is allocated among the pixels it maps to with a closed-form per-pixel coverage integral (Green's theorem applied to the triangle edges, evaluated per pixel row); triangles whose clamped bounding box covers more than 64 pixels are instead deposited along their perimeter: for each edge and pixel row, the partial coverage on the edge's span is added to the map and one step value at the span's first column to a map-sized step buffer, and a row-wise prefix sum of that buffer is added to the map after the kernel. The cost of such a triangle then scales with its perimeter rather than its area, and the result is exact up to rounding (parity maps are supported). Earlier versions clipped the triangles with the Sutherland-Hodgman ([1974](https://doi.org/10.1145/360767.360802)) algorithm.

We use Witt's ([1990](https://ui.adsabs.harvard.edu/abs/1990A&A...236..311W)) method to locate the microlensing critical curves and caustics, while taking advantage of the FMM again to decrease computation time for terms involving derivatives of the microlensing potential. In order to locate the roots of the parametric critical curve equation, we employ the [Aberth](https://doi.org/10.2307/2005621)-[Ehrlich](https://doi.org/10.1145/363067.363115) method, a cubically convergent algorithm that allows for simultaneous approximations of the roots of a polynomial. 

We use the fact that the caustics are clockwise (under the chosen critical curve parametrization) oriented closed curves to utilize the winding number in order to calculate a two dimensional map of the the number of caustic crossings (Wambsganss et al. [1992](https://ui.adsabs.harvard.edu/abs/1992A&A...258..591), Granot et al. [2003](https://ui.adsabs.harvard.edu/abs/2003ApJ...583..575G)). We use Sunday's ([2001](https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html)) algorithm to calculate the winding number of the discretized caustic polygons around the center of every pixel, efficiently creating a map that provides the number of caustic crossings.

We locate the images of infinite source lines using the method of Lewis et al. [1993](https://ui.adsabs.harvard.edu/abs/1993MNRAS.261..647L/abstract) (see also Witt [1993](https://ui.adsabs.harvard.edu/abs/1993ApJ...403..530W/abstract)) solely on CPUs, though GPUs are required to calculate the FMM coefficients. We can then also locate the microimage positions for a point source as the intersections of the images of two perpendicular lines. 

## Credits
You are free to use this code under the provided license. If you happen to use it for any work that leads to publications, it would be much appreciated if you cite the appropriate paper for either creating magnification maps ([here](https://ui.adsabs.harvard.edu/abs/2025MNRAS.541..281W/abstract)), or for calculating critical curves, caustics, and the number of caustic crossings ([here](https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.3341W/abstract)). 

I make no promise that the current state of the code will stay the same, or that things will always be backwards compatible. Poor design choices I made in the past may need be changed to make the future better! Suggested improvements and bugfixes are also always appreciated. 

I would love to be considered for involvement in any projects which make use of these tools, as I'd like to think I bring a bit of microlensing knowledge to the table :). I can be reached at [weisluke@alum.mit.edu](mailto:weisluke@alum.mit.edu).


## Dependencies
The code is written to be independent of anything besides the C++ standard libraries and the Thrust libraries that come with CUDA. We implement IPM, the FMM, the pixel coverage integral, the Aberth-Ehrlich method, Sunday's algorithm, and a similar algorithm to that of Lewis et al. [1993](https://ui.adsabs.harvard.edu/abs/1993MNRAS.261..647L/abstract) ourselves.

The bulk of the code is written using NVIDIA's CUDA, and you will need a CUDA capable device to run it (and a CUDA installation to compile it, if necessary. We require a C++20 compliant compiler in that case).

The perimeter deposit needs one extra map-sized buffer in device memory (with parity maps, the otherwise unused total-map buffer holds the steps of one parity, so it is still one extra buffer). If that buffer does not fit in the free device memory, the code falls back to the direct per-pixel deposit, which gives the same map up to rounding, and reports this at verbosity 2.

## Repository layout
The `include` directory contains the bulk of the code, which is written mostly as templated objects and functions. The `src` directory contains files for creating executables and libraries. The `bin` directory holds [compilation](#compiling) output. The `microlensing` directory contains [python](#python) code for ease of use, while the `examples` directory contains example python notebooks outlining usage.

## Compiling
Compilation has been tested with the GNU compiler v12.2.0 and the CUDA compiler v12.6.85 for Linux, and with the MSVC compiler v19.44.35226 and the CUDA compiler v12.5.82 for Windows.

First, clone the repository.
```
git clone https://github.com/weisluke/microlensing.git
```

Then, on Linux simply run 
```
source compile
```
or on Windows 
```
compile.bat
```
to run the compilation file. This should create 5 executables and 5 libraries in the `bin` directory.

The IPM shooting kernel uses cooperative-groups warp partitioning, which needs a device of compute capability 7.0 or newer (the scripts target 8.0 on Linux and 7.5 on Windows), and sizes its launch grid with the CUDA occupancy API at run time. 

I'll note that this repository contains the libraries precompiled and placed in the `./microlensing/lib/` directory already. These libraries may or may not work on your hardware, as they were compiled for a particular cluster (SCIAMA) for Linux and on a home desktop for Windows, but you can try running the example [python](#python) notebooks first to see if they do. They *should* work for Linux distributions that have GLIBC >= 2.34 and GLIBCXX >= 3.4.29 and CUDA compute capabilities >=8.0, and for Windows 11 and CUDA compute capabilities >=7.5, but no promises. If there are errors, you will need to compile everything yourself (though I'm happy to do my best to help with that if need be).

## python

In addition to command line programs, we also provide wrappers for our code that allow them to be used with python once the relevant libraries are [compiled](#compiling). This eases the need to save output from the executables to disk and then read them from disk: for IPM the magnification maps stay in device memory and each one is copied directly into a numpy array once it is created (the IPM executable streams them to files through two small pinned host buffers), while CCF, NCC and MIF, and IPM with the Windows libraries included here, copy from managed memory to a numpy array. 

If you just want the python capabilities (as I suspect is the case for most people currently) and do not wish to use the executables or contribute to the code, then instead of cloning this repository you can `pip install microlensing` (PyPI page  located [here](https://pypi.org/project/microlensing/)). The python package includes precompiled libraries as mentioned [above](#compiling), which may or may not work for you. We hope to expand to cover more architectures in the future. If the included libraries do not work, you will have to clone this repository, compile them yourself, and do a local `pip install` instead.

The `examples` directory contains *very* brief examples of the python packages I've written to create microlensing magnification maps, calculate critical curves and caustics, calculate number of caustic crossings maps, and locate microimages. There are additional examples showing how to generate lightcurves, and a note on units for the magnification maps. The `view_output.ipynb` example shows how to read the output of the IPM executable.


### Batches of IPM maps

`microlensing.IPM.batch.BatchedIPM` generates many IPM maps from one description, on one or more GPUs, and hands them back as they finish. It takes the arguments of `IPM`, plus `num_maps`. Each argument is either a single value shared by every map or a sequence with one value per map; sequences are paired element by element (use `itertools.product` for a grid). The number of maps is `num_maps` if given, otherwise the common length of the sequences (which must all agree, and must match `num_maps`), otherwise one. A string argument (`mass_function`, `starfile`, `outfile_prefix`) counts as a single value. Mismatches raise in the constructor, before any GPU work. `params(i)` returns the resolved inputs of map `i`.

Seeds are fixed before any GPU work and are reported on each result as `random_seed`. A sequence of seeds is used as given. A single seed is a *starting* seed, even for one map: it gives each map a distinct seed in `1 ... 2**31 - 1` through `numpy.random.default_rng(start)`, map `i`'s seed does not depend on `num_maps`, and `random_seed=[s]` reproduces an `IPM` run with seed `s`. With no seed, distinct seeds are drawn. Seed and prefix sequences must not contain `None`, and given seeds must lie in `1 ... 2**31 - 1`. Maps with a `starfile` get seed 0, as with `IPM`. A single `outfile_prefix` (or none, meaning `"./"`) gives map `i` the prefix `prefix + f"{i}_"`; a sequence is used as given, and duplicate prefixes raise `ValueError`. The prefixes name files only when maps are saved (`ipm.save()` on a result, or `run(save=True)` with `write_*` flags set); nothing is written otherwise, and, as with `IPM.save()`, the output directory must already exist.

```python
from microlensing.IPM.batch import BatchedIPM

# N maps, same lens, different seeds
with BatchedIPM(kappa_tot=0.4, shear=0.4, smooth_fraction=0.5,
                random_seed=range(1, 1001),                # or random_seed=42 and num_maps=1000
                half_length_y1=25, half_length_y2=25,
                num_pixels_y1=5000, num_pixels_y2=5000,
                outfile_prefix="run1/") as batch:           # names files only when maps are saved: run1/0_ipm_..., run1/1_ipm_...
    for index, ipm in batch.run():                         # as maps finish, in any order
        if isinstance(ipm, Exception):                     # a BatchMapError: this map failed, the rest go on
            print(index, ipm); continue
        ...                                                # ipm is an ordinary IPM holding its map

# sweep: one value per map; write each map to disk as it finishes and keep nothing in RAM
batch = BatchedIPM(kappa_tot=[0.3, 0.4, 0.5], shear=[0.2, 0.4, 0.4], smooth_fraction=[0.0, 0.5, 0.9],
                   random_seed=[11, 12, 13], write_maps=True, write_stars=True, ...)
records = batch.run(save=True)    # a list of SavedMap(index, outfile_prefix, random_seed, error)
```

`run()` yields `(index, ipm)` in completion order. `ipm` is an ordinary `IPM` in the state `IPM.run()` leaves it (`save()` works on it and writes the same files), with its star arrays sorted by the tree, as with `IPM`. Results hold no GPU memory, so the GPUs' buffers are reused for the next map; `close()`, or leaving the `with` block, frees the workers' GPU memory. The finished maps waiting for the consumer (in save mode, for their files to be written) are capped at `max_result_bytes` (default 8 GiB), with always room for one map: while the cap is reached, workers stop starting maps, and a map that finishes keeps its result in its worker until there is room. Maps in progress do not count against the cap, so large maps still run side by side. Host RAM therefore holds at most `max_result_bytes` of waiting results (or one result, if that is larger), plus the results of the maps in progress (at most `maps_per_gpu`, 8 by default, per GPU), plus the results your loop keeps. `run(save=True)` blocks, writes each map as it finishes with the files selected by `write_stars`, `write_maps`, `write_parities` and `write_histograms` (with none set, only parameter files are written, with a warning), and returns one `SavedMap` per map in batch order; `writers` sets the number of writer threads. By default `devices=None` uses every visible GPU, and `maps_per_gpu=None` runs at most 8 maps per GPU, fewer if memory is short; an explicit `maps_per_gpu` is only an upper limit. `last_run_stats` describes the last run: its `result_budget` is `max_result_bytes`, `peak_result_bytes` the most result bytes that waited at once, and `peak_in_flight_result_bytes` the most that waiting maps and maps in progress held together.

A failing map does not stop the batch: it comes back as `(index, BatchMapError)` (or as the `error` of its `SavedMap`). A map that runs out of device memory is retried with fewer maps on its GPU, and fails only if it cannot run alone. A GPU whose CUDA context breaks leaves the batch with a warning and its maps move to the other GPUs. Breaking out of the loop, an exception or Ctrl-C stops new maps from starting, lets running ones finish and joins the workers. `t_shoot_cells` of a batched map is wall time and can include time shared with other maps on the GPU. With `verbose` above 0, the messages that the C++ code prints for maps running at the same time interleave and do not say which map they belong to; use `verbose=0` for large batches. Maps with a `starfile` read their file twice (once for the memory estimate and once to run) and allocate their star memory anew on every run, which briefly stalls the GPU, so star-file batches overlap less than batches with generated star fields. Batching needs the Linux libraries; with others (for example the Windows ones) the maps run one at a time, with a warning, and the results are plain `IPM` objects that hold their GPU memory until they are deleted.
