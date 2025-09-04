from . import lib_ncc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class NCC(object):
    def __init__(self, infile_prefix: str = None, 
                 center_y1: float = None, center_y2: float = None, half_length_y1: float = None, half_length_y2: float = None,
                 num_pixels_y1: int = None, num_pixels_y2: int = None, over_sample: int = None,
                 write_maps: bool = False, write_histograms: bool = False,
                 outfile_prefix: str = None, verbose: int = 0):
        '''
        :param infile_prefix: prefix to be used when reading in files
        :param center_y1: y1 coordinate of the center of the number of caustic crossings map
        :param center_y2: y2 coordinate of the center of the number of caustic crossings map
        :param half_length_y1: y1 extent of the half-length of the number of caustic crossings map
        :param half_length_y2: y2 extent of the half_length of the number of caustic crossings map
        :param num_pixels_y1: number of pixels for the y1 axis
        :param num_pixels_y2: number of pixels for the y2 axis
        :param over_sample: specify the power of 2 by which to oversample the final pixels. E.g., an input of 4 means the
                            final pixel array will initially be oversampled by a value of 2^4=16 along both axes. This
                            will require 16*16=256 times more memory
        :param write_maps: whether to write number of caustic crossings maps or not
        :param write_histograms: whether to write histograms or not
        :param outfile_prefix: prefix to be used in output file names
        :param verbose: verbosity level of messages. must be 0, 1, 2, or 3
        '''
        self.lib = lib_ncc.lib

        self.obj = self.lib.NCC_init()
        self.verbose = verbose

        self.infile_prefix = infile_prefix
        
        self.center_y1 = center_y1
        self.center_y2 = center_y2
        self.half_length_y1 = half_length_y1
        self.half_length_y2 = half_length_y2
        self.num_pixels_y1 = num_pixels_y1
        self.num_pixels_y2 = num_pixels_y2
        
        self.over_sample = over_sample
        
        self.write_maps = write_maps
        self.write_histograms = write_histograms
        
        self.outfile_prefix = outfile_prefix

    def __del__(self):
        self.lib.NCC_delete(self.obj)

    @property
    def infile_prefix(self):
        return self.lib.get_infile_prefix(self.obj).decode('utf-8')
    
    @infile_prefix.setter
    def infile_prefix(self, value):
        if value is not None:
            self.lib.set_infile_prefix(self.obj, value.encode('utf-8'))

    @property
    def center_y1(self):
        return self.lib.get_center_y1(self.obj)
    
    @center_y1.setter
    def center_y1(self, value):
        if value is not None:
            self.lib.set_center_y1(self.obj, value)

    @property
    def center_y2(self):
        return self.lib.get_center_y2(self.obj)
    
    @center_y2.setter
    def center_y2(self, value):
        if value is not None:
            self.lib.set_center_y2(self.obj, value)

    @property
    def center(self):
        return (self.center_y1, self.center_y2)

    @property
    def half_length_y1(self):
        return self.lib.get_half_length_y1(self.obj)
    
    @half_length_y1.setter
    def half_length_y1(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("half_length_y1 must be > 0")
            self.lib.set_half_length_y1(self.obj, value)

    @property
    def half_length_y2(self):
        return self.lib.get_half_length_y2(self.obj)
    
    @half_length_y2.setter
    def half_length_y2(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("half_length_y2 must be > 0")
            self.lib.set_half_length_y2(self.obj, value)

    @property
    def half_length(self):
        return (self.half_length_y1, self.half_length_y2)

    @property
    def num_pixels_y1(self):
        return self.lib.get_num_pixels_y1(self.obj)
    
    @num_pixels_y1.setter
    def num_pixels_y1(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("num_pixels_y1 must be >= 1")
            self.lib.set_num_pixels_y1(self.obj, value)

    @property
    def num_pixels_y2(self):
        return self.lib.get_num_pixels_y2(self.obj)
    
    @num_pixels_y2.setter
    def num_pixels_y2(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("num_pixels_y2 must be >= 1")
            self.lib.set_num_pixels_y2(self.obj, value)

    @property
    def num_pixels(self):
        return (self.num_pixels_y1, self.num_pixels_y2)

    @property
    def pixel_scales(self):
        return (2 * self.half_length[0] / self.num_pixels[0],
                2 * self.half_length[1] / self.num_pixels[1])

    @property
    def over_sample(self):
        return self.lib.get_over_sample(self.obj)
    
    @over_sample.setter
    def over_sample(self, value):
        if value is not None:
            if value < 0:
                raise ValueError("over_sample must be >= 0")
            self.lib.set_over_sample(self.obj, value)

    @property
    def write_maps(self):
        return bool(self.lib.get_write_maps(self.obj))
    
    @write_maps.setter
    def write_maps(self, value):
        if value is not None:
            self.lib.set_write_maps(self.obj, int(value))

    @property
    def write_histograms(self):
        return bool(self.lib.get_write_histograms(self.obj))
    
    @write_histograms.setter
    def write_histograms(self, value):
        if value is not None:
            self.lib.set_write_histograms(self.obj, int(value))

    @property
    def outfile_prefix(self):
        return self.lib.get_outfile_prefix(self.obj).decode('utf-8')
    
    @outfile_prefix.setter
    def outfile_prefix(self, value):
        if value is not None:
            self.lib.set_outfile_prefix(self.obj, value.encode('utf-8'))

    def run(self):
        if not self.lib.run(self.obj, self.verbose):
            raise Exception("Error running NCC")
        
        self.num_caustic_crossings = np.ctypeslib.as_array(self.lib.get_num_crossings(self.obj), 
                                                           shape=(self.num_pixels_y2,
                                                                  self.num_pixels_y1)).copy()
    
    @property
    def extent(self):
        return np.array([[(self.center[0] - self.half_length[0]),
                          (self.center[0] + self.half_length[0])],
                         [(self.center[1] - self.half_length[1]),
                          (self.center[1] + self.half_length[1])]])

    def save(self):
        if not self.lib.save(self.obj, self.verbose):
            raise Exception("Error saving NCC")

    def plot(self, ax: Axes, cmap='viridis', **kwargs):
        if 'vmin' not in kwargs.keys():
            kwargs['vmin'] = np.min(self.num_caustic_crossings) - 0.5
        if 'vmax' not in kwargs.keys():
            kwargs['vmax'] = np.max(self.num_caustic_crossings) + 0.5

        cmap = plt.get_cmap(cmap, int(kwargs['vmax'] - kwargs['vmin']))

        img = ax.imshow(self.num_caustic_crossings, extent=self.extent.ravel(), cmap=cmap, **kwargs)
        cbar = ax.get_figure().colorbar(img, label='$N_{\\text{caustic crossings}}$')

        # make sure ticks are only at integers
        cbar.set_ticks([what for what in cbar.get_ticks() 
                        if what == int(what) and what > kwargs['vmin'] and what < kwargs['vmax']])

        ax.set_xlabel('$y_1$')
        ax.set_ylabel('$y_2$')

    def plot_hist(self, ax: Axes, bins=None, **kwargs):
        if bins is None:
            vmin, vmax = (np.min(self.num_caustic_crossings) - 0.5, 
                          np.max(self.num_caustic_crossings) + 0.5)
            bins = np.arange(vmin, vmax + 1, 1)

        ax.hist(self.num_caustic_crossings.ravel(), bins=bins, density=True, **kwargs)

        ax.set_xlabel('$N_{\\text{caustic crossings}}$')
        ax.set_ylabel('p($N_{\\text{caustic crossings}}$)')
