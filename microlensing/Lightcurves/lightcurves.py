try:
    import cupy as np
    from cupyx.scipy.signal import correlate
except ImportError:
    import numpy as np
    from scipy.signal import correlate

from microlensing.IPM.ipm import IPM
from . import util


def constant_source(ipm: IPM, source, positions = 1, return_pos: bool = False):
    '''
    Return the magnification(s) for the provided magnification map,
    constant source profile, and position(s)

    :param ipm: formally, an IPM instance that has ran and has a magnification
                map. in general, any object that has attributes magnifications,
                center, half_length, num_pixels, and pixel_scales, in (y1,y2) coordinates
    :param source: source object that must contain a 2D kernel as source.profile
                   and weight (sum of the kernel) as source.weight
    :param positions: either a position in the magnification map, 
                      a collection of positions,
                      or an integer number of random positions
    :param return_pos: whether to also return positions used
    '''
    if np.ndim(source.profile) != 2:
        raise ValueError("source.profile must be a 2D array")
    
    if isinstance(positions, int):
        positions = util.random_position(ipm, source.profile, positions)
    else:
        positions = np.atleast_1d(positions)

        if not util.valid_positions(positions, ipm, source.profile):
            raise ValueError("provided positions do not lie within the necessary border")

    # cross correlate map and profile and normalize
    # contrary to what is commonly stated, microlensing is technically a cross 
    # correlation  with the source profile, NOT a convolution
    # the difference only matters for non-radially symmetric sources
    if not isinstance(ipm.magnifications, np.ndarray):
        vals = np.array(ipm.magnifications)
    else:
        vals = ipm.magnifications
    if not isinstance(source.profile, np.ndarray):
        kernel = np.array(source.profile)
    else:
        kernel = source.profile
    if not isinstance(source.weight, np.ndarray):
        weight = np.array(source.weight)
    else:
        weight = source.weight
    correlated_map = correlate(vals, kernel, mode='same', method='fft') / weight
    interp = util.interpolated_map(correlated_map, ipm.center, 
                                   ipm.half_length, ipm.num_pixels)
    
    magnifications = interp(positions)
    
    try:
        if return_pos:
            return magnifications.get(), positions.get()
        return magnifications.get()
    except AttributeError:
        if return_pos:
            return magnifications, positions
        return magnifications

def changing_source(ipm: IPM, source, positions = 1, return_pos: bool = False):
    '''
    Return the magnifications for the provided magnification map, 
    changing source profiles, and position(s)

    :param ipm: formally, an IPM instance that has ran and has a magnification
                map. in general, any object that has attributes magnifications,
                center, half_length, num_pixels, and pixel_scales, in (y1,y2) coordinates
    :param source: source object that must contain a list of 2D kernels as source.profiles
                   and weights (sums of the kernels) as source.weights.
                   kernels must all have the same shape
    :param positions: either a position in the magnification map, 
                      a collection of positions,
                      or an integer number of random positions
    :param return_pos: whether to also return positions used
    '''
    if np.ndim(source.profiles) != 3:
        raise ValueError("source.profiles must be a 3D array")

    source_shape = (np.shape(source.profiles)[2], np.shape(source.profiles)[1])
    
    if isinstance(positions, int):
        positions = util.random_position(ipm, source.profiles[-1], positions)
    else:
        positions = np.atleast_1d(positions)

        if not util.valid_positions(positions, ipm, source.profiles[-1]):
            raise ValueError("provided positions do not lie within the necessary border")

    interp = util.interpolated_map(ipm.magnifications, ipm.center, 
                                   ipm.half_length, ipm.num_pixels)
    
    # create 2D array of x and y coordinates of the source profile
    x, y = np.meshgrid(np.arange(source_shape[0]),
                       np.arange(source_shape[1]))
    # add 0.5 to offset to center of pixels
    x = x + 0.5
    y = y + 0.5
    # and recenter at (0,0) by subtracting half the profile size
    x = x - source_shape[0] / 2
    y = y - source_shape[1] / 2
    # convert from pixels to magnification map coordinates
    x = x * ipm.pixel_scales[0]
    y = y * ipm.pixel_scales[1]

    # add desired positions to source profile x,y values 
    # (i.e. recenter source profile at each position)
    # axes -2 and -1 are now the source profile axes
    # earlier axes are the collection of positions
    x = x + np.expand_dims(positions[...,0], (-2,-1))
    y = y + np.expand_dims(positions[...,1], (-2,-1))

    # add additional (time) axis for the changing profile
    # axis -3 is time axis, -2 and -1 are still source profile
    x = np.expand_dims(x, -3)
    y = np.expand_dims(y, -3)

    if not isinstance(source.profiles, np.ndarray):
        kernels = np.array(source.profiles)
    else:
        kernels = source.profiles
    if not isinstance(source.weights, np.ndarray):
        weights = np.array(source.weights)
    else:
        weights = source.weights

    # interpolate magnifications on the source profile location grid
    # in the transpose, we want all axes to stay the same 
    # EXCEPT the coordinates need to become the last axis
    # the roll makes sure the transpose moves axes like
    # [0,1,2,3,...n] -> [1,2,3,...n,0]
    # multiply and sum over source profile axes
    new_axes = np.roll(np.arange(x.ndim + 1),-1)
    try:
        new_axes = new_axes.get()
    except AttributeError:
        pass
    
    magnifications = (np.sum(kernels
                             * interp(np.transpose(np.array([x, y]), 
                                                   axes=new_axes)), 
                             axis=(-2,-1)) 
                      / weights) # and normalize!
    
    try:
        if return_pos:
            return magnifications.get(), positions.get()
        return magnifications.get()
    except AttributeError:
        if return_pos:
            return magnifications, positions
        return magnifications
