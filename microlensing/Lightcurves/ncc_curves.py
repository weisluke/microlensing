try:
    import cupy as np
    from cupyx.scipy.ndimage import minimum_filter, maximum_filter
except ImportError as error:
    print("cupy not installed. We do not recommend using these features without GPUs, as the min/max filter runtime is high")
    raise error

from microlensing.NCC.ncc import NCC
from microlensing.SourceProfiles.uniform_disk import UniformDisk, UniformDisks
from . import util


def constant_source(ncc: NCC, source, positions = 1, return_pos: bool = False):
    '''
    Return the minimum and maximum number of caustic crossings for the provided
    ncc map, constant source profile, and position(s)

    :param ncc: formally, an ncc instance that has ran and has a number of caustic crossings
                map. in general, any object that has attributes num_caustic_crossings,
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
    if not isinstance(source, UniformDisk):
        raise TypeError("source must be a UniformDisk")

    if isinstance(positions, int):
        positions = util.random_position(ncc, source.profile, positions)
    else:
        positions = np.atleast_1d(positions)

        if not util.valid_positions(positions, ncc, source.profile):
            raise ValueError("provided positions do not lie within the necessary border")

    if not isinstance(ncc.num_caustic_crossings, np.ndarray):
        vals = np.array(ncc.num_caustic_crossings)
    else:
        vals = ncc.num_caustic_crossings
    if not isinstance(source.profile, np.ndarray):
        kernel = np.array(source.profile)
    else:
        kernel = source.profile
    
    offset = np.min(vals) - 1
    vals = vals - offset

    # take the minimum and maximum values
    min_map = minimum_filter(vals, footprint=kernel, mode='constant', cval=-np.inf)
    min_map = min_map + offset
    max_map = maximum_filter(vals, footprint=kernel, mode='constant', cval=np.inf)
    max_map = max_map + offset

    interp = util.interpolated_map(min_map, ncc.center, 
                                   ncc.half_length, ncc.num_pixels)
    interp.method = 'nearest'
    ncc_min = interp(positions)

    interp = util.interpolated_map(max_map, ncc.center, 
                                   ncc.half_length, ncc.num_pixels)
    interp.method = 'nearest'
    ncc_max = interp(positions)
    
    try:
        if return_pos:
            return ncc_min.get(), ncc_max.get(), positions.get()
        return ncc_min.get(), ncc_max.get
    except AttributeError:
        if return_pos:
            return ncc_min, ncc_max, positions
        return ncc_min, ncc_max

def changing_source(ncc: NCC, source, positions = 1, return_pos: bool = False):
    '''
    Return the magnifications for the provided magnification map, 
    changing source profiles, and position(s)

    :param ncc: formally, an ncc instance that has ran and has a number of caustic crossings
                map. in general, any object that has attributes num_caustic_crossings,
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
    if not isinstance(source, UniformDisks):
        raise TypeError("source must be a UniformDisks")

    source_shape = (np.shape(source.profiles)[2], np.shape(source.profiles)[1])
    
    if isinstance(positions, int):
        positions = util.random_position(ncc, source.profiles[-1], positions)
    else:
        positions = np.atleast_1d(positions)

        if not util.valid_positions(positions, ncc, source.profiles[-1]):
            raise ValueError("provided positions do not lie within the necessary border")

    if not isinstance(ncc.num_caustic_crossings, np.ndarray):
        vals = np.array(ncc.num_caustic_crossings)
    else:
        vals = ncc.num_caustic_crossings
    if not isinstance(source.profiles, np.ndarray):
        kernels = np.array(source.profiles)
    else:
        kernels = source.profiles
    
    offset = np.min(vals) - 1
    vals = vals - offset

    interp = util.interpolated_map(vals, ncc.center, 
                                   ncc.half_length, ncc.num_pixels)
    interp.method = 'nearest'
    
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
    x = x * ncc.pixel_scales[0]
    y = y * ncc.pixel_scales[1]

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

    # interpolate values on the source profile location grid
    # in the transpose, we want all axes to stay the same 
    # EXCEPT the coordinates need to become the last axis
    # the roll makes sure the transpose moves axes like
    # [0,1,2,3,...n] -> [1,2,3,...n,0]
    new_axes = np.roll(np.arange(x.ndim + 1),-1)
    try:
        new_axes = new_axes.get()
    except AttributeError:
        pass
    
    # take the minimum and maximum values
    res = kernels * interp(np.transpose(np.array([x, y]), axes=new_axes))
    ncc_max = np.max(res, axis=(-2,-1)) + offset
    res = np.where(res > 0, res, np.inf)
    ncc_min = np.min(res, axis=(-2,-1)) + offset
    
    try:
        if return_pos:
            return ncc_min.get(), ncc_max.get(), positions.get()
        return ncc_min.get(), ncc_max.get()
    except AttributeError:
        if return_pos:
            return ncc_min, ncc_max, positions
        return ncc_min, ncc_max
