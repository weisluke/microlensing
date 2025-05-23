try:
    import cupy as np
    from cupyx.scipy.interpolate import RegularGridInterpolator
except ImportError:
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator


def pixel_to_point(pixel, center, half_length, num_pixels):
    '''
    convert a pixel coordinate into a point in the source plane
    parameters may be 1D or 2D values

    :param pixel: pixel coordinate or array of pixel coordinates
    :param center: center of the map
    :param half_length: half_length of the map
    :param num_pixels: number of pixels of the map along the axes
    '''
    if not isinstance(pixel, np.ndarray):
        pixel = np.array(pixel)
    if not isinstance(center, np.ndarray):
        center = np.array(center)
    if not isinstance(half_length, np.ndarray):
        half_length = np.array(half_length)
    if not isinstance(num_pixels, np.ndarray):
        num_pixels = np.array(num_pixels)
    return pixel * 2 * half_length / num_pixels - half_length + center

def get_borders(map, kernel):
    '''
    return the inner border of the map after removing a region the size of the
    provided kernel around the edge

    :param map: an object that has attributes center, half_length, num_pixels,
                and pixel_scales, in (y1,y2) coordinates
    :param kernel: kernel of the source profile    
    :return: (y1_min, y1_max), (y2_min, y2_max)
    '''
    # borders of the valid region in pixel coordinates
    x = (np.shape(kernel)[1] / 2, map.num_pixels[0] - np.shape(kernel)[1] / 2)
    y = (np.shape(kernel)[0] / 2, map.num_pixels[1] - np.shape(kernel)[0] / 2)

    return pixel_to_point(np.transpose(np.array([x, y])), map.center, 
                          map.half_length, map.num_pixels).T

def random_position(map, kernel, num: int = 1):
    '''
    return a number of random valid positions in the map

    :param map: an object that has attributes center, half_length, num_pixels,
                and pixel_scales, in (y1,y2) coordinates
    :param kernel: kernel of the source profile
    :param num: number of random positions to return
    '''
    # borders of the valid region
    xlim, ylim = get_borders(map, kernel)

    rng = np.random.default_rng()
    x, y = rng.uniform(*xlim, num), rng.uniform(*ylim, num)

    # squeeze to remove unnecessary dimensions if num=1
    return np.squeeze(np.transpose(np.array([x, y])))

def valid_positions(positions, map, kernel):
    '''
    determine whether positions within the magnification map are valid
    to avoid edge effects for the given kernel

    :param positions: positions to check

    :param map: an object that has attributes center, half_length, num_pixels,
                and pixel_scales, in (y1,y2) coordinates
    :param kernel: kernel of the source profile
    '''
    if np.ndim(kernel) != 2:
        raise ValueError("kernel must be a 2D array")
    
    if positions.shape[-1] != 2:
        raise ValueError("There are not 2 coordinates per position")
    
    # borders of the valid region
    xlim, ylim = get_borders(map, kernel)
    x, y = positions[...,0], positions[...,1]

    return (np.all(x > xlim[0]) and np.all(x < xlim[1])
            and np.all(y > ylim[0]) and np.all(y < ylim[1]))

def interpolated_map(values, center, half_length, num_pixels):
    '''
    :param values: values to interpolate
    :param center: center of the map
    :param half_length: half_length of the map
    :param num_pixels: number of pixels of the map along the axes
    '''
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # add 0.5 to offset to center of pixels for interpolation
    x, y = np.arange(num_pixels[0]) + 0.5, np.arange(num_pixels[1]) + 0.5

    x = pixel_to_point(x, center[0], half_length[0], num_pixels[0])
    y = pixel_to_point(y, center[1], half_length[1], num_pixels[1])

    # reverse magnification map since (0,0) in pixel coordinates for 
    # python arrays is top left corner and we want it to be bottom left
    return RegularGridInterpolator((x,y), values[::-1].T,
                                   bounds_error=False, fill_value=None)
