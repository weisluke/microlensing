import numpy as np
import struct


def read_params(fname: str):
    '''
    Read in a whitespace delimited text file of (key value)
    lines and return a dictionary mapping keys to values

    :param fname: name of the file to read
    '''
    params = {}
    with open(fname) as f:
        for line in f:
            (key, val) = line.split()
            try:
                params[key] = float(val)
                if params[key] == int(params[key]):
                    params[key] = int(params[key])
            except ValueError:
                params[key] = val
    return params


def read_stars(fname: str, dtype=np.float32):
    '''
    Read in a binary file of star information

    :param fname: name of the file to read
    :param dtype: data type for the file. default is np.float32

    :return stars: array of stars (x1, x2, mass)
    :return rectangular: bool of whether the star field is rectangular or
                         circular
    :return corner: corner of the star field (x1, x2). if circular, (rad, 0)
    :return theta_star: Einstein radius of a unit mass point lens in arbitrary
                        units
    '''

    with open(fname) as f:
        nstars = np.fromfile(f, dtype=np.int32, count=1)[0]
        rectangular = np.fromfile(f, dtype=np.int32, 
                                  count=1).astype(np.bool_)[0]
        corner = np.fromfile(f, dtype=dtype, count=2)
        theta_star = np.fromfile(f, dtype=dtype, count=1)[0]
        stars = np.fromfile(f, dtype=dtype)
        stars = stars.reshape(nstars, 3)

    return stars, rectangular, corner, theta_star


def write_stars(fname: str, stars, rectangular: bool, corner,
                theta_star: float, dtype=np.float32):
    '''
    Write a binary file of star information

    :param fname: name of the file to write
    :param stars: array of stars in the form (x1, x2, mass)
    :param rectangular: bool of whether the star field is rectangular or
                        circular
    :param corner: array of 2 numbers representing the (x1, x2) corner of the
                   star field in the case of rectangular star fields.
                   if the star field is circular, this should be (rad, 0)
    :param theta_star: Einstein radius of a unit mass point lens in arbitrary
                       units. typically 1
    :param dtype: data type for the file. default is np.float32
    '''
    if not fname.endswith('.bin'):
        raise ValueError('fname must be a .bin file')
    
    if not isinstance(stars, np.ndarray):
        stars = np.array(stars)
    if stars.ndim != 2:
        raise ValueError("stars is not a 2D array")
    
    if dtype == np.float64:
        dtype = 'd'
    elif dtype == np.float32:
        dtype = 'f'
    else:
        raise ValueError("dtype must be np.float32 or np.float64")

    if rectangular:
        rectangular = 1
    else:
        rectangular = 0

    with open(fname, 'wb') as f:
        s = struct.pack('i', stars.shape[0])
        f.write(s)
        s = struct.pack('i', rectangular)
        f.write(s)
        s = struct.pack(dtype * 2, *corner)
        f.write(s)
        s = struct.pack(dtype, theta_star)
        f.write(s)
        s = struct.pack(dtype * len(stars.ravel()), *stars.ravel())
        f.write(s)
        f.close()


def read_array(fname: str, dtype, is_complex: bool = False):
    '''
    Read in a binary file of a 2d array of numbers

    :param fname: name of the file to read
    :param dtype: type for the array
    :param is_complex: bool, whether the numbers are complex or not
                       default is False
    '''
    with open(fname) as f:
        nrows, ncols = np.fromfile(f, dtype=np.int32, count=2)
        dat = np.fromfile(f, dtype=dtype)
        if is_complex:
            dat = dat.reshape(nrows, ncols, 2)
        else:
            dat = dat.reshape(nrows, ncols)

    return dat


def read_ragged_array(fname: str, dtype, is_complex: bool = False):
    '''
    Read in a binary file of a ragged array of numbers

    :param fname: name of the file to read
    :param dtype: type for the array
    :param is_complex: bool, whether the numbers are complex or not
                       default is False
    '''
    with open(fname) as f:
        res = []
        n_members = np.fromfile(f, dtype=np.int32, count=1)[0]
        for i in range(n_members):
            n = np.fromfile(f, dtype=np.int32, count=1)[0]
            if is_complex:
                dat = np.fromfile(f, dtype=dtype, count=n * 2)
                dat = dat.reshape(n, 2)
            else:
                dat = np.fromfile(f, dtype=dtype, count=n)
            res.append(dat)

    return res


def read_hist(fname: str):
    '''
    Read in a whitespace delimited text file of integer
    (value, num_pixels) lines

    :param fname: name of the file to read
    '''
    return np.loadtxt(fname, dtype=np.int32)
