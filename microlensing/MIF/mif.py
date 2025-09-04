from . import lib_mif
from microlensing.Stars.stars import Stars
from . import plotting

import numpy as np
import matplotlib.axes
from scipy.interpolate import RegularGridInterpolator


class MIF(object):
    def __init__(self, kappa_tot: float = None, shear: float = None, smooth_fraction: float = None, kappa_star: float = None, 
                 theta_star: float = None, mass_function: str = None, m_solar: float = None, m_lower: float = None, m_upper: float = None,
                 light_loss: float = None, rectangular: bool = None, approx: bool = None, safety_scale: float = None,
                 starfile: str = None, y1: float = None, y2: float = None, v1: float = None, v2: float = None, random_seed: int = None,
                 write_stars: bool = False, write_images: bool = False, write_image_lines: bool = False, write_magnifications: bool = False,
                 outfile_prefix: str = None, verbose: int = 0):
        '''
        :param kappa_tot: total convergence
        :param shear: shear
        :param smooth_fraction: fraction of convergence due to smoothly distributed mass
        :param kappa_star: convergence in point mass lenses
        :param theta_star: Einstein radius of a unit mass point lens in arbitrary units
        :param mass_function: mass function to use for the point mass lenses. Options are: equal, uniform, Salpeter, Kroupa, and optical_depth
        :param m_solar: solar mass in arbitrary units
        :param m_lower: lower mass cutoff in solar mass units
        :param m_upper: upper mass cutoff in solar mass units
        :param light_loss: Allowed average fraction of light lost due to scatter by the microlenses in the large deflection limit
        :param rectangular: whether the star field is rectangular (True) or circular (False)
        :param approx: whether terms for alpha_smooth should be approximated (True) or exact (False)
        :param safety_scale: ratio of the size of the star field to the size of the shooting rectangle
        :param starfile: the location of a binary file containing values for num_stars, rectangular, corner, theta_star, and the star positions and masses.
                         A whitespace delimited text file where each line contains the x1 and x2 coordinates and the mass of a microlens, in units where 
                         theta_star = 1, is also accepted. If provided, this takes precedence for all star information
        :param y1: y1 coordinate of the source or position that the source travels through
        :param y2: y2 coordinate of the source or position that the source travels through
        :param v1: y1 coordinate of the source velocity
        :param v2: y2 coordinate of the source velocity
        :param random_seed: random seed for star field generation. A value of 0 is reserved for star input files
        :param write_stars: whether to write stars or not
        :param write_images: whether to write image positions or not
        :param write_image_lines: whether to write image line positions or not
        :param write_magnifications: whether to write magnifications or not
        :param outfile_prefix: prefix to be used in output file names
        :param verbose: verbosity level of messages. must be 0, 1, 2, or 3
        '''
        self.lib = lib_mif.lib

        self.obj = self.lib.MIF_init()
        self.verbose = verbose

        self.kappa_tot = kappa_tot
        self.shear = shear

        if starfile is None:
            if smooth_fraction is not None and kappa_star is None:
                self.kappa_star = (1 - smooth_fraction) * self.kappa_tot
            self.kappa_star = kappa_star
            self.theta_star = theta_star
            self.mass_function = mass_function
            self.m_solar = m_solar
            self.m_lower = m_lower
            self.m_upper = m_upper
            if self.m_lower > self.m_upper:
                raise ValueError("m_lower must be <= m_upper")
            self.rectangular = rectangular
            assert not self.rectangular # TODO: fix cuda code so rectangular can be true
            self.random_seed = random_seed

        self.light_loss = light_loss
        self.approx = approx
        self.safety_scale = safety_scale
        
        self.starfile = starfile
        
        self.y1 = y1
        self.y2 = y2
        self.v1 = v1
        self.v2 = v2
            
        self.write_stars = write_stars
        self.write_images = write_images
        self.write_image_lines = write_image_lines
        self.write_magnifications = write_magnifications
        
        self.outfile_prefix = outfile_prefix

    def __del__(self):
        self.lib.MIF_delete(self.obj)
        
    @property
    def kappa_tot(self):
        return self.lib.get_kappa_tot(self.obj)
    
    @kappa_tot.setter
    def kappa_tot(self, value):
        if value is not None:
            if value < 0:
                raise ValueError("kappa_tot must be >= 0")
            self.lib.set_kappa_tot(self.obj, value)

    @property
    def shear(self):
        return self.lib.get_shear(self.obj)
    
    @shear.setter
    def shear(self, value):
        if value is not None:
            self.lib.set_shear(self.obj, value)

    @property
    def mu_ave(self):
        return 1 / ((1 - self.kappa_tot)**2 - self.shear**2)

    @property
    def kappa_star(self):
        return self.lib.get_kappa_star(self.obj)
    
    @kappa_star.setter
    def kappa_star(self, value):
        if value is not None:
            if value < 0:
                raise ValueError("kappa_star must be >= 0")
            elif value > self.kappa_tot:
                raise ValueError("kappa_star must be <= kappa_tot")
            self.lib.set_kappa_star(self.obj, value)

    @property
    def stellar_fraction(self):
        return self.kappa_star / self.kappa_tot

    @property
    def smooth_fraction(self):
        return 1 - self.kappa_star / self.kappa_tot

    @property
    def theta_star(self):
        return self.lib.get_theta_star(self.obj)
    
    @theta_star.setter
    def theta_star(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("theta_star must be > 0")
            self.lib.set_theta_star(self.obj, value)

    @property
    def mass_function(self):
        return self.lib.get_mass_function(self.obj).decode('utf-8')
    
    @mass_function.setter
    def mass_function(self, value):
        if value is not None:
            if value.lower() not in ['equal', 'uniform', 'salpeter', 'kroupa', 'optical_depth']:
                raise ValueError("mass_function must be equal, uniform, Salpeter, Kroupa, or optical_depth")
            self.lib.set_mass_function(self.obj, value.lower().encode('utf-8'))

    @property
    def m_solar(self):
        return self.lib.get_m_solar(self.obj)
    
    @m_solar.setter
    def m_solar(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("m_solar must be > 0")
            self.lib.set_m_solar(self.obj, value)

    @property
    def m_lower(self):
        return self.lib.get_m_lower(self.obj)
    
    @m_lower.setter
    def m_lower(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("m_lower must be > 0")
            self.lib.set_m_lower(self.obj, value)

    @property
    def m_upper(self):
        return self.lib.get_m_upper(self.obj)
    
    @m_upper.setter
    def m_upper(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("m_upper must be > 0")
            self.lib.set_m_upper(self.obj, value)

    @property
    def light_loss(self):
        return self.lib.get_light_loss(self.obj)
    
    @light_loss.setter
    def light_loss(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("light_loss must be > 0")
            elif value > 0.01:
                raise ValueError("light_loss must be <= 0.01")
            self.lib.set_light_loss(self.obj, value)

    @property
    def rectangular(self):
        return bool(self.lib.get_rectangular(self.obj))
    
    @rectangular.setter
    def rectangular(self, value):
        if value is not None:
            self.lib.set_rectangular(self.obj, int(value))

    @property
    def approx(self):
        return bool(self.lib.get_approx(self.obj))
    
    @approx.setter
    def approx(self, value):
        if value is not None:
            self.lib.set_approx(self.obj, int(value))

    @property
    def safety_scale(self):
        return self.lib.get_safety_scale(self.obj)
    
    @safety_scale.setter
    def safety_scale(self, value):
        if value is not None:
            if value < 1.1:
                raise ValueError("safety_scale must be >= 1.1")
            self.lib.set_safety_scale(self.obj, value)

    @property
    def starfile(self):
        return self.lib.get_starfile(self.obj).decode('utf-8')
    
    @starfile.setter
    def starfile(self, value):
        if value is not None:
            self.lib.set_starfile(self.obj, value.encode('utf-8'))

    @property
    def y1(self):
        return self.lib.get_y1(self.obj)
    
    @y1.setter
    def y1(self, value):
        if value is not None:
            self.lib.set_y1(self.obj, value)

    @property
    def y2(self):
        return self.lib.get_y2(self.obj)
    
    @y2.setter
    def y2(self, value):
        if value is not None:
            self.lib.set_y2(self.obj, value)

    @property
    def w0(self):
        return (self.y1, self.y2)

    @property
    def z0(self):
        return (self.y1 / (1 - self.kappa_tot + self.shear), 
                self.y2 / (1 - self.kappa_tot - self.shear))

    @property
    def v1(self):
        return self.lib.get_v1(self.obj)
    
    @v1.setter
    def v1(self, value):
        if value is not None:
            self.lib.set_v1(self.obj, value)

    @property
    def v2(self):
        return self.lib.get_v2(self.obj)
    
    @v2.setter
    def v2(self, value):
        if value is not None:
            self.lib.set_v2(self.obj, value)

    @property
    def v(self):
        return (self.v1, self.v2)

    @property
    def random_seed(self):
        return self.lib.get_random_seed(self.obj)
    
    @random_seed.setter
    def random_seed(self, value):
        if value is not None:
            self.lib.set_random_seed(self.obj, value)

    @property
    def write_stars(self):
        return bool(self.lib.get_write_stars(self.obj))
    
    @write_stars.setter
    def write_stars(self, value):
        if value is not None:
            self.lib.set_write_stars(self.obj, int(value))

    @property
    def write_images(self):
        return bool(self.lib.get_write_images(self.obj))
    
    @write_images.setter
    def write_images(self, value):
        if value is not None:
            self.lib.set_write_images(self.obj, int(value))

    @property
    def write_image_lines(self):
        return bool(self.lib.get_write_image_lines(self.obj))
    
    @write_image_lines.setter
    def write_image_lines(self, value):
        if value is not None:
            self.lib.set_write_image_lines(self.obj, int(value))

    @property
    def write_magnifications(self):
        return bool(self.lib.get_write_magnifications(self.obj))
    
    @write_magnifications.setter
    def write_magnifications(self, value):
        if value is not None:
            self.lib.set_write_magnifications(self.obj, int(value))

    @property
    def outfile_prefix(self):
        return self.lib.get_outfile_prefix(self.obj).decode('utf-8')
    
    @outfile_prefix.setter
    def outfile_prefix(self, value):
        if value is not None:
            self.lib.set_outfile_prefix(self.obj, value.encode('utf-8'))

    @property
    def num_stars(self):
        return self.lib.get_num_stars(self.obj)
    
    @property
    def corner(self):
        return (self.lib.get_corner_x1(self.obj), self.lib.get_corner_x2(self.obj))

    def run(self):
        if not self.lib.run(self.obj, self.verbose):
            raise Exception("Error running MIF")
        
        self.images = np.ctypeslib.as_array(self.lib.get_images(self.obj),
                                            shape=(self.lib.get_num_images(self.obj),2)).copy()
        self.images_inv_mags = np.ctypeslib.as_array(self.lib.get_images_mags(self.obj),
                                                     shape=(self.lib.get_num_images(self.obj),2,2)).copy()
        
        m11 = self.images_inv_mags[:,0,0] + self.images_inv_mags[:,1,0]
        m12 = self.images_inv_mags[:,1,1] - self.images_inv_mags[:,0,1]
        m21 = self.images_inv_mags[:,0,1] + self.images_inv_mags[:,1,1]
        m22 = self.images_inv_mags[:,0,0] - self.images_inv_mags[:,1,0]

        self.images_inv_mags = np.transpose([[m11,m12],[m21,m22]],(2,0,1))
        self.images_mags = 1 / np.linalg.det(self.images_inv_mags)
        
        if self.write_image_lines:
            n_image_lines = self.lib.get_num_image_lines(self.obj)
            image_lines_lengths = np.ctypeslib.as_array(self.lib.get_image_lines_lengths(self.obj),
                                                        shape=(n_image_lines,)).copy()
            
            self.image_lines = np.ctypeslib.as_array(self.lib.get_image_lines(self.obj),
                                                    shape=(self.lib.get_total_image_lines_length(self.obj),2)).copy()
            self.image_lines = np.split(self.image_lines, np.cumsum(image_lines_lengths))
            
            self.source_lines = np.ctypeslib.as_array(self.lib.get_source_lines(self.obj),
                                                      shape=(self.lib.get_total_image_lines_length(self.obj),2)).copy()
            self.source_lines = np.split(self.source_lines, np.cumsum(image_lines_lengths))
            
            self.image_lines_mags = np.ctypeslib.as_array(self.lib.get_image_lines_mags(self.obj),
                                                          shape=(self.lib.get_total_image_lines_length(self.obj),)).copy()
            self.image_lines_mags = np.split(self.image_lines_mags, np.cumsum(image_lines_lengths))

            self.image_lines = self.image_lines[:-1]
            self.source_lines = self.source_lines[:-1]
            self.image_lines_mags = self.image_lines_mags[:-1]
        else:
            self.image_lines = None
            self.source_lines = None
            self.image_lines_mags = None
        
        self.stars = Stars(np.ctypeslib.as_array(self.lib.get_stars(self.obj),
                                                 shape=(self.num_stars, 3)).copy(),
                           self.rectangular, self.corner, self.theta_star)
    
    def save(self):
        if not self.lib.save(self.obj, self.verbose):
            raise Exception("Error saving MIF")

    # contours containing 90, 99, and 99.9 % of the magnification on average
    @property
    def c90(self):
        return np.array([[(self.y1 - self.stars.r90) / np.abs(1 - self.kappa_tot + self.shear),
                          (self.y1 + self.stars.r90) / np.abs(1 - self.kappa_tot + self.shear)],
                         [(self.y2 - self.stars.r90) / np.abs(1 - self.kappa_tot - self.shear),
                          (self.y2 + self.stars.r90) / np.abs(1 - self.kappa_tot - self.shear)]])
    @property
    def c99(self):
        return np.array([[(self.y1 - self.stars.r99) / np.abs(1 - self.kappa_tot + self.shear),
                          (self.y1 + self.stars.r99) / np.abs(1 - self.kappa_tot + self.shear)],
                         [(self.y2 - self.stars.r99) / np.abs(1 - self.kappa_tot - self.shear),
                          (self.y2 + self.stars.r99) / np.abs(1 - self.kappa_tot - self.shear)]])
    @property
    def c999(self):
        return np.array([[(self.y1 - self.stars.r999) / np.abs(1 - self.kappa_tot + self.shear),
                          (self.y1 + self.stars.r999) / np.abs(1 - self.kappa_tot + self.shear)],
                         [(self.y2 - self.stars.r999) / np.abs(1 - self.kappa_tot - self.shear),
                          (self.y2 + self.stars.r999) / np.abs(1 - self.kappa_tot - self.shear)]])

    @property
    def bins(self):
        try:
            return self._bins
        except AttributeError:
            dw = 0.001 # arbitrary step size in source plane

            # x = [np.dot(what - self.w0, self.v / np.linalg.norm(self.v)) for what in self.source_lines]
            # x_min = np.min([np.min(what) for what in x])
            # x_max = np.max([np.max(what) for what in x])
            x_min = -25
            x_max = 25

            self._bins = np.arange(x_min, x_max, dw)
            self._bins = np.insert(self._bins, self._bins.size, x_max)
            return self._bins
        
    @property
    def distances(self):
        try:
            return self._distances
        except AttributeError:
            self._distances = (self.bins[:-1] + self.bins[1:]) / 2
            return self._distances
        
    @property
    def magnifications(self):
        try:
            return self._magnifications
        except AttributeError:
            self._magnifications = np.zeros(self.distances.shape)

            for src, mu in zip(self.source_lines, self.image_lines_mags):
                tmp_x = np.dot(src - self.w0, self.v / np.linalg.norm(self.v))
                tmp_y = np.abs(mu)

                # find where direction changes in source plane
                sgn = np.sign(tmp_x[1:] - tmp_x[:-1])
                sgn = np.insert(sgn, 0, sgn[0])

                # riffle so that slices start and end where sgn changes
                where = np.where(sgn[1:] != sgn[:-1])[0]
                where = np.ravel(np.column_stack((where + 1, where)))

                # split and remove empty slices
                tmp_x = np.split(tmp_x, where)
                tmp_x = [what for what in tmp_x if not len(what)==0]

                tmp_y = np.split(tmp_y, where)
                tmp_y = [what for what in tmp_y if not len(what)==0]

                # interpolate and evaluate at bin centers
                for a, b in zip(tmp_x, tmp_y):
                    interp = RegularGridInterpolator([a], b, bounds_error=False, fill_value=0)
                    self._magnifications += interp(self._distances)
            
            return self._magnifications
        
    @property
    def magnitudes(self):
        return -2.5 * np.log10(np.abs(self.magnifications / self.mu_ave))

    def plot_images(self, ax: matplotlib.axes.Axes, s=1,
                    is_ellipse=True, log_area=False, mu_min=10**-3,
                    **kwargs):

        ax.add_collection(plotting.Images(self.images, self.images_inv_mags,
                                          s, is_ellipse, log_area, mu_min))

    def plot_image_lines(self, ax: matplotlib.axes.Axes, color='black', **kwargs):

        for what in self.image_lines:
            ax.plot(*what.T, color=color, **kwargs)

    def plot_lightcurve(self, ax: matplotlib.axes.Axes, plot_magnitudes=False, **kwargs):
        if plot_magnitudes:
            dat = self.magnitudes
        else:
            dat = np.log10(self.magnifications)

        ax.plot(self.distances / self.theta_star, dat, **kwargs)

        ax.set_xlabel('(distance from $w_0$) / $\\theta_★$')
        if plot_magnitudes:
            ax.set_ylabel('microlensing $\\Delta m$ (magnitudes)')
            ax.invert_yaxis()
        else:
            ax.set_ylabel('$\\log\\mu$')

