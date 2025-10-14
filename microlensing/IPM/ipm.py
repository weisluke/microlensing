from . import lib_ipm
from . import lib_ipm_double
from microlensing.Stars.stars import Stars

import numpy as np
import matplotlib.axes


class IPM(object):
    def __init__(self, kappa_tot: float = None, shear: float = None, smooth_fraction: float = None, kappa_star: float = None, 
                 theta_star: float = None, mass_function: str = None, m_solar: float = None, m_lower: float = None, m_upper: float = None,
                 light_loss: float = None, rectangular: bool = None, approx: bool = None, safety_scale: float = None,
                 starfile: str = None, center_y1: float = None, center_y2: float = None, half_length_y1: float = None, half_length_y2: float = None,
                 num_pixels_y1: int = None, num_pixels_y2: int = None, num_rays_y: int = None, random_seed: int = None,
                 write_stars: bool = False, write_maps: bool = False, write_parities: bool = False, write_histograms: bool = False,
                 outfile_prefix: str = None, verbose: int = 0, is_double: bool = False):
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
        :param center_y1: y1 coordinate of the center of the magnification map
        :param center_y2: y2 coordinate of the center of the magnification map
        :param half_length_y1: y1 extent of the half-length of the magnification map
        :param half_length_y2: y2 extent of the half_length of the magnification map
        :param num_pixels_y1: number of pixels for the y1 axis
        :param num_pixels_y2: number of pixels for the y2 axis
        :param num_rays_y: average number of rays per pixel in the absence of lensing
        :param random_seed: random seed for star field generation. A value of 0 is reserved for star input files
        :param write_stars: whether to write stars or not
        :param write_maps: whether to write magnification maps or not
        :param write_parities: whether to write parity specific magnification maps or not
        :param write_histograms: whether to write histograms or not
        :param outfile_prefix: prefix to be used in output file names
        :param verbose: verbosity level of messages. must be 0, 1, 2, or 3
        :param is_double: whether to use float or double library
        '''
        if is_double:
            self.lib = lib_ipm_double.lib
        else:
            self.lib = lib_ipm.lib

        self.obj = self.lib.IPM_init()
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
            self.random_seed = random_seed

        self.light_loss = light_loss
        self.approx = approx
        self.safety_scale = safety_scale
        
        self.starfile = starfile
        
        self.center_y1 = center_y1
        self.center_y2 = center_y2
        self.half_length_y1 = half_length_y1
        self.half_length_y2 = half_length_y2
        self.num_pixels_y1 = num_pixels_y1
        self.num_pixels_y2 = num_pixels_y2
        
        self.num_rays_y = num_rays_y
            
        self.write_stars = write_stars
        self.write_maps = write_maps
        self.write_parities = write_parities
        self.write_histograms = write_histograms
        
        self.outfile_prefix = outfile_prefix

    def __del__(self):
        self.lib.IPM_delete(self.obj)
        
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
    def mu1(self):
        return 1 / (1 - self.kappa_tot + self.shear)

    @property
    def mu2(self):
        return 1 / (1 - self.kappa_tot - self.shear)

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
    def num_rays_y(self):
        return self.lib.get_num_rays_y(self.obj)
    
    @num_rays_y.setter
    def num_rays_y(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("num_rays_y must be >= 1")
            self.lib.set_num_rays_y(self.obj, value)

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
    def write_maps(self):
        return bool(self.lib.get_write_maps(self.obj))
    
    @write_maps.setter
    def write_maps(self, value):
        if value is not None:
            self.lib.set_write_maps(self.obj, int(value))

    @property
    def write_parities(self):
        return bool(self.lib.get_write_parities(self.obj))
    
    @write_parities.setter
    def write_parities(self, value):
        if value is not None:
            self.lib.set_write_parities(self.obj, int(value))

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

    @property
    def num_stars(self):
        return self.lib.get_num_stars(self.obj)
    
    @property
    def corner(self):
        return (self.lib.get_corner_x1(self.obj), self.lib.get_corner_x2(self.obj))

    def run(self):
        if not self.lib.run(self.obj, self.verbose):
            raise Exception("Error running IPM")
        
        self.magnifications = np.ctypeslib.as_array(self.lib.get_pixels(self.obj), 
                                                    shape=(self.num_pixels_y2,
                                                           self.num_pixels_y1)).copy()
        if self.write_parities:
            self.magnifications_minima = np.ctypeslib.as_array(self.lib.get_pixels_minima(self.obj), 
                                                               shape=(self.num_pixels_y2,
                                                                      self.num_pixels_y1)).copy()
            self.magnifications_saddles = np.ctypeslib.as_array(self.lib.get_pixels_saddles(self.obj), 
                                                                shape=(self.num_pixels_y2,
                                                                       self.num_pixels_y1)).copy()
        else:
            self.magnifications_minima = None
            self.magnifications_saddles = None

        self.stars = Stars(np.ctypeslib.as_array(self.lib.get_stars(self.obj),
                                                 shape=(self.num_stars, 3)).copy(),
                           self.rectangular, self.corner, self.theta_star)
        
    @property
    def t_shoot_cells(self):
        return self.lib.get_t_shoot_cells(self.obj)

    @property
    def magnitudes(self):
        return -2.5 * np.log10(self.magnifications / np.abs(self.mu_ave))
    
    @property
    def magnitudes_minima(self):
        if self.magnifications_minima is None:
            raise ValueError("magnifications_minima is None")
        return -2.5 * np.log10(self.magnifications_minima / np.abs(self.mu_ave))
    
    @property
    def magnitudes_saddles(self):
        if self.magnifications_saddles is None:
            raise ValueError("magnifications_saddles is None")
        return -2.5 * np.log10(self.magnifications_saddles / np.abs(self.mu_ave))
    
    @property
    def extent(self):
        return np.array([[(self.center[0] - self.half_length[0]),
                          (self.center[0] + self.half_length[0])],
                         [(self.center[1] - self.half_length[1]),
                          (self.center[1] + self.half_length[1])]])
    
    def save(self):
        if not self.lib.save(self.obj, self.verbose):
            raise Exception("Error saving IPM")

    def plot(self, ax: matplotlib.axes.Axes, cmap='viridis', plot_magnitudes=False, parity=None, **kwargs):
        if parity not in [None, '+', '-']:
            raise ValueError("parity must be None, '+', or '-'")

        if plot_magnitudes:
            if parity is None:
                dat = self.magnitudes
            elif parity == "+":
                dat = self.magnitudes_minima
            else:
                dat = self.magnitudes_saddles
            cmap = cmap + '_r'
        else:
            if parity is None:
                dat = np.log10(self.magnifications)
            elif parity == "+":
                dat = np.log10(self.magnifications_minima)
            else:
                dat = np.log10(self.magnifications_saddles)

        # if min and max aren't given, use the inner 95 percent of the data
        where = np.isfinite(dat)
        if 'vmin' not in kwargs.keys():
            kwargs['vmin'] = np.percentile(dat[where], 2.5)
        if 'vmax' not in kwargs.keys():
            kwargs['vmax'] = np.percentile(dat[where], 97.5)

        img = ax.imshow(dat, extent=self.extent.ravel(), cmap=cmap, **kwargs)
        cbar = ax.get_figure().colorbar(img)
        if plot_magnitudes:
            if parity is None:
                cbar.set_label('microlensing $\\Delta m$ (magnitudes)')
            elif parity == "+":
                cbar.set_label('microlensing $\\Delta m_\\text{microminima}$ (magnitudes)')
            else:
                cbar.set_label('microlensing $\\Delta m_\\text{microsaddles}$ (magnitudes)')

            cbar.ax.invert_yaxis()
        else:
            if parity is None:
                cbar.set_label('$\\log(\\mu)$')
            elif parity == "+":
                cbar.set_label('$\\log(\\mu_\\text{microminima})$')
            else:
                cbar.set_label('$\\log(\\mu_\\text{microsaddles})$')

        if self.theta_star == 1:
            ax.set_xlabel('$y_1 / \\theta_★$')
            ax.set_ylabel('$y_2 / \\theta_★$')
        else:
            ax.set_xlabel('$y_1$')
            ax.set_ylabel('$y_2$')

    def plot_hist(self, ax: matplotlib.axes.Axes, bins=None, dw=0.01, plot_magnitudes=False, parity=None, **kwargs):
        if parity not in [None, '+', '-']:
            raise ValueError("parity must be None, '+', or '-'")
        
        if plot_magnitudes:
            if parity is None:
                dat = self.magnitudes.ravel()
            elif parity == "+":
                dat = self.magnitudes_minima.ravel()
            else:
                dat = self.magnitudes_saddles.ravel()
        else:
            if parity is None:
                dat = np.log10(self.magnifications.ravel())
            elif parity == "+":
                dat = np.log10(self.magnifications_minima.ravel())
            else:
                dat = np.log10(self.magnifications_saddles.ravel())

        if bins is None:
            where = np.isfinite(dat)
            vmin, vmax = (np.min(dat[where]), np.max(dat[where]))
            bins = np.arange(vmin - dw, vmax + dw, dw)

        ax.hist(dat, bins=bins, density=True, **kwargs)

        if plot_magnitudes:
            if parity is None:
                ax.set_xlabel('microlensing $\\Delta m$ (magnitudes)')
                ax.set_ylabel('p($\\Delta m$)')
            elif parity == "+":
                ax.set_xlabel('microlensing $\\Delta m_\\text{microminima}$ (magnitudes)')
                ax.set_ylabel('p($\\Delta m_\\text{microminima}$)')
            else:
                ax.set_xlabel('microlensing $\\Delta m_\\text{microsaddles}$ (magnitudes)')
                ax.set_ylabel('p($\\Delta m_\\text{microsaddles}$)')

            ax.invert_xaxis()
        else:
            if parity == None:
                ax.set_xlabel('$\\log(\\mu)$')
                ax.set_ylabel('p($\\log(\\mu)$)')
            elif parity == "+":
                ax.set_xlabel('$\\log(\\mu_\\text{microminima})$')
                ax.set_ylabel('p($\\log(\\mu_\\text{microminima})$)')
            else:
                ax.set_xlabel('$\\log(\\mu_\\text{microsaddles})$')
                ax.set_ylabel('p($\\log(\\mu_\\text{microsaddles})$)')
