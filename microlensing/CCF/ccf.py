from . import lib_ccf
from microlensing.Stars.stars import Stars

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


class CCF(object):
    def __init__(self, kappa_tot: float = None, shear: float = None, smooth_fraction: float = None, kappa_star: float = None, 
                 theta_star: float = None, mass_function: str = None, m_solar: float = None, m_lower: float = None, m_upper: float = None,
                 rectangular: bool = None, approx: bool = None, safety_scale: float = None,
                 num_stars: int = None, starfile: str = None, num_phi: int = None, num_branches: int = None, random_seed: int = None,
                 write_stars: bool = False, write_critical_curves: bool = False, write_caustics: bool = False, write_length_scales: bool = False,
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
        :param rectangular: whether the star field is rectangular (True) or circular (False)
        :param approx: whether terms for alpha_smooth should be approximated (True) or exact (False)
        :param safety_scale: ratio of the size of the star field to the size of the shooting rectangle
        :param num_stars: number of stars desired
        :param starfile: the location of a binary file containing values for num_stars, rectangular, corner, theta_star, and the star positions and masses.
                         A whitespace delimited text file where each line contains the x1 and x2 coordinates and the mass of a microlens, in units where 
                         theta_star = 1, is also accepted. If provided, this takes precedence for all star information
        :param num_phi: number of steps used to vary phi in the range [0, 2*pi]
        :param num_branches: number of branches to use for phi in the range [0, 2*pi]
        :param random_seed: random seed for star field generation. A value of 0 is reserved for star input files
        :param write_stars: whether to write stars or not
        :param write_critical_curves: whether to write critical curves or not
        :param write_caustics: whether to write caustics or not
        :param write_length_scales: whether to write magnification length scales or not
        :param outfile_prefix: prefix to be used in output file names
        :param verbose: verbosity level of messages. must be 0, 1, 2, or 3
        '''
        self.lib = lib_ccf.lib

        self.obj = self.lib.CCF_init()
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
            self.num_stars = num_stars
            self.random_seed = random_seed

        self.approx = approx
        self.safety_scale = safety_scale
        
        self.starfile = starfile
        
        self.num_phi = num_phi
        if self.num_phi % 2 != 0:
            raise ValueError("num_phi must be an even number")
        self.num_branches = num_branches
        if self.num_phi % (2 * self.num_branches) != 0:
            raise ValueError("num_phi must be a multiple of 2 * num_branches")
            
        self.write_stars = write_stars
        self.write_critical_curves = write_critical_curves
        self.write_caustics = write_caustics
        self.write_mu_length_scales = write_length_scales
        
        self.outfile_prefix = outfile_prefix

    def __del__(self):
        self.lib.CCF_delete(self.obj)
        
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
    def num_stars(self):
        return self.lib.get_num_stars(self.obj)
    
    @num_stars.setter
    def num_stars(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("num_stars must be >= 1")
            self.lib.set_num_stars(self.obj, value)

    @property
    def starfile(self):
        return self.lib.get_starfile(self.obj).decode('utf-8')
    
    @starfile.setter
    def starfile(self, value):
        if value is not None:
            self.lib.set_starfile(self.obj, value.encode('utf-8'))

    @property
    def num_phi(self):
        return self.lib.get_num_phi(self.obj)
    
    @num_phi.setter
    def num_phi(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("num_phi must be >= 1")
            self.lib.set_num_phi(self.obj, value)

    @property
    def num_branches(self):
        return self.lib.get_num_branches(self.obj)
    
    @num_branches.setter
    def num_branches(self, value):
        if value is not None:
            if value < 1:
                raise ValueError("num_branches must be >= 1")
            self.lib.set_num_branches(self.obj, value)
    
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
    def write_critical_curves(self):
        return bool(self.lib.get_write_critical_curves(self.obj))
    
    @write_critical_curves.setter
    def write_critical_curves(self, value):
        if value is not None:
            self.lib.set_write_critical_curves(self.obj, int(value))

    @property
    def write_caustics(self):
        return bool(self.lib.get_write_caustics(self.obj))
    
    @write_caustics.setter
    def write_caustics(self, value):
        if value is not None:
            self.lib.set_write_caustics(self.obj, int(value))

    @property
    def write_mu_length_scales(self):
        return bool(self.lib.get_write_mu_length_scales(self.obj))
    
    @write_mu_length_scales.setter
    def write_mu_length_scales(self, value):
        if value is not None:
            self.lib.set_write_mu_length_scales(self.obj, int(value))

    @property
    def outfile_prefix(self):
        return self.lib.get_outfile_prefix(self.obj).decode('utf-8')
    
    @outfile_prefix.setter
    def outfile_prefix(self, value):
        if value is not None:
            self.lib.set_outfile_prefix(self.obj, value.encode('utf-8'))

    @property
    def corner(self):
        return (self.lib.get_corner_x1(self.obj), self.lib.get_corner_x2(self.obj))
    
    @property
    def num_roots(self):
        return self.lib.get_num_roots(self.obj)

    def run(self):
        if not self.lib.run(self.obj, self.verbose):
            raise Exception("Error running CCF")
        
        self.critical_curves = np.ctypeslib.as_array(self.lib.get_critical_curves(self.obj), 
                                                     shape=(self.num_roots * self.num_branches,
                                                            self.num_phi // self.num_branches + 1,
                                                            2)).copy()
        
        self.caustics = np.ctypeslib.as_array(self.lib.get_caustics(self.obj), 
                                              shape=(self.num_roots * self.num_branches,
                                                     self.num_phi // self.num_branches + 1,
                                                     2)).copy()
            
        if self.write_mu_length_scales:
            self.mu_length_scales = np.ctypeslib.as_array(self.lib.get_mu_length_scales(self.obj), 
                                                          shape=(self.num_roots * self.num_branches,
                                                                 self.num_phi // self.num_branches + 1)).copy()
        else:
            self.mu_length_scales = None

        self.stars = Stars(np.ctypeslib.as_array(self.lib.get_stars(self.obj),
                                                 shape=(self.num_stars, 3)).copy(),
                           self.rectangular, self.corner, self.theta_star)
        
    @property
    def t_ccs(self):
        return self.lib.get_t_ccs(self.obj)
    
    def save(self):
        if not self.lib.save(self.obj, self.verbose):
            raise Exception("Error saving CCF")

    @property
    def mu_length_scales_weights(self):
        if self.mu_length_scales is None:
            raise ValueError("mu_length_scales is None")
        
        # distances between subsequent caustic points
        dists = np.linalg.norm(self.caustics[:, 1:] - self.caustics[:, :-1], axis=2)
        dists = (np.insert(dists, 0, 0, axis=1) + np.insert(dists, dists.shape[1], 0, axis=1)) / 2

        return dists

    def plot_critical_curves(self, ax: Axes, color='black', cmap='viridis', plot_phase=False, **kwargs):

        if plot_phase:
            norm = Normalize(0, 2 * np.pi)

            lc = LineCollection(np.concatenate([self.critical_curves[:,:-1],
                                                self.critical_curves[:,1:]],
                                               axis=2).reshape(-1,2,2),
                                array = np.repeat([np.arange(0, self.num_phi) * 2 * np.pi / self.num_phi],
                                                  self.num_roots, axis=0).ravel(),
                                cmap=cmap, norm=norm)
            line = ax.add_collection(lc)
            cbar = ax.get_figure().colorbar(line, label = '$\\phi$')
            cbar.set_ticks([0, np.pi, 2 * np.pi])
            cbar.set_ticklabels([0, '$\\pi$', '$2\\pi$'])
        else:
            ax.plot(self.critical_curves[:,:,0].T, self.critical_curves[:,:,1].T, color=color, **kwargs)

        if self.theta_star == 1:
            ax.set_xlabel('$x_1 / \\theta_★$')
            ax.set_ylabel('$x_2 / \\theta_★$')
        else:
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')

    def plot_caustics(self, ax: Axes, color='black', cmap='viridis', plot_phase=False, **kwargs):

        if plot_phase:
            norm = Normalize(0, 2 * np.pi)

            lc = LineCollection(np.concatenate([self.caustics[:,:-1],
                                                self.caustics[:,1:]],
                                               axis=2).reshape(-1,2,2),
                                array = np.repeat([np.arange(0, self.num_phi) * 2 * np.pi / self.num_phi],
                                                  self.num_roots, axis=0).ravel(),
                                cmap=cmap, norm=norm)
            line = ax.add_collection(lc)
            cbar = ax.get_figure().colorbar(line, label = '$\\phi$')
            cbar.set_ticks([0, np.pi, 2 * np.pi])
            cbar.set_ticklabels([0, '$\\pi$', '$2\\pi$'])
        else:
            ax.plot(self.caustics[:,:,0].T, self.caustics[:,:,1].T, color=color, **kwargs)

        if self.theta_star == 1:
            ax.set_xlabel('$y_1 / \\theta_★$')
            ax.set_ylabel('$y_2 / \\theta_★$')
        else:
            ax.set_xlabel('$y_1$')
            ax.set_ylabel('$y_2$')

    def plot_mu_length_scales_hist(self, ax: Axes, bins=None, dw=0.01, where=..., **kwargs):
        dat = np.log10(self.mu_length_scales[where])
        weights = self.mu_length_scales_weights[where]

        if bins is None:
            vmin, vmax = (np.min(dat), np.max(dat))
            bins = np.arange(vmin - dw, vmax + dw, dw)

        ax.hist(dat.ravel(), weights=weights.ravel(), 
                density=True, bins=bins, **kwargs)
        
        if self.theta_star == 1:
            ax.set_xlabel("$\\log (d_0 / \\theta_★)$")
            ax.set_ylabel("$p(\\log (d_0 / \\theta_★))$")
        else:
            ax.set_xlabel("$\\log d_0$")
            ax.set_ylabel("$p(\\log d_0)$")
