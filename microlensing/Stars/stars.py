from . import plotting

import numpy as np


class Stars():

    def __init__(self, stars, rectangular: bool, corner, theta_star: float):
        '''
        :param stars: array of star positions and masses (x1, x2, m)
        :param rectangular: whether star field is rectangular or circular
        :param corner: (x1, x2) corner of the star field, assuming it is centered at (0,0)
        :param theta_star: Einstein radius of a unit mass point lens
        '''

        if not isinstance(stars, np.ndarray):
            self.stars = np.array(stars)
        else:
            self.stars = stars

        self.rectangular = rectangular

        self.corner = tuple(corner)

        self.theta_star = theta_star

    @property
    def num_stars(self):
        return self.stars.shape[0]

    @property
    def positions(self):
        return self.stars[:,:2]

    @property
    def masses(self):
        return self.stars[:,2]

    @property
    def mean_mass_actual(self):
        return np.mean(self.masses)

    @property
    def mean_mass2_actual(self):
        return np.mean(self.masses**2)

    @property
    def mean_mass2_ln_mass_actual(self):
        return np.mean(self.masses**2 * np.log(self.masses))

    @property
    def kappa_star(self):
        res = np.pi * self.theta_star**2 * np.sum(self.masses)
        if self.rectangular:
            res /= 4 * self.corner[0] * self.corner[1]
        else:
            res /= np.pi * (self.corner[0]**2 + self.corner[1]**2)
        return res
    
    @property
    def r90(self):
        f = np.sqrt(self.mean_mass2_actual) * np.exp(-self.mean_mass2_ln_mass_actual / self.mean_mass2_actual)
        sigma = self.theta_star * np.sqrt(self.kappa_star * self.mean_mass2_actual / self.mean_mass_actual
                                          * np.log(2 * np.exp(1 - np.euler_gamma) * f * np.sqrt(self.num_stars)))
        return 2.145966026 * sigma
    
    @property
    def r99(self):
        return self.theta_star * np.sqrt(100 * self.kappa_star * self.mean_mass2_actual / self.mean_mass_actual)
    
    @property
    def r999(self):
        return self.theta_star * np.sqrt(1000 * self.kappa_star * self.mean_mass2_actual / self.mean_mass_actual)

    def plot(self, ax, color='black', s=1, **kwargs):

        stars = plotting.Stars(self.positions, s * self.masses, np.min(self.masses), np.max(self.masses), 
                               color=color, **kwargs)
        
        ax.add_collection(stars)

        if self.theta_star == 1:
            ax.set_xlabel('$x_1 / \\theta_★$')
            ax.set_ylabel('$x_2 / \\theta_★$')
        else:
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
