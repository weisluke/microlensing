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
        return np.mean(self.stars[:, 2])

    @property
    def mean_mass2_actual(self):
        return np.mean(self.stars[:, 2]**2)

    @property
    def mean_mass2_ln_mass_actual(self):
        return np.mean(self.stars[:, 2]**2 * np.log(self.stars[:, 2]))

    def plot(self, ax, **kwargs):
        if 's' not in kwargs.keys() and 'sizes' not in kwargs.keys():
            kwargs['s'] = 10 * self.stars[:, 2]
        else:
            kwargs['s'] = kwargs['s'] * self.stars[:, 2]
        if 'c' not in kwargs.keys() and 'color' not in kwargs.keys():
            kwargs['c'] = 'black'
        if 'marker' not in kwargs.keys():
            kwargs['marker'] = '*'

        ax.scatter(self.stars[:, 0], self.stars[:, 1], **kwargs)

        if self.theta_star == 1:
            ax.set_xlabel('$x_1 / \\theta_★$')
            ax.set_ylabel('$x_2 / \\theta_★$')
        else:
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
