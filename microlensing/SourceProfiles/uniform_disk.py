try:
    import cupy as np
except ImportError:
    import numpy as np


class UniformDisk():

    def __init__(self, radius: int, max_radius: int = None):
        '''
        :param radius: radius of the disk in pixels
        :param max_radius: maximum radius of the 2D profile.
                           by construction, providing a maximum radius r
                           means the profile always has a shape (2 * r + 1)^2.
                           default value is the radius
        '''
        self.radius = radius

        if max_radius is None:
            max_radius = self.radius

        # use np.int8 to minimize memory, since it is just 0s and 1s
        kernel = np.zeros((2 * max_radius + 1, 2 * max_radius + 1),
                          dtype=np.int8)

        y, x = np.ogrid[-max_radius: max_radius + 1,
                        -max_radius: max_radius + 1]
        mask = x**2 + y**2 <= self.radius**2
        kernel[mask] = 1

        self.profile = kernel
        self.weight = np.sum(kernel)

    @property
    def half_light_radius(self):
        return self.radius / np.sqrt(2)


class UniformDisks():

    def __init__(self, max_radius: int, min_radius: int = 0, step: int = 1):
        '''
        :param max_radius: maximum radius of the 2D profiles in pixels
        :param radius: minimum radius of the 2D profiles in pixels
        :param step: step size for the radius in pixels
        '''
        # use np.int8 to minimize memory, since it is just 0s and 1s
        kernel = np.zeros((max_radius - min_radius + 1,
                           2 * max_radius + 1, 2 * max_radius + 1),
                          dtype=np.int8)

        self._radii, y, x = np.ogrid[min_radius: max_radius + 1: step,
                                     -max_radius: max_radius + 1,
                                     -max_radius: max_radius + 1]
        mask = x**2 + y**2 <= self._radii**2
        kernel[mask] = 1

        self.profiles = kernel
        self.weights = np.sum(kernel, axis=(1,2))
        self._radii = self._radii.flatten()

    @property
    def radii(self):
        try:
            return self._radii.get()
        except AttributeError:
            return self._radii

    @property
    def half_light_radii(self):
        return self.radii / np.sqrt(2)
