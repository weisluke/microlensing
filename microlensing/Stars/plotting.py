import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


class Star(PathPatch):
    def __init__(self, center, area, **kwargs):
        r = np.sqrt(area * (3 - np.tan(np.pi / 10)**2) / (10 * np.tan(np.pi / 10)))
        coords = r * np.exp(1j * (np.pi / 2 + 2 * 2 * np.pi / 5 * np.array([0,1,2,3,4])))
        coords += center[0] + 1j * center[1]
        coords = np.array([coords.real, coords.imag]).T
        path = Path(coords)
        PathPatch.__init__(self, path, **kwargs)

class Stars(PatchCollection):
    def __init__(self, centers, masses, minmass=1, maxmass=1, **kwargs):
        if maxmass != minmass:
            areas = (masses - minmass) / (maxmass - minmass)
        else:
            areas = masses
        PatchCollection.__init__(self, [Star(center, area) for center, area in zip(centers, areas)], **kwargs)

