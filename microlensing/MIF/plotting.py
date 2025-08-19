import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse
from matplotlib.collections import PatchCollection
import shapely
import shapely.plotting


class Star(PathPatch):
    def __init__(self, center, area, **kwargs):
        r = np.sqrt(area * (3 - np.tan(np.pi / 10)**2) / (10 * np.tan(np.pi / 10)))
        coords = r * np.exp(1j * (np.pi / 2 + 2 * 2 * np.pi / 5 * np.array([0,1,2,3,4])))
        coords += center[0] + 1j * center[1]
        coords = np.array([coords.real, coords.imag]).T
        path = Path(coords)
        PathPatch.__init__(self, path, **kwargs)

class Stars(PatchCollection):
    def __init__(self, centers, masses, minmass = 1, maxmass=1, **kwargs):
        if maxmass != minmass:
            areas = (masses - minmass) / (maxmass - minmass)
        else:
            areas = masses
        PatchCollection.__init__(self, [Star(center, area) for center, area in zip(centers, areas)], **kwargs)

def get_intersections(polygons1, polygons2):
    inter = [shapely.intersection(polygon, polygons2) for polygon in polygons1]
    inter = [what for row in inter for what in row]
    return inter

class Images(PatchCollection):

    def __init__(self, positions, invmags, r=1, is_ellipse=True, log_area=False, mu_min=10**-3, **kwargs):
        
        colors = {-1: '#ff7700',  # saddlepoints are orange
                   0: '#ff7700',  # saddlepoints are orange if log_area makes eigvals 0
                   1: '#0077ff'}  # minima and maxima are blue

        mags = np.linalg.inv(invmags)
        eigvals, eigvecs = np.linalg.eig(mags)

        if not is_ellipse:
            new_eigvals = np.sqrt(np.abs(eigvals[:,0] * eigvals[:,1]))
            eigvals[:,0] = np.sign(eigvals[:,0]) * new_eigvals
            eigvals[:,1] = np.sign(eigvals[:,1]) * new_eigvals
        
        if log_area:
            # scale areas logarithmically
            new_eigvals_0 = np.sqrt(np.max([[0] * len(eigvals),
                                            np.log10(np.abs(eigvals[:,0] * eigvals[:,1]) / mu_min) / np.abs(np.log10(mu_min))], 
                                           axis = 0)
                                    / np.pi * np.abs(eigvals[:,0] / eigvals[:,1]))
            new_eigvals_1 = np.sqrt(np.max([[0] * len(eigvals), 
                                            np.log10(np.abs(eigvals[:,0] * eigvals[:,1]) / mu_min) / np.abs(np.log10(mu_min))],
                                           axis = 0)
                                    / np.pi * np.abs(eigvals[:,1] / eigvals[:,0]))
            eigvals[:, 0] = np.sign(eigvals[:,0]) * new_eigvals_0
            eigvals[:, 1] = np.sign(eigvals[:,1]) * new_eigvals_1
                                        

        ellipses = [Ellipse((x[0], x[1]), r * e[0], r * e[1],
                            angle = np.arctan2(v[1, 0], v[0, 0]) * 180 / np.pi,
                            facecolor = colors[np.sign(e[0] * e[1])],
                            edgecolor = colors[np.sign(e[0] * e[1])])
                    for x, e, v in zip(positions, eigvals, eigvecs)]
        

        where_minima = (np.sign(np.prod(eigvals,axis=1)) > 0)
        where_saddles = (np.sign(np.prod(eigvals,axis=1)) < 0)
        
        verts = [what.get_verts() for what in ellipses]

        polys = np.array([shapely.geometry.Polygon(vert) for vert in verts])
        inter = get_intersections(polys[where_minima], polys[where_saddles])
        inter = [shapely.plotting.patch_from_polygon(what, color = 'black') for what in inter]

        PatchCollection.__init__(self, [*ellipses, *inter], match_original=True)
