import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import shapely
import shapely.plotting


colors = {-1: '#ff7700',  # saddlepoints are orange
           0: '#ff7700',  # saddlepoints are orange if log_area makes eigvals 0
           1: '#0077ff'}  # minima and maxima are blue

def get_intersections(polygons1, polygons2):
    inter = [shapely.intersection(polygon, polygons2) for polygon in polygons1]
    inter = [what for row in inter for what in row]
    return inter

class Images(PatchCollection):

    def __init__(self, positions, invmags, s=1, is_ellipse=True, log_area=False, mu_min=10**-3, **kwargs):

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
                                        
        # multiply radii by s to scale area
        ellipses = [Ellipse((x[0], x[1]), s * e[0], s * e[1],
                            angle = np.degrees(np.arctan2(v[1, 0], v[0, 0])),
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
