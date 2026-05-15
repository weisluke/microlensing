import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize


colors = {-1: '#ff7700',  # saddlepoints are orange
           0: '#ff7700',  # saddlepoints are orange if log_area makes eigvals 0
           1: '#0077ff'}  # minima and maxima are blue

# function to determine the order in which to join critical curves
def list_order(x):
    x = np.asanyarray(x)

    starts = np.round(x[:,0], 5) # round the first point on the curve to 5 decimal places
    starts = starts[:,0] + 1j * starts[:,1] # convert to complex numbers

    ends = np.round(x[:,-1], 5) # round final points to 5 decimal places
    ends = ends[:,0] + 1j * ends[:,1] # and convert to complex numbers

    # ensure there are no duplicated elements
    assert np.unique(starts).size == starts.size == np.unique(ends).size == ends.size

    # argsort gives indices. since we are sorting lists of the same values,
    # just in different orders, we can zip the indices required to put
    # ends and starts in the same order
    res = dict(zip(np.argsort(ends), np.argsort(starts)))
    return res

# calculate the area of a polygon defined by a list of points
# the shoelace formula adds x_i * y_i+1 - x_i+1 * y_i to the area
# the area is + for counterclockwise, - for clockwise
def shoelace(x, y):
    return (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2

class CriticalCurves(PatchCollection):

    def __init__(self, critical_curves, xrange=None, yrange=None, **kwargs):

        polygons = []
        order = list_order(critical_curves)

        while len(order.keys()) > 0:
            polygons.append(np.empty(shape=(0,2)))
            key = next(iter(order.keys())) # get the next unused key
            while key in order.keys(): # and while it is in the list of keys
                # append the chain whose start is the end of the previous one
                polygons[-1] = np.append(polygons[-1], critical_curves[key], axis=0)
                # and now use that chain's end as the next key
                key = order.pop(key)

        if xrange is not None:
            polygons = [p for p in polygons 
                        if np.max(p[:,0]) > xrange[0] and
                           np.min(p[:,0]) < xrange[1]]

        if yrange is not None:
            polygons = [p for p in polygons 
                        if np.max(p[:,1]) > yrange[0] and
                           np.min(p[:,1]) < yrange[1]]
                
        # sort polygons 
        # critical curves are traced out such that clockwise encloses positive area,
        # opposite of what the shoelace formula gives
        areas = np.array([-shoelace(*p.T) for p in polygons])
        # argsort sorts from greatest to least, need to reverse for PatchCollection
        order = np.argsort(np.abs(areas))[::-1]
        polygons = [polygons[x] for x in order]
        areas = areas[order]

        PatchCollection.__init__(self, [Polygon(p, facecolor=colors[np.sign(a)], edgecolor='black', **kwargs)
                                        for p, a in zip(polygons, areas)], 
                                 match_original=True)

class PhaseCurves(LineCollection):

    def __init__(self, critical_curves, num_roots, num_phi, num_branches,
                 xrange=None, yrange=None, cmap='viridis', **kwargs):
        # critical curve shape = (num_roots * num_branches, num_phi / num_branches + 1, 2)

        # starting points of the branches
        phi = np.linspace(0, 2 * np.pi, num_branches, endpoint=False)[:,None]
        # add on the range covered by each branch
        phi = phi + np.linspace(0, 2 * np.pi / num_branches, num_phi // num_branches + 1)
        # and extend over entirety of critical curves
        phi = np.repeat([phi.ravel()], num_roots, axis=0).ravel()

        midpoints = (critical_curves[:,:-1] + critical_curves[:, 1:]) / 2
        start = np.insert(midpoints, 0, critical_curves[:,0], axis=1)
        end = np.insert(midpoints, midpoints.shape[1], critical_curves[:,-1], axis=1)

        lines = np.concatenate([start, end], axis=2).reshape(-1,2,2)

        if xrange is not None:
            where = np.argwhere((np.max(lines[:,:,0], axis=1) > xrange[0]) 
                                & (np.min(lines[:,:,0], axis=1) < xrange[1]))[:,0]
            lines = lines[where]
            phi = phi[where]

        if yrange is not None:
            where = np.argwhere((np.max(lines[:,:,1], axis=1) > yrange[0]) 
                                & (np.min(lines[:,:,1], axis=1) < yrange[1]))[:,0]
            lines = lines[where]
            phi = phi[where]

        norm = Normalize(0, 2 * np.pi)
        LineCollection.__init__(self, lines, array = phi, cmap=cmap, norm=norm, **kwargs)
