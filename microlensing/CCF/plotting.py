import numpy as np
import shapely
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


colors = {-1: '#ff7700',  # saddlepoints are orange
           0: '#ff7700',  # saddlepoints are orange if log_area makes eigvals 0
           1: '#0077ff'}  # minima and maxima are blue

# function to determine the order in which to join critical curves
def list_order(x):
    starts = np.round(x[:,0], 5) # round the first point on the curve to 5 decimal places
    starts = starts[:,0] + 1j * starts[:,1] # convert to complex numbers

    ends = np.round(x[:,-1], 5) # round final points to 5 decimal places
    ends = ends[:,0] + 1j * ends[:,1] # and convert to complex numbers

    indices = range(0, len(starts)) # the size of the critical curve array (num_roots, basically)
    # in order, the index of the curve that starts where curve[i] ends
    where = [np.argwhere(starts == ends[i])[0,0] for i in indices]

    res = {}
    for a, b in zip(indices, where):
        res[a] = b # dictionary mapping end -> start
    return res

class CriticalCurves(PatchCollection):

    def __init__(self, critical_curves, **kwargs):

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
                
        # sort polygons 
        areas = np.array([shapely.Polygon(x).area for x in polygons])
        order = np.argsort(np.abs(areas))
        polygons = [polygons[x] for x in order]
        areas = areas[order]

        PatchCollection.__init__(self, [Polygon(p, facecolor=colors[np.sign(a)], edgecolor='black', **kwargs)
                                        for p, a in zip(polygons, areas)], 
                                 match_original=True)
