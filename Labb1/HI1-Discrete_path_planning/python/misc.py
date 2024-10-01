import scipy.io as sio
import time
import math


def loadmat(filename):
    '''
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


class Timer:
    t0 = 0.0
    dt = 0.0

    def tic(self):
        self.t0 = time.perf_counter()
        # self.t0 = time.time()

    def toc(self):
        self.dt = time.perf_counter() - self.t0
        # self.dt = time.time() - self.t0
        return self.dt


def latlong_distance(p1, p2):
    """Compute Haversine distance between points.

    LatLongDistance(p1, p2) returns distance in meters
    between points p1 and p2.

    A point p is a list/array p=[longitude, latitude]
    """
    radius = 6371  # km

    lat1 = p1[1] * math.pi / 180.0
    lat2 = p2[1] * math.pi / 180.0
    lon1 = p1[0] * math.pi / 180.0
    lon2 = p2[0] * math.pi / 180.0

    deltaLat = lat2 - lat1
    deltaLon = lon2 - lon1
    a = (math.sin(deltaLat / 2)**2
         + math.cos(lat1) * math.cos(lat2) * math.sin(deltaLon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    d = d * 1e3  # Return in m
    return d
