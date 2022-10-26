from gdsim.models import FirstOrderModel
from gdsim.models import StrejcModel

def parallelGridSearch(args):
    # main program 
    from math import fabs
    from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score   
    from sklearn.metrics.pairwise import paired_euclidean_distances
    import numpy as np
    import sys
    # sys.path.append('/home/wejsciowe/research/meshwork/')

    output, coeffs, task_number = args
    params, scope, system = coeffs
    name, variant = system
    sx, sy = output


    # print(scope)
    px = []
    py = []

    if name == 'Inertial':
        # a1 a2 T1 T2 T3 tau
        #coeffs = (5000, None, 0.00001, None, None, None)
        inert = FirstOrderModel(params, variant)
        # print(params, variant)
        px, py = inert.grid(scope)
    elif name == 'Strejc':
        inert = StrejcModel(params, variant)
        px, py = inert.grid(scope)



    diff = -1
    mse = mean_squared_error(sy, py)
    mae = mean_absolute_error(sy, py)
    medae = median_absolute_error(sy, py)
    r2 = r2_score(sy, py)

    px = py = sx = sy = None
    inert = None

    return (diff, mse, mae, medae, r2, *params), task_number
