import json
import os
import numpy as np
from functools import reduce


def static_size(x):
    inp_shape = x.shape.as_list()
    if None in inp_shape:
        return 0
    else:
        return reduce(lambda x,y: x*y, inp_shape)

def cache_json(filename, func, makedir=False, allow_load=True):
    '''Caches the result of a function in a file.

    Args:
        func -- Function with zero arguments.
    '''
    if os.path.exists(filename) and allow_load:
        with open(filename, 'r') as r:
            result = json.load(r)
    else:
        if makedir:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        result = func()
        with open(filename, 'w') as w:
            json.dump(result, w)
    return result
