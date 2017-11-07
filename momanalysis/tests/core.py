


import matplotlib.pyplot as plt
import numpy as np
import sys

def assert_array_equal(a, b):
    if not np.array_equal(a,b):
        fname = sys._getframe().f_back.f_code.co_name
        name = fname
        plt.matshow(a)
        plt.title("%s : Array 1" % name)
        plt.matshow(b)
        plt.title("%s : Array 2" % name)
        np.testing.assert_array_equal(a,b)


def assert_array_almost_equal(a, b):
    try:
        np.testing.assert_array_almost_equal(a,b)
    except:
        fname = sys._getframe().f_back.f_code.co_name
        name = fname
        plt.matshow(a)
        plt.title("%s : Array 1" % name)
        plt.matshow(b)
        plt.title("%s : Array 2" % name)
        #np.testing.assert_array_almost_equal(a,b)
        raise
