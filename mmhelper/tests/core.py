"""core.py

Common testing functions

J. Metz <metz.jp@gmail.com>
"""

import sys
import random
import string
import matplotlib.pyplot as plt
import numpy as np



def random_string(length, characters=string.ascii_letters + string.digits):
    """
    Generates a random string
    """
    return "".join(random.choice(characters) for i in range(length))


def assert_array_equal(array1, array2):
    """
    Visual array comparison
    """
    if not np.array_equal(array1, array2):
        fname = sys._getframe().f_back.f_code.co_name
        name = fname
        plt.matshow(array1)
        plt.title("%s : Array 1" % name)
        plt.matshow(array2)
        plt.title("%s : Array 2" % name)
        np.testing.assert_array_equal(array1, array2)


def assert_array_almost_equal(array1, array2):
    """
    Visual array comparison
    """
    try:
        np.testing.assert_array_almost_equal(array1, array2)
    except BaseException:
        fname = sys._getframe().f_back.f_code.co_name
        name = fname
        plt.matshow(array1)
        plt.title("%s : Array 1" % name)
        plt.matshow(array2)
        plt.title("%s : Array 2" % name)
        # np.testing.assert_array_almost_equal(a,b)
        raise
