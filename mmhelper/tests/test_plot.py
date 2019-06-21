"""test_plot.py

Test plot submodule

J. Metz <metz.jp@gmail.com>
"""

import unittest
from mmhelper import plot
import numpy as np


class TestViewStack(unittest.TestCase):
    """
    Class for testing view_stack
    """
    def setUp(self):
        self.t0_ = 100
        self.w0_ = 100
        self.h0_ = 100
        self.input_none = None
        self.input_empty = np.array(None)
        self.input_zeros = np.zeros((self.t0_, self.w0_, self.h0_))
        plot.plt.switch_backend("agg")

    def tearDown(self):
        # Close any opened figures (that might have remained from a failed
        # call)
        plot.plt.close("all")

    def test_none_input(self):
        """
        Tests response to none input
        """
        self.assertRaises(TypeError, plot.view_stack, self.input_none)

    def test_empty_input(self):
        """
        Tests response to empty input
        """
        self.assertRaises(IndexError, plot.view_stack, self.input_empty)

    def test_zeros_input(self):
        """
        Tests response to input of zeros
        """
        plot.view_stack(self.input_zeros)
        figs = plot.plt.get_fignums()
        self.assertEqual(len(figs), 1)
        fig = plot.plt.gcf()
        # Make sure the current figure has 1 axes
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        # Make sure the current figure's axes has 1 image
        ax0 = axes[0]
        ims = ax0.get_images()
        self.assertEqual(len(ims), 1)
