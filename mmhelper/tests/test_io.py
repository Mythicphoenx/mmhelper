"""
Testing for dataio.py in mmhelper module
"""
# FILE        : test_io.py
# CREATED     : 11/11/16 13:04:18
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Unittests for IO functions
#
import traceback
import unittest
import os
import tempfile
import numpy as np
import skimage.io as skio
import mmhelper.dataio as mio



class TestLoadData(unittest.TestCase):
    """
    Class for testing the loading of data
    """

    def setUp(self):
        # Create a temporary file with test input data
        # NB using high numbers to avoid annoying low contrast image
        # warning from skimage.io
        self.data = np.array([
            [100, 200, 3],
            [4, 5, 6],
            [7, 8, 9],
            [100, 200, 255],
        ], dtype='uint8')
        self.filename = tempfile.mkstemp(".tif")[1]
        skio.imsave(self.filename, self.data)

    def test_load_data(self):
        """
        Tests loading data
        """
        np.testing.assert_array_equal(
            mio.load_data(self.filename),
            self.data)

    def tearDown(self):
        try:
            os.remove(self.filename)
        except BaseException:
            print("WARNING: UNABLE TO REMOVE TEMPORARY TESTING FILE:")
            print(self.filename)
            print("DUE TO ERROR:")
            traceback.print_exc()
            print("MAKE SURE TO MANUALLY REMOVE THE FILE YOURSELF")
            print("(OR LET YOUR SYSTEM DEAL WITH IT!)")


class TestSampleData(unittest.TestCase):
    """
    Class for testing the loading of sample data
    """
    def setUp(self):
        self.default_shape = (200, 220)
        self.dtype_wells = np.dtype(np.float64)
        self.dtype_labs = np.dtype(np.uint16)

    def test_load_sample_well_data(self):
        """
        Tests the loading of the sample well data
        """
        # The data itself is random, so let's just make sure
        # we get arrays that look about right
        wells, labs = mio.load_sample_well_data()
        self.assertEqual(wells.shape, self.default_shape)
        self.assertEqual(labs.shape, self.default_shape)
        self.assertIs(wells.dtype, self.dtype_wells)
        self.assertIs(labs.dtype, self.dtype_labs)

    def tearDown(self):
        pass


class TestFluoSplit(unittest.TestCase):
    """
    Class for testing the splitting of fluorescence data
    """
    def setUp(self):
        self.default_shape = np.arange(24).reshape(2, 3, 4)

        self.brightfield_image = np.arange(12).reshape(1, 3, 4)

        self.fluo_image = np.arange(12, 24).reshape(1, 1, 3, 4)

    def test(self):
        """
        Tests the splitting of fluorescence data
        """
        data, fluo_data = mio.split_fluorescence(
            self.default_shape, num_fluo=1)
        np.testing.assert_array_equal(data, self.brightfield_image)
        np.testing.assert_array_equal(fluo_data, self.fluo_image)
