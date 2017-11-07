#
# FILE        : test_io.py
# CREATED     : 11/11/16 13:04:18
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Unittests for IO functions
#

import unittest
import numpy as np
import tempfile
import skimage.io as skio
import os
import momanalysis.io as mio
import traceback

class TestLoadData(unittest.TestCase):

    def setUp(self):
        # Create a temporary file with test input data
        # NB using high numbers to avoid annoying low contrast image
        # warning from skimage.io
        self.data = np.array([
            [100,200,3],
            [4,5,6],
            [7,8,9],
            [100,200,255],
        ], dtype='uint8')
        fid, self.filename = tempfile.mkstemp(".tif")
        skio.imsave(self.filename, self.data)

    def test_load_data(self):
        np.testing.assert_array_equal(
            mio.load_data(self.filename),
            self.data)

    def tearDown(self):
        try:
            os.remove(self.filename)
        except:
            print("WARNING: UNABLE TO REMOVE TEMPORARY TESTING FILE:")
            print(self.filename)
            print("DUE TO ERROR:")
            traceback.print_exc()
            print("MAKE SURE TO MANUALLY REMOVE THE FILE YOURSELF")
            print("(OR LET YOUR SYSTEM DEAL WITH IT!)")


class TestSampleData(unittest.TestCase):

    def setUp(self):
        self.default_shape = (200,220)
        self.dtype_wells = np.dtype(np.float64)
        self.dtype_labs = np.dtype(np.uint16)

    def test_load_sample_well_data(self):
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

    def setUp(self):
        self.default_shape = np.arange(24).reshape(2,3,4) 
        
        self.brightfield_image = np.arange(12).reshape(1,3,4) 
        
        self.fluo_image = np.arange(12,24).reshape(1,3,4) 
        
    def test(self):
        data, fluo_data = mio.split_fluorescence(self.default_shape)
        np.testing.assert_array_equal(data, self.brightfield_image)
        np.testing.assert_array_equal(fluo_data, self.fluo_image)
        
