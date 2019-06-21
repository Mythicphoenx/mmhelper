# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:28:10 2016

@author: as624
"""
import unittest
from mmhelper.tracking import frametracker_keypoints as frametrck
from mmhelper.tracking import welltracking as welltrck
from mmhelper.dataio import load_sample_full_frame
import numpy as np
from skimage import transform as tf
import scipy.ndimage as ndi


class TestFrameTracking(unittest.TestCase):
    """
    Class for testing frame tracking
    """
    def setUp(self):
        self.generate_frames()

    def generate_frames(self, tform=(12, 15), seed=1, snr=10):
        """
        Generates the frames for the tests
        """
        affinetform = tf.AffineTransform(translation=tform)
        self.lbl1 = load_sample_full_frame(
            seed=seed,
            signal_to_noise=snr)[0]
        self.lbl2 = tf.warp(self.lbl1, affinetform)
        self.transf = np.array(tform)

    def test_frametracking(self):
        """
        Tests the frame tracking function
        """
        tfrm = frametrck(self.lbl1, self.lbl2, nk=50, fn=9,
                         ft=0.001, hk=0.01, min_samples=10)
        xtrans = abs(tfrm[0] - self.transf[1])
        ytrans = abs(tfrm[1] - self.transf[0])
        self.assertLessEqual(xtrans, 1.5)
        self.assertLessEqual(ytrans, 1.5)


class TestWellTracking(unittest.TestCase):
    """
    Class for testing well tracking
    """
    def setUp(self):
        self.lbl1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                             ])
        self.lbl2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                             ])
        self.lbl2tracked = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                     [0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                     [0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                     [0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                    ])

        self.transform = np.array([0, 3])

        self.map_of_wells = {1:2, 2:3, 3:4}

        self.shift = 1

    def test_welltracking(self):
        """
        Tests the well tracking function
        """
        newframe2, well_map = welltrck(
            self.lbl1, self.lbl2, self.transform)
        np.testing.assert_array_equal(self.lbl2tracked, newframe2)
        self.assertDictEqual(self.map_of_wells, well_map)


class TestCombinedTracking(unittest.TestCase):
    """
    Class for testing the combined tracking
    """
    def setUp(self):
        self.generate_frames()
        self.map_of_wells = {1:2, 2:3, 3:4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
                             9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
                             15: 16, 16: 17}

    def generate_frames(self, tform=(12, 15), seed=1, snr=10):
        """
        Generates the frames for testing
        """
        affinetform = tf.AffineTransform(translation=tform)
        self.frame1, self.label1 = load_sample_full_frame(
            seed=seed,
            signal_to_noise=snr)[:2]
        self.label1 = ndi.label(self.label1 > 0)[0]
        self.frame2 = tf.warp(self.frame1, affinetform)
        self.label2 = tf.warp(self.label1.astype(float), affinetform)
        self.transf = np.array(tform)

    def test_full_tracking(self):
        """
        Test frame tracking followed by well tracking
        """
        label2in = ndi.label(self.label2 > 0)[0]

        measured_tfrm = frametrck(
            self.frame1,
            self.frame2,
            nk=50,
            fn=9,
            ft=0.001,
            hk=0.01,
            min_samples=10)
        newlbl2, well_map = welltrck(
            self.label1, label2in, measured_tfrm)
        np.testing.assert_array_equal(self.label2, newlbl2)
        self.assertDictEqual(self.map_of_wells, well_map)
        return

if __name__ == '__main__':
    unittest.main()
