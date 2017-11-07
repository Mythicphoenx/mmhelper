# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:28:10 2016

@author: as624
"""

from momanalysis.tracking import frametracker_keypoints as frametrck
from momanalysis.tracking import welltracking as welltrck
import unittest
import numpy as np
from skimage import transform as tf
from momanalysis.io import load_sample_full_frame
import scipy.ndimage as ndi

class TestFrameTracking(unittest.TestCase):

    def setUp(self):
        self.generate_frames()

    def generate_frames(self, tform=(12,15), seed=1, snr=10):
        affinetform = tf.AffineTransform(translation=tform)
        self.lbl1 = load_sample_full_frame(
            seed = seed,
            signal_to_noise=snr)[0]
        self.lbl2 = tf.warp(self.lbl1, affinetform)
        self.transf = np.array(tform)


    def test_frametracking(self):
        tfrm = frametrck(self.lbl1,self.lbl2, nk = 50, fn = 9, ft = 0.001, hk = 0.01, min_samples=10)
        xtrans = abs(tfrm[0]-self.transf[1])
        ytrans = abs(tfrm[1]-self.transf[0])
        self.assertLessEqual(xtrans, 1.5)
        self.assertLessEqual(ytrans, 1.5)


class TestWellTracking(unittest.TestCase):

    def setUp(self):
        self.lbl1 = np.array([[  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              ])
        self.lbl2 = np.array([[  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              [  0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                              ])
        self.lbl2tracked = np.array([[  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [  0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                     [  0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                     [  0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                     [  0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 0, 0, 0],
                                    ])

        self.transform = [0, 3]

        self.shift = 1


    def test_welltracking(self):
        newframe2 = welltrck(self.lbl1, self.lbl2, self.transform, pixls=1)
        np.testing.assert_array_equal(self.lbl2tracked, newframe2)


class TestCombinedTracking(unittest.TestCase):
    def setUp(self):
        self.generate_frames()

    def generate_frames(self, tform=(12,15), seed=1, snr=10):
        affinetform = tf.AffineTransform(translation=tform)
        self.frame1, self.label1 = load_sample_full_frame(
            seed = seed,
            signal_to_noise=snr)[:2]
        self.frame2 = tf.warp(self.frame1, affinetform)
        self.label2 = tf.warp(self.label1.astype(float), affinetform)
        self.transf = np.array(tform)


    def test_full_tracking(self):

        label2in = ndi.label(self.label2>0)[0]

        measured_tfrm = frametrck(self.frame1,self.frame2, nk = 50, fn = 9, ft = 0.001, hk = 0.01, min_samples=10)
        newlbl2 = welltrck(self.label1, label2in, measured_tfrm, pixls=10)

        np.testing.assert_array_equal(self.label2, newlbl2)

        return
        print("DETECTED TRANSFORM:", measured_tfrm)
        print("ACTUAL", self.transf)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.frame1, cmap='gray')
        plt.imshow(self.label1, cmap='jet', alpha=0.3)
        for i in range(1, int(self.label1.max()+1)):
            bw = self.label1 == i
            if not np.any(bw):
                continue
            pt = np.mean(bw.nonzero(), axis=1)
            plt.text(pt[1], pt[0], i, color="w")
        plt.title("Input label 1")

        plt.figure()
        plt.imshow(self.frame2, cmap='gray')
        plt.imshow(label2in, cmap='jet', alpha=0.3)
        for i in range(1, int(label2in.max()+1)):
            bw = label2in == i
            if not np.any(bw):
                continue
            pt = np.mean(bw.nonzero(), axis=1)
            plt.text(pt[1], pt[0], i, color="w")
        plt.title("Label2 INPUT (RANDOM)")

        plt.figure()
        plt.imshow(self.frame2, cmap='gray')
        plt.imshow(self.label2, cmap='jet', alpha=0.3)
        for i in range(1, int(self.label2.max()+1)):
            bw = self.label2 == i
            if not np.any(bw):
                continue

            pt = np.mean(bw.nonzero(), axis=1)
            plt.text(pt[1], pt[0], i, color="w")
        plt.title("Theory label 2")

        plt.figure()
        plt.imshow(self.frame2, cmap='gray')
        plt.imshow(newlbl2, cmap='jet', alpha=0.3)
        for i in range(1, int(newlbl2.max()+1)):
            bw = newlbl2 == i
            if not np.any(bw):
                continue
            pt = np.mean(bw.nonzero(), axis=1)
            plt.text(pt[1], pt[0], i, color="w")
        plt.title("Measured label 2")

        plt.figure()
        plt.title("DIFFERENCE")
        plt.imshow(self.label2 - newlbl2)

        plt.show()

if __name__ == '__main__':
    unittest.main()

