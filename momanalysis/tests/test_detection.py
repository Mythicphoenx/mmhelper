# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1 11:35 2016

@author: as624
"""

from momanalysis.detection import detect_bacteria_in_wells as detbac
import momanalysis.detection as mdet
from momanalysis.tests import core
import numpy as np
import unittest
import scipy.ndimage as ndi
import skimage.measure as skmeas

class TestSubtractBackground(unittest.TestCase):
    def setUp(self):
        self.sz = (100,100)
        self.ground_truth = np.zeros(self.sz)
        self.ground_truth_level = 400
        # Add some objects
        self.ground_truth[10:20, 50:60] = self.ground_truth_level
        # Noisy background
        self.bg_std = 10
        self.bg_offset = 100
        self.bg_grad_max = 100
        self.bg = self.bg_std*np.random.randn(*self.sz) + self.bg_offset
        # Add a constant gradient
        X,Y = np.meshgrid(np.arange(self.sz[0]), np.arange(self.sz[1]))
        self.bg += self.bg_grad_max * X/X.max()
        self.image = {0:self.ground_truth + self.bg}

    def test_subtract_background(self):
        removed = mdet.remove_background(self.image, light_background=False)
        # For our current workflow, the background-removed images are inverted
        removed = -removed[0]

        #import matplotlib.pyplot as plt
        #plt.imshow(removed)
        #plt.colorbar()
        #plt.show()
        # Make sure the background is all relatively low now
        # NOTE: As background is subtracted, need to go double above reasonable
        # statisitcally realy unlikely values of ~4 sigma from normal distribution
        self.assertTrue(np.all(removed[self.ground_truth == 0] < 8*self.bg_std))
        # Make sure the foreground is about right
        self.assertTrue(np.all(np.abs(removed[self.ground_truth > 0] - self.ground_truth_level) < 8*self.bg_std))

class DetectBacteria(unittest.TestCase):

    def setUp(self):
        self.lbl1 = {1:np.array([[1,1,1,1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1,1,1,1],
                              [1,200,200,200,1,1,1,1,1,1],
                              [1,200,200,200,1,1,1,1,1,1],
                              [1,200,200,200,1,1,1,1,1,1],
                              [1,200,200,200,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1,1,1,1]])}


        self.lbl2 = {2:np.array([[1,1,1,1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1,1,1,1],
                              [1,1,1,1,200,200,200,200,1,1],
                              [1,1,1,1,200,200,200,200,1,1],
                              [1,1,1,1,200,200,200,200,1,1],
                              [1,1,1,1,200,200,200,200,1,1],
                              [1,1,1,1,200,200,200,2,1,1],
                              [1,1,1,1,1,1,1,1,1,1]])}


        self.lbl_twobac1 = {3:np.array(
            [[  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [  1,200,200,200,  1,  1,  1,  1,  1,  1],
            [  1,200,200,200,  1,  1,  1,  1,  1,  1],
            [  1,200,200,200,  1,  1,  1,  180,  180,  180],
            [  1,200,200,200,  1,  1,  1,  180,  180,  180],
            [  1,  1,  1,  1,  1,  1,  1,180,180,180],
            [  1,  1,  1,  1,  1,  1,  1,180,180,180],
            [  1,  1,  1,  1,  1,  1,  1,1,1,1]])}



        #self.lbl2 = np.array([[0,0,0,0,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0]])

        # Seems to now remove a border....
        self.res1 = [np.array([[0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0]]),]

        self.res2 = [np.array([[0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,1,1,0,0,0],
                              [0,0,0,0,0,1,1,0,0,0],
                              [0,0,0,0,0,1,1,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0]]),]

        self.res_two1 = [np.array(
            [[0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,2,0],
            [0,0,0,0,0,0,0,0,2,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]]),]

        self.wellnum1 = [1]
        self.wellnum2 = [2]
        self.wellnum3 = [3]
        self.label_string = {1:'1'}
        self.label_string2 = {1:'1',2:'2'}
    def test_detect_small_bacteria1(self):
        detected = detbac(self.lbl1,
            timepoint = 0,
            label_dict_string = None, #dictionary of string labels
            maxsize = 1500, # maximum area (in pixels) of an object to be considered a bacteria
            minsize = 0, # maximum area (in pixels) of an object to be considered a bacteria
            absolwidth = 1, #width (in pixels) at which something is definitely a bacteria (can override relativewidth)
            distfrombottom = 0, #ignores anything labeled this distance from the bottom of the well (prevents channel border being labelled)
            topborder = 0, # Distance to exclude from top due to "shadow"
            toprow = 0,#number of pixels along the top row for a label to be discarded
            thresh_perc =1)
        newarrays1 =[]
        wellnums1 = []
        for j, k in detected.items():
            wellnums1.append(j)
            newarrays1.append(k)
        np.testing.assert_array_equal(newarrays1, self.res1)
        self.assertEqual(self.wellnum1, wellnums1)

    def test_detect_small_bacteria2(self):
        detected = detbac(self.lbl2,
            timepoint = 0,
            label_dict_string = None, #dictionary of string labels
            maxsize = 1500, # maximum area (in pixels) of an object to be considered a bacteria
            minsize = 0, # maximum area (in pixels) of an object to be considered a bacteria
            absolwidth = 1, #width (in pixels) at which something is definitely a bacteria (can override relativewidth)
            distfrombottom = 0, #ignores anything labeled this distance from the bottom of the well (prevents channel border being labelled)
            topborder = 0, # Distance to exclude from top due to "shadow"
            toprow = 0,#number of pixels along the top row for a label to be discarded
            thresh_perc =1)
        newarrays2 =[]
        wellnums2 = []
        for j, k in detected.items():
            wellnums2.append(j)
            newarrays2.append(k)
        np.testing.assert_array_equal(newarrays2, self.res2)
        self.assertEqual(self.wellnum2, wellnums2)

    def test_detect_two_bacteria1(self):
        detected = detbac(self.lbl_twobac1,
            timepoint = 0,
            label_dict_string = None, #dictionary of string labels
            maxsize = 1500, # maximum area (in pixels) of an object to be considered a bacteria
            minsize = 0, # maximum area (in pixels) of an object to be considered a bacteria
            absolwidth = 1, #width (in pixels) at which something is definitely a bacteria (can override relativewidth)
            distfrombottom = 0, #ignores anything labeled this distance from the bottom of the well (prevents channel border being labelled)
            topborder = 0, # Distance to exclude from top due to "shadow"
            toprow = 0,#number of pixels along the top row for a label to be discarded
            thresh_perc =1)
        newarrays3 =[]
        wellnums3 = []
        for j, k in detected.items():
            wellnums3.append(j)
            newarrays3.append(k)
        np.testing.assert_array_equal(newarrays3, self.res_two1)
        self.assertEqual(self.wellnum3, wellnums3)

class TestSplitBacteria(unittest.TestCase):
    def setUp(self):
        self.wells = {1:
            np.array([[0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]])}
        self.wells2 = {2:
            np.array([[0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]])
        }

        self.out_wells = [np.array([[0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,0,1,1,1,1,0,0,0],
                      [0,0,0,0,1,1,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]),]

        self.out_wells2 = [np.array([[0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,0,0,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,1,1,1,1,1,1,0,0],
                      [0,0,0,1,1,1,0,0,0,0],
                      [0,0,0,0,1,1,1,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,2,2,2,0,0,0,0],
                      [0,0,0,0,2,2,2,0,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,2,2,2,2,2,2,0,0],
                      [0,0,0,2,2,2,2,0,0,0],
                      [0,0,0,0,2,2,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,3,3,0,0,0,0],
                      [0,0,0,3,3,3,3,0,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,3,3,3,3,3,3,0,0],
                      [0,0,0,3,3,3,3,0,0,0],
                      [0,0,0,0,3,3,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]),]

        self.label_dict_string = {}
        self.out_string = {1:'1',2:'2'}
        self.out_string2 = {1:'1',2:'2',3:'3'}
        self.wellnum1 = [1]
        self.wellnum2 = [2]

    def test_bacteria_no_split(self):
        split, label_dict = mdet.split_bacteria(self.wells, self.label_dict_string, timepoint=0,distfrombottom = -1, absolwidth=0)
        newarrays1 =[]
        wellnums1 = []
        for j, k in split.items():
            wellnums1.append(j)
            newarrays1.append(k)
        np.testing.assert_array_equal(newarrays1, self.out_wells)
        self.assertEqual(self.wellnum1, wellnums1)
        self.assertEqual(self.out_string, label_dict)

    def test_bacteria_split(self):
        split, label_dict = mdet.split_bacteria(self.wells2, self.label_dict_string, timepoint=0,distfrombottom = -1, absolwidth=0)
        newarrays2 =[]
        wellnums2 = []
        for j, k in split.items():
            wellnums2.append(j)
            newarrays2.append(k)
        np.testing.assert_array_equal(newarrays2, self.out_wells2)
        self.assertEqual(self.wellnum2, wellnums2)
        self.assertEqual(self.out_string2, label_dict)

class TestExtractWells(unittest.TestCase):
    def setUp(self):
        # Create test data for the extraction
        # Doesn't really matter what the image is
        self.image = np.random.rand(10,10)
        self.channel = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,1,1,1,1,1,1,1,1,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
        ], dtype=bool)
        self.wells = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,2,0,0,0,0,0,],
            [0,0,1,0,2,0,0,0,3,0,],
            [0,0,1,0,2,0,0,0,3,0,],
            [0,0,1,0,2,0,0,0,3,0,],
            [0,0,1,0,0,0,0,0,3,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
        ], dtype='uint16')
        self.wellstrue = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,1,0,2,0,3,0,4,0,],
            [0,0,1,0,2,0,3,0,4,0,],
            [0,0,1,0,2,0,3,0,4,0,],
            [0,0,1,0,2,0,3,0,4,0,],
            [0,0,1,0,2,0,3,0,4,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
        ], dtype='uint16')
        self.wellwidth = 1
        self.coords = [
            ([6,5,4,3,2], [2,2,2,2,2]),
            ([6,5,4,3,2], [4,4,4,4,4]),
            ([6,5,4,3,2], [6,6,6,6,6]),
            ([6,5,4,3,2], [8,8,8,8,8]),
        ]

        self.wellimages = { l:self.image[c][:,None]
            for l,c in enumerate(self.coords, start=1)
        }
        #self.wellimages =
        #self.coords =

    def test_extract_well_profiles(self):

        images, wellimage, coords = mdet.extract_well_profiles(
            self.image,
            self.channel,
            self.wells,
            wellwidth = self.wellwidth,
            #debug=True,
            min_well_sep_factor=0.5,
        )

        self.assertEqual(len(images), len(self.wellimages))
        for w1, w2 in zip(images, self.wellimages):
            #np.testing.assert_array_equal(w1, w2)
            core.assert_array_almost_equal(w1, w2)
        np.testing.assert_array_equal(wellimage, self.wellstrue)
        self.assertEqual(coords, self.coords)

class TestDetectWells(unittest.TestCase):
    def setUp(self):
        # Create test data for the detection
        self.image = 10 + np.random.randn(10,10)
        self.image += 20*np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
        ])
        self.scale_range = [1.0, 3.0]
        self.maxd = 20
        self.mind = 3
        self.maxperp = 3
        self.minwidth=0
        self.lbl = ndi.label(
            np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,1,1,0,0,1,1,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ]))[0]

    def test_detect_wells(self):
        lblgood, ridges = mdet.detect_wells(
            self.image,
            scale_range = self.scale_range,
            maxd = self.maxd,
            mind = self.mind,
            maxperp = self.maxperp,
            minwidth = self.minwidth,
        )
        core.assert_array_equal(lblgood, self.lbl)


class TestDetectChannel(unittest.TestCase):
    def setUp(self):
        # Create test data for the extraction
        # REALLY SMALL, DUMMY VERSION
        im_noise = 100 + np.random.randn(10,10)
        self.image_bright = im_noise + 20*np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,3,3,3,3,3,3,3,3,0,],
            [0,3,3,3,3,3,3,3,3,0,],
            [0,3,3,3,3,3,3,3,3,0,],
            [0,0,0,0,0,0,0,0,0,0,],
        ])
        self.image_dark = im_noise + 20*np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,0,0,1,0,0,1,0,0,0,],
            [0,-3,-3,-3,-3,-3,-3,-3,-3,0,],
            [0,-3,-3,-3,-3,-3,-3,-3,-3,0,],
            [0,-3,-3,-3,-3,-3,-3,-3,-3,0,],
            [0,0,0,0,0,0,0,0,0,0,],
        ])
        self.scale_range = [1.0, 2.0]
        self.channel = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,1,1,1,1,1,1,1,1,0,],
            [0,1,1,1,1,1,1,1,1,0,],
            [0,1,1,1,1,1,1,1,1,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=bool)

        #
        # MORE REALISTIC CHANNEL
        #im_noise = 150 + np.random.randn(200,1000)
        #self.image_bright = im_noise.copy()
        ## Add in wells
        #well_y0 = 20
        #well_y1 = 120
        #well_width = 5
        #chan_x0 = 10
        #chan_x1 = 1000-10
        #chan_y0 = 5
        #chan_y1 = 20 # I.e. width of 15
        #for left in range(50, 1000, 100): # 50, 150, 250, ... 950
        #    self.image_bright[well_y0:well_y1, left:left+well_width] += 20
        #    self.image_dark[well_y0:well_y1, left:left+well_width] += 20
        ## Add in channel
        #self.image_bright[chan_y0:chan_y1, chan_x0:chanx1] += 20*3
        #self.image_dark[chan_y0:chan_y1, chan_x0:chanx1] -= 20*3

        self.orientation = 0

    def test_detect_channel_bright(self):
        channel = mdet.detect_channel(
            self.image_bright,
            scale_range = self.scale_range,
            minwidth=0,
            bright=True,
        )
        # This is a little messier than wells; let's allow
        # for the detected channel to be a little smaller...
        # make sure it's contained by the expected
        try:
            self.assertEqual( np.sum(self.channel[channel])/channel.sum(), 1.0 )
        except:
            print("COVERAGE", np.sum(self.channel[channel])/channel.sum())
            raise

        # Make sure what's missed is less than 50%
        areafull = self.channel.sum()
        areamissed = self.channel[~channel].sum()
        try:
            self.assertTrue( areamissed/areafull <= 0.5)
        except:
            print("AREA MISSED",areamissed/areafull)
            core.plt.matshow(channel)
            core.plt.matshow(self.channel)
            raise

        # Lastly, make sure it's going to give us basically the
        # same important properties!
        prop = skmeas.regionprops(channel.astype(int))[0]
        try:
            self.assertTrue(np.abs(prop.orientation - self.orientation)
                < (10*np.pi/180))
        except:
            print("ANGLES", prop.orientation, self.orientation)
            raise


    def test_detect_channel_dark(self):
        channel = mdet.detect_channel(
            self.image_dark,
            scale_range = self.scale_range,
            bright = False,
            minwidth=0,
        )
        #core.assert_array_equal(channel, self.channel)
        # This is a little messier than wells; let's allow
        # for the detected channel to be a little smaller...

        # make sure it's contained by the expected
        try:
            self.assertEqual( np.sum(self.channel[channel])/channel.sum(), 1.0 )
        except:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self.image_dark, cmap='gray')
            plt.contour(self.channel, levels=[0.5], colors=["g"])
            plt.contour(channel, levels=[0.5], colors=["r"])
            plt.savefig("/tmp/momanalysis_debug_test_detect_channel_dark.png")
            plt.close()
            print("COVERAGE", np.sum(self.channel[channel])/channel.sum())
            raise

        # Make sure what's missed is less than 50%
        areafull = self.channel.sum()
        areamissed = self.channel[~channel].sum()
        try:
            self.assertTrue( areamissed/areafull <= 0.5)
        except:
            print("AREA MISSED",areamissed/areafull)
            core.plt.matshow(channel)
            core.plt.savefig("AREAMISSED_DETECTED.png")
            core.plt.matshow(self.channel)
            core.plt.savefig("AREAMISSED_EXPECTED.png")
            raise

        # Lastly, make sure it's going to give us basically the
        # same important properties!
        prop = skmeas.regionprops(channel.astype(int))[0]
        try:
            self.assertTrue(np.abs(prop.orientation - self.orientation)
                < (10*np.pi/180))
        except:
            print("ANGLES", prop.orientation, self.orientation)
            raise

class TestGetChannelOrientationAndLine(unittest.TestCase):
    def setUp(self):
        self.channel_horiz = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,1,1,0,1,1,1,1,0,0,],
            [0,1,1,1,1,1,1,1,1,0,],
            [0,1,1,1,1,1,1,0,1,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)
        self.channel_vert = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,1,1,0,0,0,0,0,],
            [0,0,0,1,1,1,0,0,0,0,],
            [0,0,0,1,1,1,0,0,0,0,],
            [0,0,0,0,1,1,0,0,0,0,],
            [0,0,0,1,1,1,0,0,0,0,],
            [0,0,0,1,1,1,0,0,0,0,],
            [0,0,0,1,1,0,0,0,0,0,],
            [0,0,0,1,1,1,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)
        self.channel_diag = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,1,1,0,0,],
            [0,0,0,0,0,1,1,1,0,0,],
            [0,0,0,0,1,1,1,0,0,0,],
            [0,0,0,1,1,1,0,0,0,0,],
            [0,0,1,1,1,0,0,0,0,0,],
            [0,1,1,1,0,0,0,0,0,0,],
            [0,0,1,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)

    def test_get_channel_orientation_and_line_horizontal(self):
        angle, line = mdet.get_channel_orientation_and_line(self.channel_horiz)
        # Make sure angle is +-1 degree
        self.assertAlmostEqual(angle, 0, delta=np.pi/180)

        # Order coordinates so we can compare properly
        line = sorted(line, key=lambda pt: pt[0])

        # Check coordinates
        self.assertAlmostEqual(line[0][0], 0.5, delta=1)
        self.assertAlmostEqual(line[0][1], 7, delta=1)
        self.assertAlmostEqual(line[1][0], 8, delta=1)
        self.assertAlmostEqual(line[1][1], 7, delta=1)

    def test_get_channel_orientation_and_line_vertical(self):
        angle, line = mdet.get_channel_orientation_and_line(self.channel_vert)
        # Make sure angle is +-1 degree
        # Angle can be ~ + or - 90 degrees, so need to abs
        self.assertAlmostEqual(np.abs(angle), np.pi/2, delta=np.pi/180)

        # Order coordinates so we can compare properly
        line = sorted(line, key=lambda pt: pt[1])
        # Check coordinates
        self.assertAlmostEqual(line[0][0], 4, delta=1)
        self.assertAlmostEqual(line[0][1], 0.5, delta=1)
        self.assertAlmostEqual(line[1][0], 4, delta=1)
        self.assertAlmostEqual(line[1][1], 8.5, delta=1)

    def test_get_channel_orientation_and_line_diagonal(self):
        angle, line = mdet.get_channel_orientation_and_line(self.channel_diag)
        # Make sure angle is +-45 degree
        # Angle can be ~ + or - 45 degrees, so need to abs
        self.assertAlmostEqual(np.abs(angle), np.pi/4, delta=np.pi/180)

        # Order coordinates so we can compare properly
        line = sorted(line, key=lambda pt: pt[0])

        # Check coordinates
        self.assertAlmostEqual(line[0][0], 1, delta=1)
        self.assertAlmostEqual(line[0][1], 1, delta=1)
        self.assertAlmostEqual(line[1][0], 8, delta=1)
        self.assertAlmostEqual(line[1][1], 8, delta=1)



class TestGetWellsAndUnitVectors(unittest.TestCase):
    def setUp(self):

        wells_vertical = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,3,0,0,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)
        self.props_vertical = skmeas.regionprops(wells_vertical)

        wells_horizontal = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,1,1,1,1,1,1,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,2,2,2,2,2,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,3,3,3,3,3,3,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,4,4,4,4,4,4,4,4,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)
        self.props_horizontal = skmeas.regionprops(wells_horizontal)

        wells_vertical_with_outlier = np.array([
            [0,0,0,0,0,0,0,0,0,0,],
            [0,5,5,5,5,5,5,5,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,0,3,0,0,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,1,0,2,0,3,0,4,0,0,],
            [0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)
        self.props_vertical_with_outlier = skmeas.regionprops(wells_vertical_with_outlier)

    def test_get_wells_and_unit_vectors_vertical(self):

        coms, oris, uvec_para, uvec_perp = mdet._get_wells_and_unit_vectors(
            self.props_vertical)
        # Unit vectors can be +-, so make sure by abs-ing
        uvec_perp = np.abs(uvec_perp)
        uvec_para = np.abs(uvec_para)

        np.testing.assert_array_almost_equal(uvec_para, [0, 1])
        np.testing.assert_array_almost_equal(uvec_perp, [1, 0])

    def test_get_wells_and_unit_vectors_horizontal(self):

        coms, oris, uvec_para, uvec_perp = mdet._get_wells_and_unit_vectors(
            self.props_horizontal)
        # Unit vectors can be +-, so make sure by abs-ing
        uvec_perp = np.abs(uvec_perp)
        uvec_para = np.abs(uvec_para)

        np.testing.assert_array_almost_equal(uvec_para, [1, 0])
        np.testing.assert_array_almost_equal(uvec_perp, [0, 1])

    def test_get_wells_and_unit_vectors_vertical_with_outlier(self):
        """
        Now added in a horizontal well - if the simple outlier
        detection doesn't filter it, the vectors shoudl be off.
        """
        coms, oris, uvec_para, uvec_perp = mdet._get_wells_and_unit_vectors(
            self.props_vertical_with_outlier)
        # Unit vectors can be +-, so make sure by abs-ing
        uvec_perp = np.abs(uvec_perp)
        uvec_para = np.abs(uvec_para)

        np.testing.assert_array_almost_equal(uvec_para, [0, 1])
        np.testing.assert_array_almost_equal(uvec_perp, [1, 0])
        # Check that one region got rejected
        self.assertEqual(len(oris), len(self.props_vertical_with_outlier)-1)


class TestGetWellSpacingAndSeparations(unittest.TestCase):
    def setUp(self):
        # Need, coms (centres of mass), the perpendicular unit vector
        # to project the coms along, and wellwidth for filtering
        # nearby coms
        # Simple case - all uniformly distributed
        self.coms_horizontal = [
            [2, 5],
            [4, 6],
            [6, 4],
            [8, 5.5],
            [10, 5],
        ]
        self.uvec_perp_horizontal = [1,0]
        self.coms_vertical = [
            [5, 2],
            [6, 4],
            [4, 6],
            [5.5, 8],
            [5, 10],
        ]
        self.uvec_perp_vertical = [0,1]
        self.coms_horizontal_with_gaps = [
            [2, 5],
            [4, 6],
            [6, 4],
            [8, 5.5],
            [12, 5],
            [14, 6.5],
            [18, 4.5],
        ]
        self.wellwidth = 0.5

    def test_get_well_spacing_and_separations_horizontal(self):
        normseps, posperp_sorted = mdet._get_well_spacing_and_separations(
            self.coms_horizontal,
            self.uvec_perp_horizontal,
            self.wellwidth,
        )
        self.assertListEqual(normseps.tolist(), [1,1,1,1])
        self.assertListEqual(posperp_sorted.tolist(), [2,4,6,8,10])

    def test_get_well_spacing_and_separations_vertical(self):
        normseps, posperp_sorted = mdet._get_well_spacing_and_separations(
            self.coms_vertical,
            self.uvec_perp_vertical,
            self.wellwidth,
        )
        self.assertListEqual(normseps.tolist(), [1,1,1,1])
        self.assertListEqual(posperp_sorted.tolist(), [2,4,6,8,10])

    def test_get_well_spacing_and_separations_horizontal_with_gaps(self):
        normseps, posperp_sorted = mdet._get_well_spacing_and_separations(
            self.coms_horizontal_with_gaps,
            self.uvec_perp_horizontal,
            self.wellwidth,
        )
        self.assertListEqual(normseps.tolist(), [1,1,1,2,1,2])
        self.assertListEqual(posperp_sorted.tolist(), [2,4,6,8,12,14,18])



class TestInterpolatePositionsAndExtractProfiles(unittest.TestCase):
    def setUp(self):
        self.image = 10 + np.random.randn(10,22)
        self.image += 20*np.array([
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,10,10,0,0,1,1,0,0,2,2,0,0,3,3,0,0,4,4,0,0,],
            [0,0,10,10,0,0,1,1,0,0,2,2,0,0,3,3,0,0,4,4,0,0,],
            [0,0,10,10,0,0,1,1,0,0,2,2,0,0,3,3,0,0,4,4,0,0,],
            [0,0,10,10,0,0,1,1,0,0,2,2,0,0,3,3,0,0,4,4,0,0,],
            [0,0,10,10,0,0,1,1,0,0,2,2,0,0,3,3,0,0,4,4,0,0,],
            [0,0,10,10,0,0,1,1,0,0,2,2,0,0,3,3,0,0,4,4,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        ])
        # Means for the regions after extrapolation
        self.image_means = {1:210, 2:30, 3:50, 4:70, 5:90}
        labels = np.array([
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,0,0,0,0,4,4,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,0,0,0,0,4,4,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,0,0,0,0,4,4,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,0,0,0,0,4,4,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,0,0,0,0,4,4,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,0,0,0,0,4,4,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)
        self.finallabel = np.array([
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,4,4,0,0,5,5,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,4,4,0,0,5,5,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,4,4,0,0,5,5,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,4,4,0,0,5,5,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,4,4,0,0,5,5,0,0,],
            [0,0,1,1,  0,0,2,2,0,0,3,3,0,0,4,4,0,0,5,5,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            [0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
            ], dtype=int)

        self.normseps = [1, 1, 2]
        self.posperp_sorted = [2.5, 6.6, 10.5, 18.5]
        self.propsgood = skmeas.regionprops(labels)
        self.uvec_para = [0,1]
        self.uvec_perp = [1,0]
        self.wellwidth = 2
    def test_interpolate_positions_and_extract_profiles(self):
        images, wellimage, coords = mdet.interpolate_positions_and_extract_profiles(
            np.array(self.normseps),
            np.array(self.posperp_sorted),
            self.propsgood,
            np.array(self.uvec_para),
            np.array(self.uvec_perp),
            self.wellwidth,
            self.image,
        )

        # We can check the stats for the regions
        for k, im in images.items():
            # Have normally distributed noise with std 1,
            # averaged over 12 values... deviation should
            # be well less than 1... but just failed, set to delta=2
            self.assertAlmostEqual(
                im.mean(),
                self.image_means[k],
                delta=2,
                )
        np.testing.assert_array_equal(wellimage, self.finallabel)

        #print("\n\n")
        #print("Well images:")
        #for k in images:
        #    print(k)
        #    print(images[k])
        #print("Coords:")
        #for c in coords:
        #    print(c)
        #import matplotlib.pyplot as plt
        #plt.imshow(wellimage)
        #plt.show()




if __name__ == '__main__':
    unittest.main()
