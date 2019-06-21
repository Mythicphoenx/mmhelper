# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1 11:35 2016

@author: as624
"""

import unittest
from mmhelper.detection.bacteria import detect_bacteria_in_all_wells as detbac
from mmhelper.comparison import match_labels, determine_precision_recall
import mmhelper.detection.bacteria as mdet
import mmhelper.detection.wells as mdet_wells
import numpy as np
import skimage.measure as skmeas


class TestSubtractBackground(unittest.TestCase):
    """
    class for testing background subtraction
    """
    def setUp(self):
        self.sz0 = (100, 100)
        self.ground_truth = np.zeros(self.sz0)
        self.ground_truth_level = 400
        # Add some objects
        self.ground_truth[10:20, 50:60] = self.ground_truth_level
        # Noisy background
        self.bg_std = 10
        self.bg_offset = 100
        self.bg_grad_max = 100
        self.bkg = self.bg_std * np.random.randn(*self.sz0) + self.bg_offset
        # Add a constant gradient
        x0_ = np.meshgrid(np.arange(self.sz0[0]), np.arange(self.sz0[1]))[0]
        self.bkg += self.bg_grad_max * x0_ / x0_.max()
        self.image = {0: self.ground_truth + self.bkg}

    def test_subtract_background(self):
        """
        Tests background subtraction
        """
        removed = mdet_wells.remove_background(self.image, light_background=False)
        # For our current workflow, the background-removed images are inverted
        removed = -removed[0]

        #import matplotlib.pyplot as plt
        # plt.imshow(removed)
        # plt.colorbar()
        # plt.show()
        # Make sure the background is all relatively low now
        # NOTE: As background is subtracted, need to go double above reasonable
        # statisitcally realy unlikely values of ~4 sigma from normal
        # distribution
        self.assertTrue(
            np.all(removed[self.ground_truth == 0] < 8 * self.bg_std))
        # Make sure the foreground is about right
        self.assertTrue(
            np.all(
                np.abs(
                    removed[self.ground_truth > 0] - self.ground_truth_level) < 8
                * self.bg_std))


class DetectBacteria(unittest.TestCase):
    """
    Unittests for detecting bacteria
    """

    def setUp(self):
        self.lbl1 = {1: np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 211, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 211, 211, 211, 1, 1, 1, 1, 1, 1],
                                  [1, 211, 211, 211, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 211, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        self.lbl2 = {2: np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 211, 211, 1, 1, 1],
                                  [1, 1, 1, 1, 211, 211, 211, 211, 1, 1],
                                  [1, 1, 1, 1, 211, 211, 211, 211, 1, 1],
                                  [1, 1, 1, 1, 211, 211, 211, 211, 1, 1],
                                  [1, 1, 1, 1, 1, 211, 211, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        self.lbl_twobac1 = {3: np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 185, 185, 1, 1, 1, 1, 1],
             [1, 1, 185, 185, 185, 185, 1, 1, 1, 1],
             [1, 1, 185, 185, 185, 185, 1, 1, 1, 1],
             [1, 1, 185, 185, 185, 185, 1, 1, 1, 1],
             [1, 1, 1, 185, 185, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 215, 215, 1, 1, 1, 1, 1],
             [1, 1, 215, 215, 215, 215, 1, 1, 1, 1],
             [1, 1, 215, 215, 215, 215, 1, 1, 1, 1],
             [1, 1, 215, 215, 215, 215, 1, 1, 1, 1],
             [1, 1, 1, 215, 215, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )}

        # self.lbl2 = np.array([[0,0,0,0,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,1,1,1,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0],
        #                      [0,0,0,0,0,0,0,0,0,0]])

        # Seems to now remove a border....
        self.res1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.res2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.res_two1 = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
             [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        self.wellnum1 = [1]
        self.wellnum2 = [2]
        self.wellnum3 = [3]
        self.label_string = {1: '1'}
        self.label_string2 = {1: '1', 2: '2'}

    def test_detect_small_bacteria1(self):
        """
        Test the detection of small bacteria
        """
        detected = detbac(self.lbl1,
                          # maximum area (in pixels) of an object to be
                          # considered a bacteria
                          maxsize=1500,
                          # maximum area (in pixels) of an object to be
                          # considered a bacteria
                          minsize=0,
                          # width (in pixels) at which something is definitely
                          # a bacteria (can override relativewidth)
                          absolwidth=0.1,
                          # ignores anything labeled this distance from the
                          # bottom of the well (prevents channel border being
                          # labelled)
                         )

        for k in detected.values():
            man_ids, det_ids, man_id_dict, det_id_dict = match_labels(
                self.res1, k)
            precision_scores, recall_scores = determine_precision_recall(
                self.res1, k, man_ids, det_ids, man_id_dict, det_id_dict)
            assert np.all(np.array(precision_scores) >= 0.75)
            assert np.all(np.array(recall_scores) >= 0.75)

    def test_detect_small_bacteria2(self):
        """
        A test for detecting small bacteria
        """
        detected = detbac(self.lbl2,
                          # maximum area (in pixels) of an object to be
                          # considered a bacteria
                          maxsize=1500,
                          # maximum area (in pixels) of an object to be
                          # considered a bacteria
                          minsize=0,
                          # width (in pixels) at which something is definitely
                          # a bacteria (can override relativewidth)
                          absolwidth=0.1,
                         )
        for k in detected.values():
            man_ids, det_ids, man_id_dict, det_id_dict = match_labels(
                self.res2, k)
            precision_scores, recall_scores = determine_precision_recall(
                self.res2, k, man_ids, det_ids, man_id_dict, det_id_dict)
            assert np.all(np.array(precision_scores) >= 0.75)
            assert np.all(np.array(recall_scores) >= 0.75)

    def test_detect_two_bacteria1(self):
        """
        Test two detect two bacteria
        """
        detected = detbac(self.lbl_twobac1,
                          # maximum area (in pixels) of an object to be
                          # considered a bacteria
                          maxsize=1500,
                          # maximum area (in pixels) of an object to be
                          # considered a bacteria
                          minsize=0,
                          # width (in pixels) at which something is definitely
                          # a bacteria (can override relativewidth)
                          absolwidth=0.1,
                         )
        for k in detected.values():
            man_ids, det_ids, man_id_dict, det_id_dict = match_labels(
                self.res_two1, k)
            precision_scores, recall_scores = determine_precision_recall(
                self.res_two1, k, man_ids, det_ids, man_id_dict, det_id_dict)
            assert np.all(np.array(precision_scores) >= 0.75)
            assert np.all(np.array(recall_scores) >= 0.75)


class TestSplitBacteria(unittest.TestCase):
    """
    Class for testing the splitting of bacteria
    """
    def setUp(self):
        self.wells = {1:
                      np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
        self.wells2 = {2:
                       np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                 [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                 [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                 [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                      }

        self.wells_int = {1:
                      np.array([[0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                [0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                [0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
        self.wells2_int = {2:
                       np.array([[0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 400, 400, 400, 400, 400, 400, 0, 0],
                                 [0, 0, 0, 400, 400, 400, 400, 0, 0, 0],
                                 [0, 0, 0, 0, 400, 400, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                      }

        self.out_wells = [np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                    [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                    [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                    [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), ]

        self.out_wells2 = [np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                     [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                     [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                                     [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                                     [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
                                     [0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
                                     [0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
                                     [0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                                 ]

        self.out_string = {1: '1', 2: '2'}
        self.out_string2 = {1: '1', 2: '2', 3: '3'}
        self.wellnum1 = [1]
        self.wellnum2 = [2]

    def test_bacteria_no_split(self):
        """
        Test bacteria that don't need splitting
        """
        split = mdet.split_bacteria_in_all_wells(
            self.wells,
            self.wells_int,
            min_skel_length=3,
        )
        newarrays1 = []
        wellnums1 = []
        for j, k in split.items():
            wellnums1.append(j)
            newarrays1.append(k)
        np.testing.assert_array_equal(newarrays1, self.out_wells)
        self.assertEqual(self.wellnum1, wellnums1)

    def test_bacteria_split(self):
        """
        Test bacteria that need splitting
        """
        split = mdet.split_bacteria_in_all_wells(
            self.wells2,
            self.wells2_int,
            min_skel_length=4,
        )
        newarrays2 = []
        wellnums2 = []
        for j, k in split.items():
            wellnums2.append(j)
            newarrays2.append(k)
        np.testing.assert_array_equal(newarrays2, self.out_wells2)
        self.assertEqual(self.wellnum2, wellnums2)


class TestExtractWells(unittest.TestCase):
    """
    Class for testing the extraction of well profiles
    """
    def setUp(self):
        # Create test data for the extraction
        # Doesn't really matter what the image is
        self.image = np.random.rand(10, 10)
        self.channel = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=bool)
        self.wells = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, ],
            [0, 0, 1, 0, 2, 0, 0, 0, 3, 0, ],
            [0, 0, 1, 0, 2, 0, 0, 0, 3, 0, ],
            [0, 0, 1, 0, 2, 0, 0, 0, 3, 0, ],
            [0, 0, 1, 0, 0, 0, 0, 0, 3, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype='uint16')
        self.wellstrue = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype='uint16')
        self.wellwidth = 1
        self.coords = {
            1: ([6, 5, 4, 3, 2], [0, 0, 0, 0, 0]),
            2: ([6, 5, 4, 3, 2], [2, 2, 2, 2, 2]),
            3: ([6, 5, 4, 3, 2], [4, 4, 4, 4, 4]),
            4: ([6, 5, 4, 3, 2], [6, 6, 6, 6, 6]),
            5: ([6, 5, 4, 3, 2], [8, 8, 8, 8, 8]),
        }

        self.wellimages = {l: self.image[c][:, None]
                           for l, c in self.coords.items()
                          }

    def test_extract_well_profiles(self):
        """
        Tests extract_well_profiles
        """
        images, wellimage, coords = mdet_wells.extract_well_profiles(
            self.image,
            self.wells,
            wellwidth=self.wellwidth,
            min_well_sep_factor=0.5,
        )

        self.assertEqual(len(images), len(self.wellimages))
        for well1, well2 in zip(images, self.wellimages):
            np.testing.assert_array_equal(well1, well2)
            #core.assert_array_almost_equal(w1, w2)
        np.testing.assert_array_equal(wellimage, self.wellstrue)
        self.assertEqual(coords, self.coords)


class TestDetectWells(unittest.TestCase):
    """
    Class for testing well detection
    """
    def setUp(self):
        # Create test data for the detection
        self.image = 10 + np.random.randn(30, 30)
        self.lbl = np.zeros((30, 30), dtype=int)
        self.lbl[6: 9, 6:24] = 1
        self.lbl[16: 19, 6:24] = 2

        self.image[5:10, 5:25] -= 20
        self.image[6: 9, 6:24] += 40
        self.image[15:20, 5:25] -= 20
        self.image[16: 19, 6:24] += 40

        self.scale_range = [1.0, 3.0]
        self.maxd = 40
        self.mind = 3
        self.maxperp = 10
        self.minwidth = 0
        self.min_outline_area = 0

    def test_detect_initial_well_masks(self):
        """
        Tests the detect_initial_well_masks function
        """
        lblgood = mdet_wells.detect_initial_well_masks(
            self.image,
            scale_range=self.scale_range,
            maxd=self.maxd,
            mind=self.mind,
            maxperp=self.maxperp,
            min_outline_area=self.min_outline_area,
            merge_length=0,
            debug="",
        )[0]
        man_ids, det_ids, man_id_dict, det_id_dict = match_labels(
            self.lbl, lblgood)
        precision_scores, recall_scores = determine_precision_recall(
            self.lbl, lblgood, man_ids, det_ids, man_id_dict, det_id_dict)
        try:
            assert np.all(np.array(precision_scores) > 0.9)
            assert np.all(np.array(recall_scores) > 0.9)
            #core.assert_array_equal(lblgood, self.lbl)
        except BaseException:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self.image, cmap='gray')
            plt.title("Input image")
            plt.savefig("test_detect_initial_well_masks_fail_input_image.jpg")
            plt.figure()
            plt.imshow(lblgood)
            plt.title("Got labels")
            plt.savefig("test_detect_initial_well_masks_fail_detected_labels.jpg")
            plt.figure()
            plt.imshow(self.lbl)
            plt.title("Expected labels")
            plt.savefig("test_detect_initial_well_masks_fail_expected_labels.jpg")
            plt.close("all")
            raise

class TestGetWellsAndUnitVectors(unittest.TestCase):
    """
    Class for testing get_wells_and_unit_vectors
    """
    def setUp(self):

        wells_vertical = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=int)
        self.props_vertical = skmeas.regionprops(wells_vertical)

        wells_horizontal = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 2, 2, 2, 2, 2, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 4, 4, 4, 4, 4, 4, 4, 4, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=int)
        self.props_horizontal = skmeas.regionprops(wells_horizontal)

        wells_vertical_with_outlier = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 5, 5, 5, 5, 5, 5, 5, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=int)
        self.props_vertical_with_outlier = skmeas.regionprops(
            wells_vertical_with_outlier)

    def test_get_wells_vertical(self):
        """
        Test for get_wells_and_unit_vectors on vertical wells
        """

        coms, oris, uvec_para, uvec_perp = mdet_wells.get_wells_and_unit_vectors(
            self.props_vertical)
        # Unit vectors can be +-, so make sure by abs-ing
        uvec_perp = np.abs(uvec_perp)
        uvec_para = np.abs(uvec_para)

        np.testing.assert_array_almost_equal(uvec_para, [0, 1])
        np.testing.assert_array_almost_equal(uvec_perp, [1, 0])

    def test_get_wells_horizontal(self):
        """
        Test for get_wells_and_unit_vectors on horizontal wells
        """

        coms, oris, uvec_para, uvec_perp = mdet_wells.get_wells_and_unit_vectors(
            self.props_horizontal)
        # Unit vectors can be +-, so make sure by abs-ing
        uvec_perp = np.abs(uvec_perp)
        uvec_para = np.abs(uvec_para)

        np.testing.assert_array_almost_equal(uvec_para, [1, 0])
        np.testing.assert_array_almost_equal(uvec_perp, [0, 1])

    def test_get_wells_vertical_outlier(self):
        """
        Now added in a horizontal well - if the simple outlier
        detection doesn't filter it, the vectors should be off.
        """
        coms, oris, uvec_para, uvec_perp = mdet_wells.get_wells_and_unit_vectors(
            self.props_vertical_with_outlier)
        # Unit vectors can be +-, so make sure by abs-ing
        uvec_perp = np.abs(uvec_perp)
        uvec_para = np.abs(uvec_para)

        np.testing.assert_array_almost_equal(uvec_para, [0, 1])
        np.testing.assert_array_almost_equal(uvec_perp, [1, 0])
        # Check that one region got rejected
        self.assertEqual(len(oris), len(self.props_vertical_with_outlier) - 1)


class TestGetWellSpacingAndSeparations(unittest.TestCase):
    """
    Class for testing well_spacing_and_seps
    """
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
        self.uvec_perp_horizontal = [1, 0]
        self.coms_vertical = [
            [5, 2],
            [6, 4],
            [4, 6],
            [5.5, 8],
            [5, 10],
        ]
        self.uvec_perp_vertical = [0, 1]
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

    def test_well_spacing_horizontal(self):
        """
        Test for the function well_spacing_and_seps on horizontal wells
        """
        normseps, posperp_sorted = mdet_wells.well_spacing_and_seps(
            self.coms_horizontal,
            self.uvec_perp_horizontal,
            self.wellwidth,
        )
        self.assertListEqual(normseps.tolist(), [1, 1, 1, 1])
        self.assertListEqual(posperp_sorted.tolist(), [2, 4, 6, 8, 10])

    def test_well_spacing_vertical(self):
        """
        Test for the function well_spacing_and_seps on vertical wells
        """
        normseps, posperp_sorted = mdet_wells.well_spacing_and_seps(
            self.coms_vertical,
            self.uvec_perp_vertical,
            self.wellwidth,
        )
        self.assertListEqual(normseps.tolist(), [1, 1, 1, 1])
        self.assertListEqual(posperp_sorted.tolist(), [2, 4, 6, 8, 10])

    def test_well_spacing_horiz_gaps(self):
        """
        Test for the function well_spacing_and_seps on horizontal wells with gaps
        """
        normseps, posperp_sorted = mdet_wells.well_spacing_and_seps(
            self.coms_horizontal_with_gaps,
            self.uvec_perp_horizontal,
            self.wellwidth,
        )
        self.assertListEqual(normseps.tolist(), [1, 1, 1, 2, 1, 2])
        self.assertListEqual(posperp_sorted.tolist(), [2, 4, 6, 8, 12, 14, 18])


class TestInterpolatePositionsAndExtractProfiles(unittest.TestCase):
    """
    Class for testing well interpolation and extracting well profiles
    """
    def setUp(self):
        self.image = 10 + np.random.randn(10, 22)
        self.image += 20 * np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 10, 10, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 10, 10, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 10, 10, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 10, 10, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 10, 10, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 10, 10, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ])
        # Means for the regions after extrapolation
        self.image_means = {1: 210, 2: 30, 3: 50, 4: 70, 5: 90}
        labels = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=int)
        self.finallabel = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 5, 5, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 5, 5, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 5, 5, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 5, 5, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 5, 5, 0, 0, ],
            [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 5, 5, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=int)

        self.normseps = [1, 1, 2]
        self.posperp_sorted = [2.5, 6.6, 10.5, 18.5]
        self.propsgood = skmeas.regionprops(labels)
        self.uvec_para = [0, 1]
        self.uvec_perp = [1, 0]
        self.wellwidth = 2

    def test_interp_pos_extract_profs(self):
        """
        Test for interpolate_pos_extract_profs
        """
        images, wellimage, coords = mdet_wells.interpolate_pos_extract_profs(
            np.array(
                self.normseps), np.array(
                    self.posperp_sorted), self.propsgood, np.array(
                    self.uvec_para), np.array(
                    self.uvec_perp), self.wellwidth, self.image, )

        # We can check the stats for the regions
        for k, im0 in images.items():
            # Have normally distributed noise with std 1,
            # averaged over 12 values... deviation should
            # be well less than 1... but just failed, set to delta=2
            self.assertAlmostEqual(
                im0.mean(),
                self.image_means[k],
                delta=2,
            )
        np.testing.assert_array_equal(wellimage, self.finallabel)

class TestRelabelBacteria(unittest.TestCase):
    """
    class for testing bacteria relabelling
    """
    def setUp(self):
        self.old_labels = {
            1 : np.array([[1,0,0,2],]),
            2 : np.array([[1,0,0,2],]),
        }
        self.expected = {
            1 : np.array([[1,0,0,2],]),
            2 : np.array([[3,0,0,4],]),
        }

    def test_simple_relabel(self):
        """
        Tests relabelling of bacteria
        """
        result = mdet.relabel_bacteria(self.old_labels)
        self.assertListEqual(list(self.expected.keys()), list(result.keys()))
        for k, v in result.items():
            np.testing.assert_array_equal(v, self.expected[k])

class TestFilterBacteria(unittest.TestCase):
    """
    class for testing bacteria relabelling
    """
    def setUp(self):
        self.old_labels = np.array([
            [0,0,0,0,0,0,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,0,0,0,0,0,0],
            ])

        self.expected = np.array([
            [0,0,0,0,0,0,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,0,0,0,0,0,0],
            ])

        self.expected_nothing = np.array([
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            ])


        self.min_av_width = 1
        self.minsize = 30
        self.maxsize = 50

        self.min_width_too_big = 4
        self.minsize_too_big = 100
        self.maxsize_too_small = 20

    def test_simple_filter_nothing(self):
        """
        Tests filtering, with nothing to remove
        """
        result = mdet.filter_bacteria(
            self.old_labels,
            self.min_av_width,
            self.minsize,
            self.maxsize,
            )[0]
        np.testing.assert_array_equal(result, self.expected)

    def test_filter_area_too_small(self):
        """
        Tests filter when area is too small
        """
        result = mdet.filter_bacteria(
            self.old_labels,
            self.min_av_width,
            self.minsize_too_big,
            self.maxsize,
            )[0]
        np.testing.assert_array_equal(result, self.expected_nothing)

    def test_filter_area_too_big(self):
        """
        Tests filter when area is too big
        """
        result = mdet.filter_bacteria(
            self.old_labels,
            self.min_av_width,
            self.minsize,
            self.maxsize_too_small,
            )[0]
        np.testing.assert_array_equal(result, self.expected_nothing)

    def test_filter_too_narrow(self):
        """
        Tests filter when bacteria is too narrow
        """
        result = mdet.filter_bacteria(
            self.old_labels,
            self.min_width_too_big,
            self.minsize,
            self.maxsize,
            )[0]
        np.testing.assert_array_equal(result, self.expected_nothing)


if __name__ == '__main__':
    unittest.main()
