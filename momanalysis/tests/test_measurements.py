# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:51:22 2016

@author: as624
"""

import momanalysis.measurements as mmeas
import momanalysis.output as moutp
import numpy as np
import unittest
import csv
import tempfile
import os

class TestFindFilename(unittest.TestCase):

    def setUp(self):
        self.inputname = 'testinputfilename.test'

        self.output = 'testinputfilename'

    def test_input_filename(self):
        self.assertEqual( self.output, mmeas.find_input_filename(self.inputname, test=True)[1])

class TestCounts(unittest.TestCase):

    def setUp(self):
        self.wells = np.array([
            [0,0,0,0,0],
            [0,1,0,2,0],
            [0,1,0,2,0],
            [0,1,0,2,0],
        ])
        self.bacteria = np.array([
            [0,0,0,0,0],
            [0,1,0,0,0],
            [0,0,0,3,0],
            [0,2,0,0,0],
        ])
        self.counts = [2,1]

        # Using just individual labels
        self.bacteria_labels = {1:
            np.array([[1,0,2,2,2,0,3],]),2:
            np.array([[0,0,0],]),3:
            np.array([[1,],])
        }
        self.counts2 = [3, 0, 1]

    def test_counts_just_labels(self):
        self.assertEqual( self.counts2, mmeas.count_bacteria_in_wells(self.bacteria_labels))


class TestBacteriaMeasurements(unittest.TestCase):

    def setUp(self):
        self.bacteria_labels = np.array([[1,1,1,1,0,0],
                      [1,1,1,1,0,0],
                      [2,2,2,2,2,0],
                      [2,2,2,2,2,0]])

        self.areaarray = [['1',8],['2',10]]

        self.length1 = 4
        self.length2 = 5
        self.lbl_dict = {1:'1', 2:'2'}

    def test_bacteria_measurements(self):
        measurements = mmeas.bacteria_measurements(self.lbl_dict,self.bacteria_labels)
        areas = [row[0:2] for row in measurements]
        length1 = measurements[0][2]
        length2 = measurements[1][2]
        self.assertEqual(self.areaarray, areas)
        self.assertTrue((self.length1-1) <= length1 <= (self.length1+1))
        self.assertTrue((self.length2-1) <= length2 <= (self.length2+1))

class TestFluoMeasurements(unittest.TestCase):

    def setUp(self):
        self.bacteria_labels = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,3,3,3],
                                      [1,1,1,0,0,0,0,0,0,3,3,3],
                                      [1,1,1,0,0,0,0,0,0,3,3,3],
                                      [1,1,1,0,0,0,0,0,0,3,3,3],
                                      [1,1,1,0,0,0,0,0,0,3,3,3],
                                      [0,0,0,0,0,0,0,0,0,3,3,3],
                                      [0,0,0,2,2,2,0,0,0,3,3,3],
                                      [0,0,0,2,2,2,0,0,0,3,3,3],
                                      [0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0]])

        self.fluo_image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,3000,3000,3000],
                                      [2000,2000,2000,0,0,0,0,0,0,3000,3000,3000],
                                      [2100,2100,2100,0,0,0,0,0,0,3800,3800,3800],
                                      [2100,2100,2100,0,0,0,0,0,0,3000,3000,3000],
                                      [2200,2200,2200,0,0,0,0,0,0,3000,3000,3000],
                                      [0,0,0,0,0,0,0,0,0,3000,3000,3000],
                                      [0,0,0,3200,3200,3200,0,0,0,3000,3000,3000],
                                      [0,0,0,3300,3300,3300,0,0,0,3000,3000,3000],
                                      [0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0]])

        self.bkground = 100

        self.bkg_sem = 1

        self.fluo_values = {1:[2100.0,2000.0],2:[3250.0,3150.0],3:[3100.0,3000.0]}

    def test_fluorescentmeasurements(self):
        fluo = mmeas.fluorescence_measurements(self.bacteria_labels, self.bkground, self.bkg_sem, self.fluo_image)
        self.assertEqual(self.fluo_values, fluo)

class TestFluorescenceBackground(unittest.TestCase):

    def setUp(self):
        self.wells0 = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0],
                              [1,1,1,0,0,0,0,0,0,3,3,3],
                              [1,1,1,0,0,2,0,0,0,3,3,3],
                              [1,1,1,0,2,2,2,0,0,3,3,3],
                              [1,1,1,0,2,2,2,0,0,3,3,3],
                              [1,1,1,0,2,2,2,0,0,3,3,3]])

        self.fluo_image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0],
                              [100,100,100,0,0,0,0,0,0,300,300,300],
                              [100,100,100,0,0,200,0,0,0,300,300,300],
                              [100,100,100,0,200,200,200,0,0,300,300,300],
                              [100,100,100,0,200,200,200,0,0,300,300,300],
                              [100,100,100,0,200,200,200,0,0,300,300,300]])

        self.background = 200

    def test_background_fluorescence(self):
        bkground, bkg_sem = mmeas.fluorescence_background(self.wells0, self.fluo_image)
        self.assertEqual(bkground, self.background)

if __name__ == '__main__':
    unittest.main()


