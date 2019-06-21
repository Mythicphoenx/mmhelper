# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:51:22 2016
Unit tests for measurements in mmhelper module
@author: as624
"""

import unittest
from unittest import mock
import mmhelper.measurements as mmeas
from mmhelper.measurements_class import BacteriaData, IndividualBacteria
import numpy as np
from skimage.measure import regionprops


class TestFindFilename(unittest.TestCase):
    """
    Class for testing the function for finding the filename
    """
    def setUp(self):
        self.inputname = 'testinputfilename.test'

        self.output = 'testinputfilename'

    @mock.patch('os.makedirs')
    def test_input_filename(self, mock_osmakedirs):
        """
        Tests the innput filename is found properly
        """
        self.assertEqual(
            self.output, mmeas.find_input_filename(self.inputname)[1])
        # Check that makedirs was called twice
        self.assertEqual(mock_osmakedirs.call_count, 2)


class TestCounts(unittest.TestCase):
    """
    Class for testing counting the number of bacteria in wells
    """
    def setUp(self):
        self.wells = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0],
        ])
        self.bacteria = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 2, 0, 0, 0],
        ])
        self.counts = [2, 1]

        # Using just individual labels
        self.bacteria_labels = {
            1: np.array([[1, 0, 2, 2, 2, 0, 3], ]),
            2: np.array([[0, 0, 0], ]),
            3: np.array([[1, ], ])
        }
        self.counts2 = [3, 0, 1]

    def test_counts_just_labels(self):
        """
        Tests function count_bacteria_in_wells
        """
        self.assertEqual(
            self.counts2, mmeas.count_bacteria_in_wells(self.bacteria_labels))


class TestBacteriaMeasurements(unittest.TestCase):
    """
    Class for testing bacteria measurements
    """
    def setUp(self):
        self.bacteria_labels = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0],
            [2, 2, 2, 2, 0, 0],
            [2, 2, 2, 2, 0, 0],
            [2, 2, 2, 2, 0, 0],
            [0, 2, 2, 0, 0, 0],
        ])

        self.bacteria_fluo = [np.array([
            [0, 180, 180, 0, 0, 0],
            [180, 180, 180, 180, 0, 0],
            [180, 180, 180, 180, 0, 0],
            [180, 180, 180, 180, 0, 0],
            [180, 180, 180, 180, 0, 0],
            [180, 180, 180, 180, 0, 0],
            [0, 180, 180, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 200, 200, 0, 0, 0],
            [200, 200, 200, 200, 0, 0],
            [200, 200, 200, 200, 0, 0],
            [200, 200, 200, 200, 0, 0],
            [0, 200, 200, 0, 0, 0]
        ])]
        self.well_label = 42
        self.measurements = BacteriaData()
        self.measurements.bacteria[1] = IndividualBacteria(1)
        self.measurements.bacteria[1].bacteria_label = '2'
        self.measurements.bacteria[1].bf_measurements = {
            "Well label": [self.well_label], "Area": [24],
            "Width": [4], "Length": [7]}
        self.measurements.bacteria[1].raw_fluorescence = {(0, 0): 180}
        self.measurements.bacteria[1].actual_fluorescence = {(0, 0): 170}
        self.measurements.bacteria[1].integrated_fluorescence = {(0, 0): 4080}
        self.measurements.bacteria[1].output_line = []
        self.measurements.bacteria[1].timepoints = [1]
        self.measurements.bacteria[1].num_fluo = 0
        self.measurements.bacteria[1].headings_line = [
            'well label',
            'lineage',
            'area',
            'length',
            'width',
            'raw_fluorescence',
            'fluorescence',
            'integrated_fluorescence']
        self.measurements.bacteria[1].measurements_output = [
            self.well_label, '1', 24, 7.118052168020874,
            4.163331998932266, 180.0, 170.0, 4080.0]

        self.measurements.bacteria[2] = IndividualBacteria(2)
        self.measurements.bacteria[2].bacteria_label = '2'
        self.measurements.bacteria[2].bf_measurements = {
            "Well label": [self.well_label],
            "Area": [16], "Width": [4], "Length": [5]}
        self.measurements.bacteria[2].raw_fluorescence = {(0, 0): 200}
        self.measurements.bacteria[2].actual_fluorescence = {(0, 0): 190}
        self.measurements.bacteria[2].integrated_fluorescence = {(0, 0): 3040}
        self.measurements.bacteria[2].output_line = []
        self.measurements.bacteria[2].timepoints = [1]
        self.measurements.bacteria[2].num_fluo = 0
        self.measurements.bacteria[2].headings_line = [
            'well label',
            'lineage',
            'area',
            'length',
            'width',
            'raw_fluorescence',
            'fluorescence',
            'integrated_fluorescence']
        self.measurements.bacteria[2].measurements_output = [
            self.well_label, '2', 16, 4.898979485566356, 4.0,
            200.0, 190.0, 3040.0]

        self.lbl_dict = {1: '1', 2: '2'}
        self.timepoint = 0
        self.fluo_values = [(10, 1), ]

    def test_bf_measurements(self):
        """
        Tests the measuring of bacteria brightfield measurments
        """
        measurements = BacteriaData()
        for region in regionprops(self.bacteria_labels):
            measurements.add_bac_data(
                region.label, self.lbl_dict, region, self.timepoint,
                well_label=self.well_label)
        for bac_num, actual_data in measurements.bacteria.items():
            manual_data = self.measurements.bacteria[bac_num]
            for meas, actual_val in actual_data.bf_measurements.items():
                manual_val = manual_data.bf_measurements[meas]
                self.assertTrue((manual_val[0] - 0.5) <=
                                actual_val[0] <= (manual_val[0] + 0.5))

    def test_fluo_measurements(self):
        """
        Tests the measuring of bacteria fluorescence measurements
        """
        measurements = BacteriaData()
        for region in regionprops(self.bacteria_labels):
            measurements.add_bac_data(
                region.label, self.lbl_dict, region, self.timepoint)
            measurements.measure_fluo(
                region, self.bacteria_fluo, self.fluo_values, self.timepoint)
        for bac_num, actual_data in measurements.bacteria.items():
            manual_data = self.measurements.bacteria[bac_num]
            self.assertTrue((manual_data.raw_fluorescence) ==
                            (actual_data.raw_fluorescence))
            self.assertTrue((manual_data.actual_fluorescence)
                            == (actual_data.actual_fluorescence))
            self.assertTrue((manual_data.integrated_fluorescence)
                            == (actual_data.integrated_fluorescence))

    def test_measurements_output(self):
        """
        Tests the measurements can be compiled properly for output
        """
        measurements = BacteriaData()
        for region in regionprops(self.bacteria_labels):
            measurements.add_bac_data(
                region.label, self.lbl_dict, region, self.timepoint,
                well_label=self.well_label)
            measurements.measure_fluo(
                region, self.bacteria_fluo, self.fluo_values, self.timepoint)
        measurements.compile_results(max_tpoint=1)
        for bac_num, actual_data in measurements.bacteria.items():
            manual_data = self.measurements.bacteria[bac_num]
            # Couldn't use assertListEqual here
            # because of floating-point inequality
            self.assertEqual(
                len(manual_data.measurements_output),
                len(actual_data.measurements_output))
            for item1, item2 in zip(
                    manual_data.measurements_output,
                    actual_data.measurements_output):
                self.assertAlmostEqual(item1, item2)
            self.assertListEqual(
                manual_data.headings_line,
                actual_data.headings_line)


class TestFluoMeasurements(unittest.TestCase):
    """
    Class for testing measuring fluorescence
    """
    def setUp(self):
        self.bacteria_labels = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.fluo_image = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 3000, 3000],
             [2000, 2000, 2000, 0, 0, 0, 0, 0, 0, 3000, 3000, 3000],
             [2100, 2100, 2100, 0, 0, 0, 0, 0, 0, 3800, 3800, 3800],
             [2100, 2100, 2100, 0, 0, 0, 0, 0, 0, 3000, 3000, 3000],
             [2200, 2200, 2200, 0, 0, 0, 0, 0, 0, 3000, 3000, 3000],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 3000, 3000],
             [0, 0, 0, 3200, 3200, 3200, 0, 0, 0, 3000, 3000, 3000],
             [0, 0, 0, 3300, 3300, 3300, 0, 0, 0, 3000, 3000, 3000],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.bkground = 100

        self.bkg_sem = 1

        self.fluo_results = [2100, 2000, 24000]
        self.fluo_results2 = [3250, 3150, 18900]
        self.fluo_results3 = [3100, 3000, 72000]

        self.fluo_measurements = [self.fluo_results,
                                  self.fluo_results2, self.fluo_results3]

    def test_fluorescentmeasurements(self):
        """
        Tests the function fluorescence_measurements
        """
        fluo_output = []
        for region in regionprops(self.bacteria_labels):
            fluo, background_fluo, integrated_fluorescence = \
                mmeas.fluorescence_measurements(
                    region, self.fluo_image, (self.bkground, self.bkg_sem))
            fluo_output.append(
                [fluo, background_fluo, integrated_fluorescence])
        self.assertEqual(self.fluo_measurements, fluo_output)


class TestFluorescenceBackground(unittest.TestCase):
    """
    Class for testing the measurements of fluorescence background
    """
    def setUp(self):
        self.wells = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3],
                               [1, 1, 1, 0, 0, 2, 0, 0, 0, 3, 3, 3],
                               [1, 1, 1, 0, 2, 2, 2, 0, 0, 3, 3, 3],
                               [1, 1, 1, 0, 2, 2, 2, 0, 0, 3, 3, 3],
                               [1, 1, 1, 0, 2, 2, 2, 0, 0, 3, 3, 3]])

        self.bact = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 4, 4],
                              [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 4, 4],
                              [0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 4, 4],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4]])

        self.fluo_image = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [100, 100, 100, 0, 0, 0, 0, 0, 0, 300, 300, 300],
             [100, 100, 100, 0, 0, 200, 0, 0, 0, 300, 300, 300],
             [100, 100, 100, 0, 200, 200, 200, 0, 0, 300, 300, 300],
             [100, 100, 100, 0, 200, 200, 200, 0, 0, 300, 300, 300],
             [100, 100, 100, 0, 200, 200, 200, 0, 0, 300, 300, 300]])
        # Background sum should be 13*100 + 5*200 + 7*300 = 4400
        # Mean = 4400 / 25 = 176

        self.background = 176

    def test_background_fluorescence(self):
        """
        Tests the function for determining the fluorescence_background
        """
        bkground = mmeas.fluorescence_background(
            self.wells,
            self.bact,
            self.fluo_image,
        )[0]

        self.assertEqual(bkground, self.background)


if __name__ == '__main__':
    unittest.main()
