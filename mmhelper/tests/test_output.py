"""
Functions to test the output
"""
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Test the output functions
#

import unittest
import tempfile
import os
import csv
import mmhelper.output as mout
from mmhelper.measurements_class import BacteriaData, IndividualBacteria

class TestFinalOuput(unittest.TestCase):
    """
    Class used to set up the testing of the final output
    """
    def setUp(self):
        self.measurements = BacteriaData()
        self.measurements.bacteria[1] = IndividualBacteria(1)
        self.measurements.bacteria[1].headings_line = [
            'lineage',
            'area',
            'length',
            'width',
            'raw_fluorescence',
            'fluorescence',
            'integrated_fluorescence']
        self.measurements.bacteria[1].measurements_output = [
            '1', 24, 7.118052168020874, 4.163331998932266, 180.0, 170.0, 4080.0]
        self.measurements.bacteria[2] = IndividualBacteria(2)
        self.measurements.bacteria[2].headings_line = [
            'lineage',
            'area',
            'length',
            'width',
            'raw_fluorescence',
            'fluorescence',
            'integrated_fluorescence']
        self.measurements.bacteria[2].measurements_output = [
            '2', 16, 4.898979485566356, 4.0, 200.0, 190.0, 3040.0]

        self.read_csv = [['lineage',
                          'area',
                          'length',
                          'width',
                          'raw_fluorescence',
                          'fluorescence',
                          'integrated_fluorescence'],
                         ['1',
                          '24',
                          '7.118052168020874',
                          '4.163331998932266',
                          '180.0',
                          '170.0',
                          '4080.0'],
                         ['2',
                          '16',
                          '4.898979485566356',
                          '4.0',
                          '200.0',
                          '190.0',
                          '3040.0'],
                        ]

        self.output_dir = tempfile.mkdtemp()
        self.expected_output_file = os.path.join(self.output_dir,
                                                 "Results.csv")
        # Make sure it doesn't exist!
        if os.path.isfile(self.expected_output_file):
            os.remove(self.expected_output_file)

    def test_final_output(self):
        """
        Tests the final output
        """
        mout.final_output(
            self.measurements,
            self.output_dir,
        )

        # Now check we got what we expected
        self.expected_csv_file = os.path.join(
            self.output_dir,
            'Results.csv'  # need to change this when we don't have hardcoded filename
        )
        with open(self.expected_csv_file, 'rt') as csvfile:
            output = csv.reader(csvfile, delimiter=',')
            outputtest = []
            for row in output:
                outputtest.append(row)
        self.assertTrue(outputtest == self.read_csv)

    def tearDown(self):
        # Remove the csv file
        os.remove(self.expected_csv_file)
        # Remove the temporary folder
        os.rmdir(self.output_dir)
