#
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Test the output functions
#

import unittest
import numpy as np
import tempfile
import skimage.io as skio
import os
import momanalysis.output as mout
import traceback
import csv

class TestOutputCSV(unittest.TestCase):
    def setUp(self):
        #
        # IMPORTANT NOTE
        #
        # This is run before EACH class method
        # Similarly the tearDown is run after EACH class
        # method - so they are not once-only initializations / cleanups
        #
        # This means for example, that we can re-use the output directory
        # between the normal and fluoro example.
        self.inputmeasurements = [[1,4,8,10],[2,6,9,13]]

        self.readCSV = [["Bacteria Number", "Area", "Length", "Width"],
                        ['1','4','8','10'],
                        ['2','6','9','13',]]

        # This is deprecated... (see tempfile docs)
        #self.csv_test_filename = tempfile.mktemp()
        #self.csv_test_filename_1 = tempfile.mktemp()
        self.output_dir = tempfile.mkdtemp()
        _, self.csv_test_filename = tempfile.mkstemp(
            dir=self.output_dir)
        # Using this method, we create some intermediate temporary files
        # that we don't need...we're only using this to
        # generate the base filename
        # TODO: Probably better to use mkdtemp and then just create a
        # random string instead of using tempfile for the filename base
        os.remove(self.csv_test_filename)

        # Now we only need the final parts of the filename
        # at the moment, though I propose to pass in full paths and remove
        # passing in a directory name and a relative file name...
        self.csv_test_filename = os.path.basename(self.csv_test_filename)
        self.inputmeasurements_fluo = [[1,4,8,20,15,33,60],[2,6,9,35,30,27,180]]

        self.bkground = 5

        self.readCSV_2 = [['Background', '5', '','','','',''],
                        ["Bacteria Number", "Area", "Length", "Width", "fluorescence", "fluo - background", "fluo sum - background"],
                        ['1','4','8','20','15','33','60'],
                        ['2','6','9','35','30','27','180']]


    def test_input_filename(self):
        # Run the csv output function
        mout.output_csv(
            self.inputmeasurements,
            self.output_dir,
            self.csv_test_filename,
            1)
        # Now check we got what we expected
        self.expected_csv_file = os.path.join(
            self.output_dir,
            '%s_timepoint_%d.csv' % (self.csv_test_filename,1)
        )
        with open( self.expected_csv_file, 'rt') as csvfile:
            output = csv.reader(csvfile, delimiter=',')
            outputtest = []
            for row in output:
                outputtest.append(row)
        self.assertEqual(outputtest, self.readCSV)

    def test_input_filename_fluo(self):
        mout.output_csv(
            self.inputmeasurements_fluo,
            self.output_dir,
            self.csv_test_filename,
            2,
            bg=self.bkground,
            fl=True)

        self.expected_csv_file = os.path.join(
            self.output_dir,
            '%s_timepoint_%d.csv' % (self.csv_test_filename,2)

        )
        with open( self.expected_csv_file, 'rt') as csvfile:
            output = csv.reader(csvfile, delimiter=',')
            outputtest = []
            for row in output:
                outputtest.append(row)
        self.assertEqual(outputtest, self.readCSV_2)

    def tearDown(self):
        # Remove the csv file
        os.remove(self.expected_csv_file)
        # Remove the temporary folder
        os.rmdir(self.output_dir)


class TestOuputFigure(unittest.TestCase):
    """
    Test generating output figures:
    * Data image with
        - contour of well + label text
        - contour of bacteria + label text
    * Data image with
        - Semi-transparent overlay of channel + wells segmentation
    * Ridges image
    The output folder is an input, and the output pattern is
    wells_%0.6d_%0.6d.jpg%(t, i))
    where t is the time index (pass as input)
    and i is the figure number (auto-generated, should range from 0 to 2 inclusive)
    """
    def setUp(self):
        # Create data
        # "Realistic"
        #self.size = (400,600)
        #self.data = 10*np.random.randn(self.size) + 100
        # Toy
        self.data = np.array([
            [ 10, 12, 12, 11, 10],
            [ 11,  5, 11,  4,  9],
            [ 12,  4, 11,  5, 11],
            [ 10,  6, 12,  4, 12],
            [ 10, 12, 12, 11, 10],
        ])
        self.wells = np.array([
            [  0,  0,  0,  0,  0],
            [  0,  1,  0,  2,  0],
            [  0,  1,  0,  2,  0],
            [  0,  1,  0,  2,  0],
            [  0,  0,  0,  0,  0],
        ])
        self.ridges = np.array([
            [  0,  0,  0,  0,  0],
            [  0,  1,  0,  1,  0],
            [  0,  1,  0,  1,  0],
            [  0,  1,  0,  1,  0],
            [  0,  0,  0,  0,  0],
        ])
        # Old style bacteria...
        #self.bacteria = np.array([
        #    [  0,  0,  0,  0,  0],
        #    [  0,  1,  0,  3,  0],
        #    [  0,  0,  0,  0,  0],
        #    [  0,  2,  0,  0,  0],
        #    [  0,  0,  0,  0,  0],
        #])
        self.bacteria = {
            1:np.array([1]),
            2:np.array([2]),
            3:np.array([3]),
        }
        self.label_dict_string = {
            1:"BACT 1",
            2:"BACT 2",
            3:"BACT 3",
        }
        self.coords = [
            ([1,1]),
            ([1,3]),
            ([3,1]),
        ]
        self.channel = np.array([
            [  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  0],
            [  1,  1,  1,  1,  1],
        ])

        # Output destination
        self.folder = tempfile.mkdtemp()
        self.pattern = "wells_%0.6d_%0.6d.jpg" # Take t then i
        self.NUMFIGS = 3

        # Dummy time index
        self.tindex = 42

        # Files that should be created
        self.expected_figures = [ os.path.join(
                self.folder,
                self.pattern % (self.tindex, i)
            ) for i in range(self.NUMFIGS)
        ]

        # Create images myself

    def test_output_figures(self):
        mout.output_figures(
            self.data,
            self.tindex,
            self.channel,
            self.wells,
            None, #profiles, UNUSED
            self.bacteria,
            self.coords,
            self.wells,
            self.ridges,
            self.folder,
            self.label_dict_string,
        )
        # Make sure the expected figures have been cerated
        for f in self.expected_figures:
            self.assertTrue(os.path.isfile(f))

    def tearDown(self):
        # Delete files
        for f in self.expected_figures:
            os.remove(f)
        # Delete folder
        os.rmdir(self.folder)

class TestFinalOuput(unittest.TestCase):

    def setUp(self):
        # Default value for non-fluoro data
        self.fluo = None
        self.fluoresc = False

        # Location of individual csv files
        self.folder = tempfile.mkdtemp()

        # How many time points
        self.tmax = 50
        # How many bacteria per time point?
        self.nbac = 5
        # Number of measurements is 4 for non-fluoro data
        self.nmeas = 4 # (See headers below)

        # Make some (tmax) csv files
        self.csv_files = []
        header = ["Bacteria Number", "Area", "Length", "Width"]
        _, self.filename = tempfile.mkstemp(
            dir=self.folder,
            #suffix='timepoint_%d.csv'%t
        )
        # Remove the actual file - don't need it
        os.remove(self.filename)

        self.expected_output_file = os.path.join(self.folder,
            "Combined_Output.csv")
        # Make sure it doesn't exist!
        if os.path.isfile(self.expected_output_file):
            os.remove(self.expected_output_file)


        for t in range(self.tmax):
            fname = "%s_timepoint_%d.csv"%(self.filename, t)
            with open(fname, "w") as fout:
                wrt = csv.writer(fout)
                wrt.writerow(header)
                for nb in range(1, self.nbac+1):
                    wrt.writerow([
                        nb,                             # Label
                        np.random.randint(100,200),     # Area
                        np.random.randint(10,20),       # Length
                        np.random.randint(5, 10),       # Width
                    ])
            self.csv_files.append(fname)



    def test_final_output(self):
        mout.final_output(
            self.folder,
            self.filename,
            self.fluo,
            self.fluoresc,
            self.tmax,
        )
        # Make sure the output file exists and looks right
        self.assertTrue(os.path.isfile(self.expected_output_file))
        # Make sure it has the correct dimensions
        csvdata = list(csv.reader(open(self.expected_output_file)))
        # Should have a time index row, a header row, plus nbac measurement rows
        self.assertEqual(len(csvdata), 2+self.nbac)
        # Should have nmeas * tmax columns
        for row in csvdata:
            self.assertEqual(len(row), self.nmeas*self.tmax)


    def tearDown(self):
        for f in self.csv_files:
            os.remove(f)
        try:
            os.remove(os.path.join(self.folder, "Combined_Output.csv"))
        except:
            pass
        os.rmdir(self.folder)
