"""test_main.py

Test suite for the main submodule

J. Metz <metz.jp@gmail.com>
"""
import unittest
from unittest import mock
import tempfile
import os
from mmhelper import main
from mmhelper import utility
from mmhelper.tests import core

class TestBatch(unittest.TestCase):
    """
    Class for testing the loading of batch data
    """

    def setUp(self):
        self.non_existant_file = os.path.join(
            tempfile.gettempdir(),
            core.random_string(20)
        )
        self.filenames_none = None
        self.filenames_empty = []
        self.filenames_invalid = [self.non_existant_file]
        self.args_null = []
        self.kwargs_null = []

    @mock.patch('mmhelper.main.run_analysis_pipeline', autospec=True)
    @mock.patch('mmhelper.dataio.folder_all_areas', autospec=True)
    def test_batch_empty_filenames(
            self,
            mock_folder_all_areas,
            mock_run_analysis_pipeline,
    ):
        """
        Tests the response to an empty filename
        """
        self.assertRaises(ValueError,
                          main.batch_run,
                          self.filenames_empty,
                          self.args_null,
                          self.kwargs_null
                         )

    @mock.patch('mmhelper.main.run_analysis_pipeline', autospec=True)
    @mock.patch('mmhelper.dataio.folder_all_areas', autospec=True)
    def test_batch_none_filenames(
            self,
            mock_folder_all_areas,
            mock_run_analysis_pipeline
    ):
        """
        Tests teh response to no filename
        """
        self.assertRaises(ValueError,
                          main.batch_run,
                          self.filenames_none,
                          self.args_null,
                          self.kwargs_null
                         )

    @mock.patch('mmhelper.main.run_analysis_pipeline', autospec=True)
    @mock.patch('mmhelper.dataio.folder_all_areas', autospec=True)
    def test_single_file(
            self,
            mock_folder_all_areas,
            mock_run_analysis_pipeline
    ):
        """
        Tests the response to trying to load a single file
        """
        mock_folder_all_areas.return_value = {0: self.filenames_invalid}
        main.batch_run(
            self.filenames_invalid,
            self.args_null,
            self.kwargs_null
        )

    @mock.patch('mmhelper.main.run_analysis_pipeline', autospec=True)
    @mock.patch('mmhelper.dataio.folder_all_areas', autospec=True)
    def test_single_file_fail(
            self,
            mock_folder_all_areas,
            mock_run_analysis_pipeline
    ):
        """
        Tests response fails to single file
        """
        mock_folder_all_areas.return_value = {0: self.filenames_invalid}

        def error_raiser():
            """
            Raises an error
            """
            raise Exception("AAARG")
        mock_run_analysis_pipeline.side_effect = error_raiser

        with self.assertLogs(utility.logger, level='INFO') as cm0:
            main.batch_run(
                self.filenames_invalid,
                self.args_null,
                self.kwargs_null
            )
            self.assertEqual(
                cm0.output, [
                    'ERROR:mmhelper:Failed analysis on area number: 0', ])


class TestRunAnalysisPipeline(unittest.TestCase):
    """
    Class for tests on running the analysis pipeline
    """
    def setUp(self):
        self.non_existant_file = os.path.join(
            tempfile.gettempdir(),
            core.random_string(20)
        )

    def test_none_filenames(self):
        """
        Test response to no filename
        """
        self.assertRaises(TypeError, main.run_analysis_pipeline, None)

    @mock.patch("mmhelper.dataio.input_folder")
    def test_bad_file(self, mock_input_folder):
        """
        Test response to bad filename
        """
        mock_input_folder.return_value = [self.non_existant_file]
        # NB Can pass in None as input_folder is patched
        self.assertRaises(FileNotFoundError, main.run_analysis_pipeline, None)

    @mock.patch("mmhelper.dataio.input_folder")
    def test_bad_file_tifffile(self, mock_input_folder):
        """
        Test response to bad tiff file
        """
        mock_input_folder.return_value = [self.non_existant_file]
        self.assertRaises(
            FileNotFoundError,
            main.run_analysis_pipeline,
            None,
            loader="tifffile",
        )

    @mock.patch("mmhelper.dataio.input_folder")
    def test__tifffile_missing(self, mock_input_folder):
        """
        Test for missing a tifffile
        """
        import sys
        sys.path = []
        mock_input_folder.return_value = [self.non_existant_file]
        self.assertRaises(
            FileNotFoundError,
            main.run_analysis_pipeline,
            None,
            loader="tifffile",
        )


if __name__ == '__main__':
    unittest.main()
