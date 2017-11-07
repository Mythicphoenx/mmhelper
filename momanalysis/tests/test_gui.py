#
# FILE        : test_gui.py
# CREATED     : 22/02/17 11:38:44
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Added gui testing
#

import sys
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

import momanalysis.gui as mgui

app = QApplication(sys.argv)

class TestMomanalysisGui(unittest.TestCase):
    def setUp(self):
        """
        Create the GUI
        """
        self.gui = mgui.MainWindow()

    def test_defaults(self):
        """
        Check all the default settings
        """
        # Start should be disabled
        self.assertFalse(self.gui.mw.startbutton.isEnabled())
        # Default mode is brightfield
        self.assertTrue(self.gui.mw.brightfield_box.isChecked())
        self.assertFalse(self.gui.mw.comb_fluorescence.isChecked())
        self.assertFalse(self.gui.mw.seper_fluorescence.isChecked())



if __name__ == "__main__":
    unittest.main()
