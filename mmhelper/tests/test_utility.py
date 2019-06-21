"""test_utility.py

Test utility submodule.
utility only really contains the logger definition, and
a simple function to test for color capabilities of the
terminal.

J. Metz <metz.jp@gmail.com>
"""

import unittest
from unittest import mock
from mmhelper import utility
from mmhelper.tests import core
import random
import logging


class TestSupportsColor(unittest.TestCase):
    """
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True
    """
    @mock.patch('sys.stdout.isatty', lambda: True)
    @mock.patch('sys.platform', "Pocket PC")
    def test_pocket_pc(self):
        self.assertFalse(utility.supports_color())

    @mock.patch('sys.stdout.isatty', lambda: True)
    @mock.patch('sys.platform', "win32")
    @mock.patch('os.environ', "")
    def test_win32_no_ANSICON(self):
        self.assertFalse(utility.supports_color())

    @mock.patch('sys.stdout.isatty', lambda: True)
    @mock.patch('sys.platform', "win32")
    @mock.patch('os.environ', "ANSICON")
    def test_win32_with_ANSICON(self):
        self.assertTrue(utility.supports_color())

    @mock.patch('sys.stdout.isatty', lambda: False)
    @mock.patch('sys.platform', "win32")
    @mock.patch('os.environ', "ANSICON")
    def test_win32_with_ANSICON_but_no_tty(self):
        self.assertFalse(utility.supports_color())

    @mock.patch('sys.stdout.isatty', lambda: True)
    @mock.patch('sys.platform', "linux")
    @mock.patch('os.environ', "ANSICON")
    def test_linux_with_ANSICON(self):
        self.assertTrue(utility.supports_color())

    @mock.patch('sys.stdout.isatty', lambda: False)
    @mock.patch('sys.platform', "linux")
    @mock.patch('os.environ', "ANSICON")
    def test_linux_with_ANSICON_no_tty(self):
        self.assertFalse(utility.supports_color())


class TestFormatterMessage(unittest.TestCase):
    """
    def formatter_message(message, use_color = True):
        if use_color:
            message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
            message = message.replace("$RED", COLOR_SEQ % (30 + RED))
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
            message = message.replace("$RED", "")
        return message
    """

    def setUp(self):
        self.message_parts = [core.random_string(20) for i in range(5)]
        self.tokens_in = [
            "$RESET",
            "$RED",
            "$BOLD"
        ]
        self.tokens_out = [
            utility.RESET_SEQ,
            utility.COLOR_SEQ % (30 + utility.RED),
            utility.BOLD_SEQ,
        ]
        self.message = ""
        self.message_color = ""
        for i in range(5):
            self.message += random.choice(self.message_parts)
            self.message_color += random.choice(self.message_parts)
            self.message_color += random.choice(self.tokens_in)

    def test_color_message_mono_format(self):
        messageout = utility.formatter_message(self.message_color, False)
        self.assertNotEqual(messageout, self.message_color)
        for t in self.tokens_in:
            self.assertNotIn(t, messageout)
        for t in self.tokens_out:
            self.assertNotIn(t, messageout)

    def test_color_message_color_format(self):
        messageout = utility.formatter_message(self.message_color, True)
        self.assertNotEqual(messageout, self.message_color)
        for t in self.tokens_in:
            self.assertNotIn(t, messageout)
        # Check that at least one token is in output
        tokens_out_in = [t in messageout for t in self.tokens_out]
        self.assertTrue(any(tokens_out_in))

    def test_mono_message_mono_format(self):
        messageout = utility.formatter_message(self.message, False)
        self.assertEqual(messageout, self.message)
        for t in self.tokens_in:
            self.assertNotIn(t, messageout)
        for t in self.tokens_out:
            self.assertNotIn(t, messageout)

    def test_mono_message_color_format(self):
        messageout = utility.formatter_message(self.message, True)
        self.assertEqual(messageout, self.message)
        for t in self.tokens_in:
            self.assertNotIn(t, messageout)
        for t in self.tokens_out:
            self.assertNotIn(t, messageout)


class TestColoredFormatter(unittest.TestCase):
    def setUp(self):
        self.message = ""
        self.record = mock.MagicMock()
        self.record.levelname = logging.DEBUG

    def test_create_empty_message_no_color(self):
        formatter = utility.ColoredFormatter(self.message, use_color=False)
        self.assertFalse(formatter.use_color)
        result = formatter.format(self.record)

    def test_create_empty_message_with_color(self):
        formatter = utility.ColoredFormatter(self.message, use_color=True)
        self.assertTrue(formatter.use_color)
        result = formatter.format(self.record)
