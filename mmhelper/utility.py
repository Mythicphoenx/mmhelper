#
# FILE        : utility.py
# CREATED     : 30/09/16 13:07:39
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Utlity functions, such as logging
#

import logging
import os
import sys

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = list(range(8))
# The background is set with 40 plus the number of the color, and the foreground with 30
# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace(
            "$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
        message = message.replace("$RED", COLOR_SEQ % (30 + RED))
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
        message = message.replace("$RED", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (
                30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        if self.use_color:
            record.msg = COLOR_SEQ % (30 + GREEN) + record.msg + RESET_SEQ

        return logging.Formatter.format(self, record)


class MonoFormatter(logging.Formatter):
    # def __init__(self, msg):
    #    logging.Formatter.__init__(self, msg)
    #
    # def format(self, record):
    #    return logging.Formatter.format(self, record)
    pass


class ColoredLogger(logging.Logger):
    FORMAT = "".join([
        "[$BOLD%(name)-8s$RESET][%(levelname)-18s] :",
        " ($BOLD%(filename)s$RESET:$RED%(lineno)d$RESET)",
        "  %(message)s",
    ])
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)
        color_formatter = ColoredFormatter(self.COLOR_FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        self.addHandler(console)
        return

    '''
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
        if args:
            msg += " " + " ".join(map(str, args))
            args = []
        rv = logging.LogRecord(name, level, fn, lno, msg,
                               args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if (key in ["message", "asctime"]) or (key in rv.__dict__):
                    raise KeyError(
                        "Attempt to overwrite %r in LogRecord" % key)
                rv.__dict__[key] = extra[key]
        return rv
    '''


class MonoLogger(logging.Logger):
    FORMAT = "".join([
        "[%(name)-8s][%(levelname)-18s] :",
        " (%(filename)s%(lineno)d)",
        "  %(message)s",
    ])
    #    "[$BOLD%(name)-8s$RESET][%(levelname)-18s] :",
    #    " ($BOLD%(filename)s$RESET:$RED%(lineno)d$RESET)",
    #    "  %(message)s",
    MONOFORMAT = formatter_message(FORMAT, False)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)
        self.formatter = MonoFormatter(self.MONOFORMAT)
        console = logging.StreamHandler()
        console.setFormatter(self.formatter)
        self.addHandler(console)
        return

    '''
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
        if args:
            msg += " " + " ".join(map(str, args))
            args = []
        rv = logging.LogRecord(name, level, fn, lno, msg,
                               args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if (key in ["message", "asctime"]) or (key in rv.__dict__):
                    raise KeyError(
                        "Attempt to overwrite %r in LogRecord" % key)
                rv.__dict__[key] = extra[key]
        return rv
    '''


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.

    Found on stackoverflow, apparently from Django project:
    https://github.com/django/django/blob/master/django/core/management/color.py
    """
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


if supports_color():
    logger = ColoredLogger("mmhelper")
else:
    logger = MonoLogger("mmhelper")
logger.setLevel(logging.WARNING)
