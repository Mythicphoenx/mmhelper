#
# FILE        : __main__.py
# CREATED     : 08/11/16 15:14:01
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Testing - with options
#



# By default run standard test discovery, but add a call to matplotlib.pyplot.show
import matplotlib.pyplot as plt
import sys
from unittest import main

if __name__ == '__main__':
    sys.path.append("..")

    #unittest.TestLoader
    main(module=None, exit=False, verbosity=2)
    plt.show()
