"""
The __main__.py file for mmhelper

Takes the command line arguments and passes them into the main module
"""
# FILE        : __main__.py
# CREATED     : 22/09/16 12:48:59
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : CLI module interface - invoked using python -m mmhelper
#

import argparse
from mmhelper.main import run_analysis_pipeline
from mmhelper.main import batch_run

from mmhelper.utility import logger, logging

# Parse command line arguments

PARSER = argparse.ArgumentParser(
    description="MOther Machine ANALYSIS module")
PARSER.add_argument("filename", nargs="+",
                    help="Data file(s) to load and analyse")
PARSER.add_argument("-t", default=None, type=int,
                    help="Frame to run to for multi-frame stacks")
PARSER.add_argument("--invert", action="store_true",
                    help="Invert Brightfield image")
PARSER.add_argument("--brightchannel", action="store_true",
                    help="Detect channel as bright line instead of dark line")
PARSER.add_argument("--show", action="store_true", help="Show the plots")
PARSER.add_argument("--limitmem", action="store_true",
                    help="Limit memory to 1/3 of system RAM")
PARSER.add_argument("--debug", action="store_true", help="Debug")
PARSER.add_argument(
    "--loader",
    default="default",
    choices=[
        'default',
        'tifffile'],
    help="Switch IO method")
PARSER.add_argument("--channel", type=int, default=None,
                    help="Select channel for multi-channel images")
PARSER.add_argument("--tdim", type=int, default=0,
                    help="Identify time channel if not 0")
PARSER.add_argument(
    "--output",
    default=None,
    help="name of output file, if not specified it will be the same as input")
PARSER.add_argument("--fluo", default=None,
                    help="stack of matching fluorescent images")
PARSER.add_argument(
    "-f",
    action="store_true",
    help="stack of images is contains alternating matching fluorescent images")
PARSER.add_argument(
    "-ba",
    action="store_true",
    help="a folder containing multiple image areas to run at the same time")
PARSER.add_argument(
    "-sf", default=1, type=float,
    help="a scale factor for detection, to allow detection for different objectives")
PARSER.add_argument(
    "-nf",
    type=int,
    default=0,
    help="Number of different fluorescent reporter channels to be analysed")
ARGS = PARSER.parse_args()


if ARGS.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


if ARGS.limitmem:
    # ------------------------------
    # Memory managment
    # ------------------------------
    import resource
    import os
    GB = 1024.**3
    MEM_BYTES = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    MEM_GIB = MEM_BYTES / GB  # e.g. 3.74
    logger.debug("MANAGING MEMORY USAGE...")
    logger.debug("SYSTEM MEMORY: %0.2f GB" % MEM_GIB)
    logger.debug("LIMITING PROGRAM TO %0.2f GB" % (MEM_GIB / 3))
    LIM = MEM_BYTES // 3
    RSRC = resource.RLIMIT_AS
    SOFT, HARD = resource.getrlimit(RSRC)
    resource.setrlimit(RSRC, (LIM, HARD))  # limit
# Run appropriate analysis

if ARGS.ba is True:
    batch_run(
        ARGS.filename,
        output=ARGS.output,
        tmax=ARGS.t,
        invert=ARGS.invert,
        show=ARGS.show,
        debug=ARGS.debug,
        brightchannel=ARGS.brightchannel,
        loader=ARGS.loader,
        channel=ARGS.channel,
        tdim=ARGS.tdim,
        fluo=ARGS.fluo,
        fluoresc=ARGS.f,
        batchrun=ARGS.ba,
        scale_factor=ARGS.sf,
        num_fluo=ARGS.nf,
    )

elif ARGS.ba is False:
    run_analysis_pipeline(
        ARGS.filename,
        output=ARGS.output,
        tmax=ARGS.t,
        invert=ARGS.invert,
        show=ARGS.show,
        debug=ARGS.debug,
        brightchannel=ARGS.brightchannel,
        loader=ARGS.loader,
        channel=ARGS.channel,
        tdim=ARGS.tdim,
        fluo=ARGS.fluo,
        fluoresc=ARGS.f,
        batchrun=ARGS.ba,
        scale_factor=ARGS.sf,
        num_fluo=ARGS.nf,
    )
