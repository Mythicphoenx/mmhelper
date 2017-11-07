#
# FILE        : __main__.py
# CREATED     : 22/09/16 12:48:59
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : CLI module interface - invoked using python -m momanalysis
#

import argparse
from momanalysis.main import run_analysis_pipeline
from momanalysis.main import batch

from momanalysis.utility import logger

logger.setLevel("DEBUG")

# Parse command line arguments

parser = argparse.ArgumentParser(
    description="MOther Machine ANALYSIS module")
parser.add_argument("filename", nargs="+", help="Data file(s) to load and analyse")
parser.add_argument("-t", default=None, type=int, help="Frame to run to for multi-frame stacks")
parser.add_argument("--invert", action="store_true", help="Invert Brightfield image")
parser.add_argument("--brightchannel", action="store_true", help="Detect channel as bright line instead of dark line")
parser.add_argument("--show", action="store_true", help="Show the plots")
parser.add_argument("--limitmem", action="store_true", help="Limit memory to 1/3 of system RAM")
parser.add_argument("--debug", action="store_true", help="Debug")
parser.add_argument("--loader", default="default", choices=['default', 'tifffile'], help="Switch IO method")
parser.add_argument("--channel", type=int, default=None, help="Select channel for multi-channel images")
parser.add_argument("--tdim", type=int, default=0, help="Identify time channel if not 0")
parser.add_argument("--output", default=None, help ="name of output file, if not specified it will be the same as input")
parser.add_argument("--fluo", default=None, help ="stack of matching fluorescent images")
parser.add_argument("-f", action="store_true", help="stack of images is contains alternating matching fluorescent images")
parser.add_argument("-ba", action="store_true", help="a folder containing multiple image areas to run at the same time")
args = parser.parse_args()


if args.limitmem:
    #------------------------------
    # Memory managment
    #------------------------------
    import resource
    import os
    gb = 1024.**3
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes/gb  # e.g. 3.74
    logger.debug("MANAGING MEMORY USAGE...")
    logger.debug("SYSTEM MEMORY: %0.2f GB" % mem_gib)
    logger.debug("LIMITING PROGRAM TO %0.2f GB" % (mem_gib/3))
    lim = mem_bytes//3
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (lim, hard)) #limit
# Run appropriate analysis

if args.ba == True:
    batch(
        args.filename,
        output=args.output,
        tmax=args.t,
        invert=args.invert,
        show=args.show,
        debug=args.debug,
        brightchannel=args.brightchannel,
        loader=args.loader,
        channel=args.channel,
        tdim=args.tdim,
        fluo=args.fluo,
        fluoresc=args.f,
        batch=args.ba
        )  
    
elif args.ba == False:
    run_analysis_pipeline(
        args.filename,
        output=args.output,
        tmax=args.t,
        invert=args.invert,
        show=args.show,
        debug=args.debug,
        brightchannel=args.brightchannel,
        loader=args.loader,
        channel=args.channel,
        tdim=args.tdim,
        fluo=args.fluo,
        fluoresc=args.f,
        batch=args.ba
        )



