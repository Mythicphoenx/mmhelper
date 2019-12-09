"""cli.py

CLI module interface - invoked via __main__.py
using python -m mmhelper

J. Metz <metz.jp@gmail.com>
"""
import argparse
import os
import resource
from mmhelper.main import run_analysis_pipeline
from mmhelper.main import batch_run
from mmhelper.utility import logger, logging
from mmhelper.gui import run_gui


def get_args(return_parser=False):
    """
    Creates an ArgumentParser instance, adds all required
    command-line inputs and switches, and rerturns the parsed
    command-line arguments object
    """
    parser = argparse.ArgumentParser(
        description="Mother Machine analysis helper module")
    parser.add_argument("filename", nargs="*",
                        help="Data file(s) to load and analyse")
    parser.add_argument("-t", default=None, type=int,
                        help="Frame to run to for multi-frame stacks")
    parser.add_argument("--invert", action="store_true",
                        help="Invert Brightfield image")
    parser.add_argument(
        "--brightchannel", action="store_true",
        help="Detect channel as bright line instead of dark line")
    parser.add_argument("--show", action="store_true", help="Show the plots")
    parser.add_argument("--limitmem", action="store_true",
                        help="Limit memory to 1/3 of system RAM")
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument(
        "--loader", default="default",
        choices=['default', 'tifffile'], help="Switch IO method")
    parser.add_argument("--channel", type=int, default=None,
                        help="Select channel for multi-channel images")
    parser.add_argument("--tdim", type=int, default=0,
                        help="Identify time channel if not 0")
    parser.add_argument(
        "--output",
        default=None,
        help="Name of output file. Defaults to the same as input")
    parser.add_argument("--fluo", default=None,
                        help="stack of matching fluorescent images")
    parser.add_argument(
        "-f", action="store_true",
        help="Images stack contains alternating matching fluorescent images")
    parser.add_argument(
        "-ba", action="store_true",
        help="Folder containing multiple image areas to run in batch mode")
    parser.add_argument(
        "-sf", default=1, type=float,
        help="".join([
            "Scale factor for detection. "
            "Allows detection using different objectives"])
    )
    parser.add_argument(
        "-nf",
        type=int,
        default=0,
        help="Number of different fluorescent channels to be analysed")
    parser.add_argument(
        "-g", "--gui", action="store_true",
        help="Run in GUI mode")
    parser.add_argument(
        "--exit-on-error", action="store_true",
        help="Exit as soon as an error occurs instead of continuing")
    if return_parser:
        return parser.parse_args(), parser
    return parser.parse_args()


def run_cli():
    """
    Main CLI run function. This should be called to interface
    with command-line arguments
    """
    args, parser = get_args(return_parser=True)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        import sys
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(
            mode='Verbose', color_scheme='Linux', call_pdb=1)
    else:
        logger.setLevel(logging.INFO)

    if args.limitmem:
        # ------------------------------
        # Memory managment
        # ------------------------------
        gb_in_bytes = 1024.**3
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_gib = mem_bytes / gb_in_bytes  # e.g. 3.74
        lim = mem_bytes // 3
        logger.debug("Managing memory usage...")
        logger.debug("System memory: %0.2f GB", mem_gib)
        logger.debug("Limiting program to %0.2f GB", lim)
        rsrc = resource.RLIMIT_AS
        _, hard = resource.getrlimit(rsrc)
        resource.setrlimit(rsrc, (lim, hard))  # limit
    # Run appropriate analysis

    if args.gui:
        run_gui(args.debug)
        sys.exit()

    if not args.filename:
        print("At least one filename needed when not in gui mode")
        parser.print_help()
        sys.exit()

    if args.ba:
        batch_run(
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
            batchrun=args.ba,
            scale_factor=args.sf,
            num_fluo=args.nf,
            exit_on_error=args.exit_on_error,
        )

    else:
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
            batchrun=args.ba,
            scale_factor=args.sf,
            num_fluo=args.nf,
            exit_on_error=args.exit_on_error,
        )
