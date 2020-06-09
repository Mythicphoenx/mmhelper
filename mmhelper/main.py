"""
Functions for running the main run_analysis_pipeline
"""
import os
import traceback
import numpy as np
from mmhelper.utility import logger
import mmhelper.dataio as mio
import mmhelper.detection as mdet
import mmhelper.tracking as mtrack
import mmhelper.measurements as mmeas
import mmhelper.output as mout


def batch_run(filenames, *args, **kwargs):
    """Used to run multiple image areas one after another for batch processing

    Parameters
    ------
    filenames : list or tuple
        list or tuple containing the path to the input folder
    """
    if not isinstance(filenames, (list, tuple)):
        raise ValueError("filenames must be a list or tuple")
    if len(filenames) != 1:
        raise ValueError(
            "batch takes exactly 1 input folder for parameter `filenames`")
    filedict = mio.folder_all_areas(filenames[0])
    for area, files in sorted(filedict.items()):
        files = sorted(files)
        # feed list into analysis pipeline as well as the area number
        # (dict.key) for output folders
        try:
            run_analysis_pipeline(files, area_num=area, *args, **kwargs)
        except BaseException:
            logger.error("Failed analysis on area number: %s", area)


def run_analysis_pipeline(
        # pylint: disable=too-many-statements,too-many-branches,too-many-locals
        # pylint: disable=too-many-arguments
        filenames,
        output=None,
        tmax=None,
        invert=False,
        show=False,
        debug=False,
        brightchannel=False,
        loader='default',
        channel=None,
        tdim=0,
        fluo=None,
        fluoresc=False,
        batchrun=False,
        area_num=None,
        scale_factor=1,
        num_fluo=0,
        exit_on_error=False,
        # logger=logger,
):
    """Main analysis pipeline
    Executes the full analysis pipeline, from input images to output images and
    tables.

    Parameters
    ----------
    filenames : list of strings
        List of files to load (can be single file)
    output    : String
        Name of output file base (default : input file name)
    tmax      : Integer
        Frame to run analysis until (default : all)
    invert    : Boolean
        Whether the image needs to be inverted (default : False)
    show      : Boolean
        Whether to "show" the plots on the screen (default : False)
    debug     : Boolean
        Whether to add debugging outputs (default : False)
    brightchannel : Boolean
        Detect channel as a bright instead of dark line (default : False)
    loader  : string (options: tifffile, default)
        Which image loader to use (default : default)
    channel : Integer
        Select channel dimension from multi-channel
        image (default : None - single  channel image)
    tdim : Integer
        Select time dimension (default : 0)
    fluo : List of strings
        List of fluorescence images to load (default : None)
    fluoresc : Boolean
        Image stack contains alternating fluorescence images (default : False)
    batchrun : Boolean
        Whether input is a folder containing multiple image areas to
        run concurrently (default : False)
    area_num : Integer
        Area number for dealing with multiple areas (default : None)
    scale_factor : float, optional
        Scale factor for detection parameters (default : 1)
    num_fluo : int, optional
        The number of fluorescence channels for each brightfield image
    exit_on_error : Boolean, optional
        Exit if any errors are encountered (default : False)
    """
    # See if the input is a folder or files
    filenames = mio.input_folder(filenames)

    dir_name, output, image_dir = mmeas.find_input_filename(
        filenames[0],
        out=output,
        batch=batchrun,
        im_area=area_num,
        debug=debug)
    logger.debug("Starting run_analysis_pipeline")
    logger.debug("  invert: %s", invert)
    logger.debug("  show: %s", show)
    logger.info("Loading data...")

    if fluoresc and not num_fluo:
        num_fluo = 1

    fluo_data = None

    if loader == "tifffile":
        try:
            import tifffile
        except BaseException:
            loader = "default"

    if loader == "default":
        data = mio.load_data(filenames)
        if fluo is not None:
            fluo_data = mio.load_data(fluo)
    elif loader == "tifffile":
        data = tifffile.imread(filenames).astype(float)
    else:
        logger.error("Invalid loader specified [%s]", loader)
        return ''

    logger.debug("Initial load: %s", data.shape)
    if num_fluo >= 1:
        data, fluo_data = mio.split_fluorescence(data, num_fluo)

    if data.ndim == 2:
        data = data[None]

    if tdim:
        data = np.rollaxis(data, tdim)

    if tmax:
        data = data[:tmax]
    if channel is not None:
        data = data[:, channel, ...]
    if invert:
        if isinstance(data, (list, tuple)):
            data = [d.max() - d for d in data]
        else:
            data = data.max() - data
    logger.info("Data loaded: %s of type %s", data.shape, data.dtype)

    logger.info("Detecting channels, wells, and bacteria...")
    allwellcoords = []
    allwellimages = []
    # allridges = []
    allbacteria = []

    # ------------------------------
    # Run detection on every frame
    # ------------------------------
    for tpoint, frame in enumerate(data):
        empty_image = np.zeros(frame.shape, dtype="int16")
        try:
            if debug:
                dirpart, filenamepart = os.path.split(filenames[0])
                filenamebase = "debug_%0.4d_%s" % (tpoint, filenamepart)
                debugnow = os.path.join(dirpart, filenamebase)
                logger.debug("Saving any debugging data to [%s]", dirpart)
                logger.debug("  using the prefix [%s]", filenamebase)
            else:
                debugnow = ""
            (labelled_wellimg,
             bacteria_image,
             wellcoords,
             bacteria) = mdet.run_detection(
                 frame, brightchannel,
                 debug=debugnow, scale_factor=scale_factor)
            mout.output_detection_figures(
                frame, labelled_wellimg, bacteria_image, tpoint, image_dir)
            allwellcoords.append(wellcoords)
            allbacteria.append(bacteria)
            allwellimages.append(labelled_wellimg)
            logger.info("Detection complete on frame %s", tpoint + 1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, exiting")
            return
        except Exception:
            logger.error(
                "Detection failed for area number: %s timepoint %s",
                area_num, tpoint + 1)
            logger.error("Exception: %s", traceback.format_exc())
            if exit_on_error:
                raise
            # dictionary of well ids : coords of well pixels
            allwellcoords.append({})
            # should be dictionary well id : label array of bacteria
            allbacteria.append({})
            # should be 2d label array
            allwellimages.append(empty_image)

    # ------------------------------
    # Run tracking on every frame (after the first)
    # ------------------------------
    logger.info("Tracking...")
    (allwellimages, allwellcoords,
     allbacteria, bacteria_lineage) = mtrack.run_tracking(
         data, allwellimages, allwellcoords, allbacteria)
    logger.info("Creating tracking figures...")
    mout.output_tracking_figures(
        data,
        allwellimages,
        allwellcoords,
        allbacteria,
        image_dir,
        bacteria_lineage)

    # ------------------------------
    # Run analysis and output
    # ------------------------------
    logger.info("Measuring...")
    measurements = mmeas.get_measurements(
        data, fluo_data, allwellimages,
        allwellcoords, allbacteria, bacteria_lineage)[0]
    mout.final_output(measurements, dir_name)

    # for k, v in measurements.items():
    #    print("ID:",k)
    #    for t,vv in v.items():
    #        print("  T:",t,"Measurements:", vv)

    # mout.final_output(dir_name, output, fluo, fluoresc, final_timepoint+1)
    # Generate output images

    return dir_name
