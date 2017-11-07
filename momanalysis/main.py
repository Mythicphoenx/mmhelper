
from momanalysis.utility import logger
import momanalysis.io as mio
import momanalysis.plot as mplot
import momanalysis.detection as mdet
import momanalysis.tracking as mtrack
import momanalysis.measurements as mmeas
import momanalysis.output as outp
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np

def batch(filenames, *args, **kwargs):
    in_folder = []
    in_folder.append(os.path.abspath(filenames[0]))
    """Used to run multiple image areas one after another for batch processing"""
    filedict = mio.folder_all_areas(in_folder)
    for area, files in sorted(filedict.items()):
        files = sorted(files)
        #feed list into analysis pipeline as well as the area number (dict.key) for output folders
        try:
            run_analysis_pipeline(files, area_num=area, *args, **kwargs)
        except:
            logger.error("Failed analysis on area number: %s" %(area))

def run_analysis_pipeline(
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
        batch=False,
        area_num = None
        #logger=logger,
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
        Whether to detect channel as a bright instead of dark line (default : False)
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
        Whether image stack contains alternating fluorescence images (default : False)
    batch : Boolean
        Whether input is a folder containing multiple image areas to
        run concurrently (default : False)
    area_num : Integer
        Area number for dealing with multiple areas (default : None)
    """
    #See if the input is a folder or files
    filenames = mio.input_folder(filenames)

    label_dict_string = None

    dir_name, output, image_dir = mmeas.find_input_filename(filenames[0], out = output, batch = batch, im_area = area_num)

    logger.debug("Starting run_analysis_pipeline")
    logger.debug("  invert:", invert)
    logger.debug("  show:", show)
    logger.info("Loading data...")
    fluo_data = None

    if loader == "tifffile":
        try:
            import tifffile
        except:
            loader = "default"

    if loader == "default":
        data = mio.load_data(filenames)
        if fluo is not None:
            fluo_data = mio.load_data(fluo)
    elif loader == "tifffile":
        data = tifffile.imread(filenames).astype(float)
    else:
        logger.error("Invalid loader specified [%s]" % str(loader))
        return

    logger.debug("Initial load:", data.shape)
    if fluoresc:
        data, fluo_data = mio.split_fluorescence(data)

    if data.ndim == 2:
        data = data[None]

    if fluo is not None or fluoresc is not False:
        if fluo_data.ndim ==2:
            fluo_data = fluo_data[None]

    if tdim:
        data = np.rollaxis(data, tdim)

    if tmax:
        data = data[:tmax]
    if channel is not None:
        data = data[:,channel,...]
    if invert:
        raise Exception("HANDLE ME BETTER")
        if isinstance(data, (list, tuple)):
            data = [ d.max() - d for d in data ]
        else:
            data = data.max() - data
    logger.info("Data loaded:", data.shape, "of type", data.dtype)

    logger.info("Detecting channels, wells, and extracting well profiles...")
    allwells = []
    allchannels = []
    allridges = []
    allbacteria = []
    figs = []
    lastimage =None
    lastframe = None
    lastbac = None
    final_timepoint = len(data)-1
    for t, d in enumerate(data):
        try:
            lastframe, channel, ridges, wells, bacteria, label_dict_string, lastimage, lastbac = run_frame_analysis(t,d,fluo_data,label_dict_string,lastframe,final_timepoint,lastbac,lastimage,image_dir,dir_name,output,tmax,invert,show,debug,brightchannel,loader,channel,tdim,fluo,fluoresc,batch,area_num)
            allwells.append(wells)
            allchannels.append(channel)
            allridges.append(ridges)
            allbacteria.append(bacteria)
        except:
            logger.error("Analysis failed for area number: %s timepoint %s" %(area_num,t))
            logger.error("Exception:%s" % traceback.format_exc())
    outp.final_output(dir_name, output, fluo, fluoresc, final_timepoint+1)
    return dir_name

def run_frame_analysis(t,d,fluo_data,label_dict_string,lastframe,final_timepoint,lastbac,lastimage,image_dir,dir_name,output,tmax,invert,show,debug,brightchannel,loader,channel,tdim,fluo,fluoresc,batch,area_num):

    logger.info("T = %d"%t)
    tracked = False
    if debug:
        plt.figure()
        plt.imshow(d)
        plt.savefig("DEBUG_DATAINL_%0.6d.jpg"%t)
        plt.close()
    channel = mdet.detect_channel(d, bright=brightchannel)
    if debug:
        plt.figure()
        plt.imshow(channel)
        plt.savefig("DEBUG_CHANNEL_%0.6d.jpg"%t)
        plt.close()
    wells0, ridges = mdet.detect_wells(d)
    if debug:
        plt.figure()
        plt.imshow(wells0)
        plt.savefig("DEBUG_WELLS_INITIAL_%0.6d.jpg"%t)
        plt.close()
    if lastframe is not None:
        # Estimate transform
        tform = mtrack.frametracker(lastframe, d, debug=debug)

        logger.debug("\tFrame tracking registered a transform of:", str(tform))
        # Update the well labels
        # INFO
        # at the moment the second output is how many wells
        # the frame has shifted - need to check this -
        # the value is used in bacteria tracking to allign
        # the wells as they aren't labelled
        profiles0, wellimg, coords = mdet.extract_well_profiles(d, channel, wells0, debug=debug)
        wells = mtrack.welltracking(lastimage, wellimg, tform)
        tracked = True
    else:        
        wells = wells0
    lastwells = wells
    if debug:
        plt.figure()
        plt.imshow(wells)
        plt.savefig("DEBUG_WELLS_POST_TRACKING_%0.6d.jpg"%t)
        plt.close()

    #wells = mdet.detect_wells(d)
    profiles0, wellimg, coords = mdet.extract_well_profiles(d, channel, wells, debug=debug, tracked=tracked)
    lastimage =  wellimg

    if len(profiles0) < 2:
        logger.error("Less than 2 wells detected in extract_well_profiles, aborting")
        return

    #print("DEBUG: SAVING WELL PROFILES (WITHOUT BG SUBTRACTION")
    #import skimage.io as skio
    #skio.imsave("DEBUG_profiles.tif", np.array(profiles0))

    profiles = mdet.remove_background(profiles0)
    logger.info("\t%d wells extracted..."%(len(profiles)))
    if not profiles:
        logger.warn("\tNo wells extracted, aborting")
        if debug:
            plt.figure()
            plt.imshow(d, cmap='gray')
            plt.title("Data frame")
            plt.figure()
            plt.imshow(channel)
            plt.title("Channels")
            plt.figure()
            plt.imshow(ridges)
            plt.title("Ridges")
            plt.figure()
            plt.imshow(wells0)
            plt.title("Initial wells")
            plt.figure()
            plt.imshow(wells)
            plt.title("Wells after tracking")
            plt.figure()
            plt.imshow(wellimg)
            plt.title("After extracting profiles")
            plt.show()
        return


    # Detect bacteria
    if lastframe is not None:
        bacteria = mdet.detect_bacteria_in_wells(profiles, t, label_dict_string)
        bacteria, label_dict_string = mdet.split_bacteria(bacteria, label_dict_string, t)
        bacteria, label_dict_string = mtrack.bacteria_tracking(lastbac, bacteria, label_dict_string)
        bac_counts = mmeas.count_bacteria_in_wells(bacteria)
        total_bac = sum(bac_counts)
        logger.info("\t%d bacteria (total)..."%total_bac)
        logger.info("\tper well:", str(bac_counts))
    else:
        bacteria = mdet.detect_bacteria_in_wells(profiles, t, label_dict_string)
        bacteria, label_dict_string = mdet.split_bacteria(bacteria, label_dict_string, t)
        #logger.info("\t%d bacteria (total)..."%max((b.max() for b in bacteria)))
        bac_counts = mmeas.count_bacteria_in_wells(bacteria)
        total_bac = sum(bac_counts)
        logger.info("\t%d bacteria (total)..."%total_bac)
        logger.info("\tper well:", str(bac_counts))

    lastbac = bacteria
    numberbac = max(b.max() for b in bacteria.values()) #this takes the max label from the previous image
    lastframe = d
    segfull = np.zeros(wellimg.shape, dtype="int16")
    for s, c in zip(bacteria, coords):
        segfull[c] = bacteria[s]
    # Generate output images
    outp.output_figures(d, t, channel, wells, profiles, bacteria, coords, wellimg, ridges, image_dir, label_dict_string)
    #Measurements and output
    if fluo is None and fluoresc is False:
        bac_meas = mmeas.bacteria_measurements(label_dict_string, segfull)
        outp.output_csv(bac_meas,dir_name, output, t)
    else:
        bkground, bg_sem = mmeas.fluorescence_background(wells0,segfull, fluo_data[t])#need to use this in measurements and output
        fluo_meas = mmeas.fluorescence_measurements(segfull, bkground, bg_sem, fluo_data[t]) #while writing fluorescence function
        bac_meas = mmeas.bacteria_measurements(label_dict_string, segfull, fluo_values=fluo_meas)
        outp.output_csv(bac_meas,dir_name, output, t, bg=bkground, fl=True)
    return lastframe, channel, ridges, wells, bacteria, label_dict_string, lastimage, lastbac

