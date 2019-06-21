# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:17:59 2016

@author: as624
"""

import os
from time import strftime
from math import sqrt
import scipy.ndimage as ndi
from skimage.measure import regionprops
import numpy as np
from mmhelper.utility import logger
from mmhelper.measurements_class import BacteriaData


def find_input_filename(fname, out=None, batch=False,
                        im_area=None, debug=False):
    """
    Takes the filename input and splits it into simply the filename
    which is then used to name the csv later
    Also creates output directories.

    Parameters
    ------
    fname : string
        A string containing the input filename
    out : string, optional
        A string containing an optional output file_name (default : None)
    batch : Boolean, optional
        Whether or not the analysis is being run in batch
        mode (default : False)
    im_area : string, optional
        The area of the chip being analysed (default : None)
    debug : Boolean, optional
        If true then the filename is stamped with 'debug' instead of the time

    Returns
    ------
    new_dir : string
        The full path for the main output directory for the results
    file_name : string
        The name of the input file
    image_dir : string
        The full path to the the directory for output images

    """
    dir_name = os.path.split(os.path.realpath(fname))[0]

    timestamp = "debug" if debug else strftime("%Y%m%d_%H%M%S")

    if out is None:
        file_name = os.path.splitext(os.path.basename(fname))[0]
        if batch is True:
            new_folder = "Area" + im_area + "_results_" + timestamp
        else:
            new_folder = file_name + "_results_" + timestamp
        new_dir = os.path.join(dir_name, new_folder)
    else:
        file_name = out
        new_folder = file_name
        new_dir = os.path.join(dir_name, new_folder)

    img = "output_images"
    image_dir = os.path.join(new_dir, img)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    elif not debug:
        logger.critical("Results folder already exists [{}]".format(new_dir))
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    elif not debug:
        logger.critical("Image folder already exists [{}]".format(image_dir))

    return new_dir, file_name, image_dir


def get_measurements(data, fluo_data, fullwellimages,
                     allwellcoords, allbacteria, bacteria_lineage):
    """Takes the original image as well as the detected outputs and
    creates an instance of the measurements class containing the
    information for each bacteria

    Parameters
    ------
    data : ndarray
        Array containing all the original data (2D if just one frame)
    fluo_data : ndarray
        Array containing all the original fluorescent data (2D if
        just one frame)
    fullwellimages : list
        A list where each entry is a labelled image showing the
        detected wells
    allwellcoords : list
        A list where each entry is a dictionary containing wells and
        their coordinates
    allbacteria : list
        A list where each entry is a dictionary containing labelled
        images showing the detected bacteria
    bacteria_lineage : dictionary
        Keys are the 'actual' bacteria label and the value
        a reference to their lineage

    Returns
    ------
    measurements : Custom class instance
        Its attribute "bacteria" is a dictionary containing information on
        each individual bacteria
    fluorescence_backgrounds : list
        A list of the background fluorescence value for each timepoint
    """
    measurements = BacteriaData()
    fluorescence_backgrounds = {}
    for tpoint, (fullwells, bacteria, coords) in enumerate(zip(
            fullwellimages, allbacteria, allwellcoords)):
        # Create bacteria labelled image
        bacteriaim = np.zeros_like(fullwells)
        for welllabel in coords:
            bacteriaim[coords[welllabel]] = bacteria[welllabel]
        for region in regionprops(bacteriaim):
            well_label = fullwells[tuple(region.coords[0])]
            measurements.add_bac_data(
                region.label, bacteria_lineage, region, tpoint,
                well_label=well_label)
            if fluo_data is None:
                continue
            timepoint_fluo = [f[0] for f in fluo_data]
            fluo_values = [
                fluorescence_background(
                    fullwells,
                    bacteriaim,
                    fluo_im) for fluo_im in timepoint_fluo]
            fluorescence_backgrounds[tpoint] = [values[0]
                                                for values in fluo_values]
            measurements.measure_fluo(
                region, timepoint_fluo, fluo_values, tpoint)
    measurements.compile_results(max_tpoint=len(data))
    return measurements, fluorescence_backgrounds


def count_bacteria_in_wells(bacteria_labels):
    """
    For each well, return the number of bacteria it contains;
    NOTE: This works on lists of bacteria labels - use count_bacteria
          if working with full size labels of wells and bacteria

    Parameters:
    ------
    bacteria_labels : dictionary
        A dictionary containing labelled images showing the detected bacteria

    Returns
    ------
    numbac2 : list
        List containing the number of bacteria detected in each well
    """
    numbac2 = []
    for bac in bacteria_labels.values():
        numbac = ndi.label(bac)[1]
        numbac2.append(numbac)
    return numbac2
    # return [ndi.label(w>0)[1] for w in bacteria_labels]


def fluorescence_measurements(
    region,
    fluo_image,
    bkg,
):
    """Takes a region props area from a brightfield image and determines
    the mean fluorescence in a matching fluorescent image.
    Returns the average fluorescence, background deducted fluorescence
    (if its within 2 Standard errors of the background this is 0),
    and intergrated fluorescence

    Parameters
    ------
    region : Region props area
        A region props area with coordinates of the bacteria of interest
    fluo_image : ndarray (2D)
        The matching fluorescent image for the detected bacteria
    bkg : tuple
        tuple in the format (background fluorescence, background SEM) for the
        respective image
    """
    bkground = bkg[0]
    bg_2sem = bkg[0] + (bkg[1])
    fluo = np.mean(fluo_image[tuple(np.array(region.coords).T)])
    background_fluo = [0 if (fluo - bg_2sem) < 0 else (fluo - bkground)][0]
    integrated_fluorescence = background_fluo * region.area
    return fluo, background_fluo, integrated_fluorescence


def fluorescence_background(wells, bacteria, fluo_image):
    """
    Removes bacteria segmentation from well segmentation, then calculates
    the fluorescence_background as the mean of the remaining segmentation
    (in the fluorescece channel)
    """
    background_wells = (wells > 0) ^ (bacteria > 0)

    backg = fluo_image[background_wells]
    bkground = np.mean(backg)
    bg_stdev = np.std(backg)
    bg_sem = bg_stdev / (sqrt(len(backg)))
    return bkground, bg_sem
