# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:17:59 2016

@author: as624
"""
import csv
import scipy.ndimage as ndi
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from skimage.measure import regionprops
import os
import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime
from math import sqrt


def find_input_filename(fname, out = None, batch=False, im_area=None, test=False):
    """Takes the filename input and splits it into simply the filename
    which is then used to name the csv later"""
    dir_name = os.path.split(os.path.realpath(fname))[0]

    timestamp = strftime("%d%m%Y_%H%M%S", gmtime())

    if out is None:
        file_name = os.path.splitext(os.path.basename(fname))[0]
        if batch == True:
            new_folder = "Area"+ im_area + "_results_" + timestamp
        else:
            new_folder = file_name + "_results_" + timestamp
        new_dir = os.path.join(dir_name, new_folder)
    else:
        file_name = out
        new_folder = file_name + "_" + timestamp
        new_dir = os.path.join(dir_name, new_folder)

    img = "output_images"
    image_dir = os.path.join(new_dir, img)
    if test is False:
        os.makedirs(new_dir)
        os.makedirs(image_dir)

    return new_dir, file_name, image_dir

def bacteria_measurements(label_dict_string,segfull, fluo_values=None):
    """
    Takes a list of labelled images representing individual wells
    and returns a string containing the area and length of bacteria
    """

    measurementsArray = []
    lblvalues = label_dict_string.values()
    original_lbls = []
    list1 = []
    for lbl in lblvalues:
        newlbl = int(float(lbl.split('_')[0]))
        original_lbls.append(newlbl)
    Nbac = segfull.max()
    for bac in range(1, Nbac+1):
        region = regionprops(segfull)
        for r in (r for r in region if r.label == bac):
            A = r.area
            L = r.major_axis_length
            W = r.minor_axis_length
            string_lbl = label_dict_string[bac]
            list1 = [string_lbl,A, L, W]
            if fluo_values is not None:
                fluo, fluo_bk = fluo_values[r.label]
                list1 = [string_lbl,A, L, W, fluo, fluo_bk, (fluo_bk*A)]
            continue
        if not list1 and bac in original_lbls:
            string_lbl = str(bac)
            list1 = [string_lbl,"-","-","-"]
            if fluo_values is not None:
                list1 = [string_lbl,"-","-","-","-","-","-"]
        elif not list1:
            string_lbl = str(bac)
            list1 = [string_lbl,"-","-","-"]
            if fluo_values is not None:
                list1 = [string_lbl,"-","-","-","-","-","-"]
        measurementsArray.append(list1)
        list1=[]

    return measurementsArray

def count_bacteria_in_wells(bacteria_labels):
    """
    For each well, return the number of bacteria it contains;
    NOTE: This works on lists of bacteria labels - use count_bacteria
          if working with full size labels of wells and bacteria
    """
    numbac2=[]
    for n, bac in bacteria_labels.items():
        numbac=ndi.label(bac)[1]
        numbac2.append(numbac)
    return numbac2
    #return [ndi.label(w>0)[1] for w in bacteria_labels]

def fluorescence_measurements(labelled_image, bkground, bg_sem, fluo_image):
    fluo_values = {}
    bg_2sem = bkground + (2*bg_sem)
    for region in regionprops(labelled_image):
        fluo_1 = []
        fluo = np.mean(fluo_image[labelled_image==region.label])
        if fluo > bg_2sem:
            fluobg = fluo-bkground
        else:
            fluobg = 0
        fluo_1 = [fluo, fluobg]
        fluo_values[region.label] = fluo_1
    return fluo_values

def fluorescence_background(wells,segfull, fluo_image):
    allwells = wells
    allwells[allwells>0] = 1
    allbac = np.copy(segfull)
    allbac[allbac>0] = 1
    background_wells = allwells-allbac
    
    backg = fluo_image[background_wells>0]
    bkground = np.mean(backg)
    bg_stdev = np.std(backg)
    bg_sem = bg_stdev/(sqrt(len(backg)))
    return bkground, bg_sem



