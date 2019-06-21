# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 09:59:13 2017

@author: as624
"""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def output_detection_figures(
        image, wells, bacteria, timeindex, output_dir):
    """
    Produces and saves figures showing the output from the detection

    Parameters
    ------
    image : ndarray (2D)
        The initial image that detection was run on
    wells : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    bacteria : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    timeindex : int
        The timepoint that has been analysed
    output_dir : str (path)
        Where to save the images
    """
    # For detection figures, labels not needed (I think)?
    plt.figure(figsize=(16, 12))
    plt.imshow(image, cmap='gray')
    plt.contour(wells > 0, levels=[0.5], colors=['y'])
    #plt.contour(channel>0, levels=[0.5], colors=['r'])
    for lab_bac in range(1, bacteria.max() + 1):
        col = plt.cm.gist_rainbow((lab_bac / 9.1) % 1)
        plt.contour(bacteria == lab_bac, levels=[0.5], colors=[col])
    plt.savefig(os.path.join(
        output_dir, "detection_frame_{:06d}".format(timeindex)))
    plt.close()


def output_tracking_figures(
        data,
        fullwellimages,
        wellcoords,
        allbacteria,
        output_dir,
        bacteria_lineage):
    """
    Produces and saves figures showing the output after tracking

    Parameters
    ------
    data : list of ndarrays
        List of initial image that detection was run on
    fullwellimages : list of ndarrays
        List of labelled images showing the detected wells
    wellcoords : list of arrays
        Each entry contains a further list where each entry contains well coordinates
    allbacteria : list of arrays
        List of labelled images showing the detected bacteria
    output_dir : str (path)
        Where to save the images
    bacteria_lineage : dictionary
        A dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage
    """
    for tpoint, (image, fullwells, bacteria, coords) in enumerate(
            zip(data, fullwellimages, allbacteria, wellcoords)):
        # For detection figures, labels not needed (I think)?
        plt.figure(figsize=(16, 12))
        plt.imshow(image, cmap='gray')
        if len(np.unique(fullwells)) == 1:
            plt.savefig(os.path.join(
                output_dir, "tracking_frame_{:06d}".format(tpoint)))
            plt.close()
            continue
        plt.contour(fullwells > 0, levels=[0.5], colors=['y'])
        bacteriaim = np.zeros_like(fullwells)
        for welllabel in coords:
            bacteriaim[coords[welllabel]] = bacteria[welllabel]
            # Add in well labels top left(?) of well contour
            #bw = fullwells == welllabel
            # if not np.any(bw):
            #    continue
            #pos0 = bw.nonzero()
            pos = (np.min(coords[welllabel][0]), np.max(coords[welllabel][1]))
            plt.text(pos[1], pos[0], "%d" % welllabel, color="y")

        for lab_bac in range(1, bacteriaim.max() + 1):
            col = plt.cm.gist_rainbow((lab_bac / 9.1) % 1)
            bw0 = bacteriaim == lab_bac
            if not np.any(bw0):
                continue
            plt.contour(bw0, levels=[0.5], colors=[col])
            pos0 = bw0.nonzero()
            if len(pos0[0]) == 0 or len(pos0[1]) == 0:
                continue
            #lab_string = label_dict_string[lab_bac]
            pos = (np.min(pos0[0]), np.max(pos0[1]))
            plt.text(pos[1], pos[0], str(bacteria_lineage[lab_bac]), color=col)
        plt.savefig(os.path.join(
            output_dir, "tracking_frame_{:06d}".format(tpoint)))
        plt.close()


def final_output(measurements, output_dir):
    """outputs a final csv with information on the bacteria detected

    Parameters
    ------
    measurements : Custom class instance
        Its attribute "bacteria" is a dictionary containing information on
        each individual bacteria
    output_dir : str (path)
        Where to write the csv
    """
    output_csv_file = os.path.join(output_dir, 'Results.csv')
    with open(output_csv_file, "w", newline='') as file0:
        writer = csv.writer(file0)
        for numbac, (bac) in enumerate(measurements.bacteria.values()):
            if numbac == 0:
                writer.writerow(bac.headings_line)
            writer.writerow(bac.measurements_output)
