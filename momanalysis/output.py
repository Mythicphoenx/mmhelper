# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 09:59:13 2017

@author: as624
"""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

def output_csv(measurementsArray, dir_name, filename, timepoint,bg=None, fl=False):
    """takes the measurements of the bacteria and writes it to a csv"""
    outfilename = os.path.join(
        dir_name,
        '%s_timepoint_%s.csv' %(filename, timepoint)
    )

    with open(outfilename, "w", newline='') as f:
        """need to work out how we will do it for multiple frames"""
        writer = csv.writer(f)
        if fl is False:
            writer.writerow(["Bacteria Number", "Area", "Length", "Width"])
        else:
            writer.writerow(["Background", bg, "","","","",""])
            writer.writerow(["Bacteria Number", "Area", "Length", "Width", "fluorescence", "fluo - background", "fluo sum - background"])
        writer.writerows(measurementsArray)

def output_figures(d, timeindex, channel, wells, profiles, bacteria, coords, wellimg, ridges, image_dir, label_dict_string):
    #FIXME: MISSING DOCSTRING

    #Generate images
    figs = []
    #C
    plt.figure() #to fix platform specific bug with matplotlib initialisation (same as line above "#plt.switch_backend('qt5agg')")
    """if profiles:
        figs.append(plt.figure(figsize=(16,12)))
        plt.imshow(np.hstack(profiles.values()))
        plt.colorbar()
    """
    figs.append(plt.figure(figsize=(16,12)))
    plt.imshow(d, cmap='gray')
    plt.contour(wellimg, levels=[0.5], colors=['y'])
    for lab_well in range(1, wellimg.max()+1):
        bw = wellimg == lab_well
        if not np.any(bw):
            continue
        pos0 = bw.nonzero()
        pos = (np.min(pos0[0]), np.max(pos0[1]))
        plt.text(pos[1], pos[0], "%d"%lab_well, color="y")

    # FIXME: LAZY WAY TO DO THIS FOR NOW
    segfull = np.zeros(wellimg.shape, dtype="int16")
    for s, c in zip(bacteria, coords):
        segfull[c] = bacteria[s]

    Nbac = segfull.max()
    for lab_bac in range(1, Nbac+1):
        bw = segfull == lab_bac
        pos0 = bw.nonzero()
        if len(pos0[0]) == 0 or len(pos0[1]) == 0:
            continue
        lab_string = label_dict_string[lab_bac]
        pos = (np.min(pos0[0]), np.max(pos0[1]))
        col = plt.cm.gist_rainbow((lab_bac/9.1)%1)
        plt.contour( bw, levels=[0.5], colors=[col])
        plt.text(pos[1], pos[0], "%s"%lab_string, color=col)


    figs.append(plt.figure())
    plt.imshow(d, cmap='gray')
    plt.imshow((channel>0) + 2*(wells>0) , alpha=0.4)
    figs.append(plt.figure())
    plt.imshow(ridges)

    for i, f in enumerate(figs):
        f.savefig(os.path.join(image_dir, "wells_%0.6d_%0.6d.jpg"%(timeindex, i)))
        plt.close(f)

def final_output(dir_name, filename, fluo, fluoresc, tmax):
    #FIXME: MISSING DOCSTRING
    timepoints = list(range(0,tmax)) #create a list of timepoints
    rows_csvs = []
    if fluo is None and fluoresc is False:
        no_data = ["", "", "", ""] #if there is no data at a given timepoint this will be written
        time_row = ["",""]
    else:
        no_data = ["", "", "", "", "", "", ""]
        time_row = ["","","","",""]
    time_row2 = []
    for t in timepoints:
        timestamp = ["Time =", t]
        time_row2.extend(timestamp)
        time_row2.extend(time_row)
        individual_csv_filename = os.path.join(
            dir_name,
            '%s_timepoint_%s.csv' %(filename, t)
        )
        try:
            with open(individual_csv_filename, "r",newline='') as f:
                reader2 = list(csv.reader(f))
                rows_csvs.append(len(reader2)) #find the length of all the individual timepoint csvs
        except:
            rows_csvs.append(int(0))
    max_rows = max(rows_csvs) #find the maximum length

    combined_csv_filename = os.path.join(
        dir_name,
        'Combined_Output.csv',
    )
    with open(combined_csv_filename, "w", newline='') as o:
        writer = csv.writer(o)
        writer.writerow(time_row2)
        for r in range(0,max_rows):
            current_row = []
            for t in timepoints:
                individual_csv_filename = os.path.join(
                    dir_name,
                    '%s_timepoint_%s.csv' %(filename, t)
                )
                try:
                    with open(individual_csv_filename, "r",newline='') as f:
                        num_rows = rows_csvs[t] #find the number of rows for this timepoint
                        if r < num_rows: #if the row exists extract it
                            temp_row = next(itertools.islice(csv.reader(f), r, None))
                            current_row.extend(temp_row) #create one long list of values
                        else: #if row is empty
                            current_row.extend(no_data)
                except:
                    current_row.extend(no_data)
            writer.writerow(current_row) #write the new row
