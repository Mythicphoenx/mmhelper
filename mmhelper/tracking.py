# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:38:00 2016

@author: as624
"""
import tempfile
import os
import itertools
from skimage.feature import (
    match_descriptors,
    plot_matches,
    ORB,
    match_template,
)
import skimage.util as skutil
from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.measure import regionprops
import numpy as np
from mmhelper.utility import logger
import mmhelper.bacteria_tracking as bactrack
import matplotlib.pyplot as plt


# =========================================================
# Main "do everything" tracking function
# =========================================================

def run_tracking(data, allwellimages, allwellcoords, allbacteria, debug=False):
    """
    Takes in two frames of data, wells, and bacteria, and performs
    tracking

    Parameters
    ------
    data : list of ndarrays
        List of initial image that detection was run on
    allwellimages : list of ndarrays
        List of labelled images showing the detected wells
    allwellcoords : list of arrays
        Each entry contains a further list where each entry contains well coordinates
    allbacteria : list of arrays
        List of labelled images showing the detected bacteria
    debug     : Boolean
        Whether to add debugging outputs (default : False)

    Returns
    ------
    allwellimages : list of ndarrays
        List of labelled images showing the tracked wells
    allwellcoords : list of arrays
        Each entry contains a further list where each entry contains well coordinates
    allbacteria : list of arrays
        List of labelled images showing the tracked bacteria
    bacteria_lineage : dictionary
        A dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage
    """
    previous_wellimage = allwellimages[0]

    bacteria_lineage = {}

    for key, bacteria_image in allbacteria[0].items():
        bacteria_lineage.update({l.label: str(l.label)
                                 for l in regionprops(bacteria_image)})

    for tpoint in range(1, len(data)):
        # Frame shift
        frame_shift = frametracker(data[tpoint - 1], data[tpoint], debug=debug)
        logger.debug("\tFrame tracking registered a transform of: %s",
                     str(frame_shift))
        # Well tracking - find mappings from the previous image to the current
        # image
        if (len(np.unique(previous_wellimage)) > 1) and (
                len(np.unique(allwellimages[tpoint])) > 1):
            logger.debug("\tRunning well tracking...")
            wellimage_tracked, well_map = welltracking(
                previous_wellimage, allwellimages[tpoint], frame_shift)
            # Apply the mappings to the well dictionary and the bacteria
            # dictionary
            allwellcoords[tpoint] = {idkey: allwellcoords[tpoint][listindex]
                                     for listindex, idkey in well_map.items()}
            allbacteria[tpoint] = {idkey: allbacteria[tpoint][listindex]
                                   for listindex, idkey in well_map.items()}
            previous_wellimage = wellimage_tracked
            logger.debug("\tRunning bacterial lineage tracking...")
            # Do lineage tracking
            allbacteria[tpoint], bacteria_lineage = bacteria_tracking(
                allbacteria[tpoint - 1],
                allbacteria[tpoint],
                bacteria_lineage,
            )

        elif (len(np.unique(previous_wellimage)) == 1) and (len(np.unique(allwellimages[tpoint])) > 1):
            previous_wellimage = allwellimages[tpoint]
            if not bacteria_lineage:
                maxid = 0
            else:
                maxid = max(bacteria_lineage.keys())
            for key, bacteria_image in allbacteria[tpoint].items():
                newim = bacteria_image.copy()
                for i, bac in enumerate(regionprops(
                        bacteria_image), start=maxid + 1):
                    newim[bacteria_image == bac.label] = i
                    bacteria_lineage[i] = str(i)
                    maxid = i
                allbacteria[tpoint][key] = newim
        else:
            previous_wellimage = allwellimages[tpoint]
    logger.debug("\trun_tracking finished!")
    return allwellimages, allwellcoords, allbacteria, bacteria_lineage
    """
        allbacteria[t] = bacteria_tracking(allbacteria[t-1], allbacteria[t])
    """
    # At the moment there is no bacteria_lineage for the first loop so it
    # doesn't work as it uses the max(value) as 'smax' for each loop.
    # Either need to create the bacteria_lineage during the first
    # loop. But this is mainly an issue at the end of 'label_most_likely' function
    # as bacteria_lineage is mainly for lineage labelling. So perhaps easier to just
    # rethink it

# =========================================================
# Individual tracking functions
# =========================================================


def frametracker(*args, **kwargs):
    """
    Wrap the specific frametracker we're going to use
    #TODO: Add in switches re methods
    """

    return frametracker_template(*args, **kwargs)


def frametracker_template(
        img1,
        img2,
        border=100,
        debug=False,
):
    """
    Determine overall frame shift using template
    matching (normalized cross-correlation)

    Parameters
    ------
    img1 : ndarray (2D)
        The original image to analyse
    img2 : ndarray (2D)
        The new image to analyse
    border : int, optional
        Border to remove from image to create the template (default : 100)
    debug : Boolean
        Whether to add debugging outputs (default : False)

    Returns
    ------
    trans : array [-dy,-dx]
        The frame shift between wellim1 and wellim2
    """
    template = img1[border:-border, border:-border]
    xcorr = match_template(
        img2,       # where to look
        template,   # the template to look for
    )
    y_crds, x_crds = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    dy_ = y_crds - border
    dx_ = x_crds - border

    if debug:
        plt.figure()
        plt.imshow(img2, cmap='gray')
        rect = plt.Rectangle((x_crds, y_crds), template.shape[1], template.shape[0],
                             edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.savefig("DEBUG_FRAMETRACKING_LOCATION.jpg")
        plt.close()
        plt.figure()
        plt.imshow(template, cmap='gray')
        plt.savefig("DEBUG_FRAMETRACKING_TEMPLATE.jpg")
        plt.close()
        plt.figure()
        imcol = np.zeros(img1.shape + (3,))
        imcol[y_crds:y_crds + template.shape[0], x_crds:x_crds + template.shape[1], 0] = template
        imcol[..., 1] = img2
        imcol -= imcol.min()
        imcol /= imcol.max()
        plt.imshow(imcol)
        plt.savefig("DEBUG_FRAMETRACKING_OVERLAY.jpg")
        plt.close()

    return np.array([-dy_, -dx_])


def frametracker_keypoints(
        img1,
        img2,
        nk=50,
        fn=9,
        ft=0.001,
        hk=0.1,
        min_samples=10,
        xchange=300,
        ychange=30,
        debug=False,
):
    """
    Determine overall frame shift using ORB detection
    and keypoint matching

    Parameters
    ------
    img1 : ndarray (2D)
        The original image to analyse
    img2 : ndarray (2D)
        The new image to analyse
    nk : int, float
        The number of keypoints to use (default : 50)
    fn : int, optional
         fast_n from skimage.feature.ORB
         Minimum number of consecutive pixels out of 16 pixels on the circle
         that should all be either brighter or darker (default : 9)
    ft : float, optional
        fast_threshold from skimage.feature.ORB
        Threshold used to decide whether the pixels on the circle are
        brighter, darker or similar (default : 0.001)
    hk : float, optional
        harris_k from skimage.feature.ORB
        Sensitivity factor to separate corners from edges (default : 0.1)
    min_samples : int, optional
        min_samples from skimage.measure.ransac
        The minimum number of data points to fit a model to (default : 10)
    xchange : int, optional
        Maximum change along x-axis (default : 300)
    ychange : int, optional
        Maximum change along y-axis (default : 30)
    debug : Boolean
        Whether to add debugging outputs (default : False)

    Returns
    ------
    trans : array [-dy,-dx]
        The frame shift between wellim1 and wellim2
    """
    img1 = skutil.img_as_float(img1)
    img2 = skutil.img_as_float(img2)

    descriptor_extractor = ORB(
        n_keypoints=nk,
        fast_n=fn,
        fast_threshold=ft,
        harris_k=hk,
    )

    # determine keypoints and extract coordinates for first image
    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    # determine keypoints and extract coordinates for second image
    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # determine matching coordinates
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # create empty lists
    src = []
    dst = []

    for matches in matches12:
        # find index of the match from original image and image being compared
        a_index = matches[0]
        b_index = matches[1]
        # use the index from above to find the original coordinates from the
        # images
        a1orig = keypoints1[a_index]
        b1orig = keypoints2[b_index]
        # Create a list of the matched coordinates
        a1x = a1orig[1]
        a1y = a1orig[0]
        b1x = b1orig[1]
        b1y = b1orig[0]
        xch = abs(a1x - b1x)
        ych = abs(a1y - b1y)
        # Create a list of the matched coordinates
        if (xch < xchange) & (ych < ychange):
            src.append(a1orig)
            dst.append(b1orig)

    src = np.array(src)
    dst = np.array(dst)

    if debug:
        plt.figure()
        plt.imshow(img1, cmap='gray')
        plt.plot(keypoints1[:, 1], keypoints1[:, 0], '.r')
        plt.savefig("DEBUG_WELLTRACKING_frame_a_plus_keypoints.jpg")
        plt.close()
        plt.figure()
        plt.imshow(img2, cmap='gray')
        plt.plot(keypoints2[:, 1], keypoints2[:, 0], '.r')
        plt.savefig("DEBUG_WELLTRACKING_frame_b_plus_keypoints.jpg")
        plt.close()
        plt.figure()
        ax0 = plt.gca()
        plot_matches(ax0, img1, img2, keypoints1, keypoints2, matches12)
        plt.savefig("DEBUG_WELLTRACKING_matches.jpg")
        plt.close()

    try:
        # estimate affine transform model using all coordinates
        model = AffineTransform()
        model.estimate(src, dst)

        # robustly estimate affine transform model with RANSAC
        model_robust = ransac(
            (dst, src),
            AffineTransform, min_samples=min_samples, residual_threshold=2,
            max_trials=100)[0]
        trans = model_robust.translation
        return trans
    except BaseException:
        debugfolder = tempfile.mkdtemp()
        plt.figure()
        plt.imshow(img1, cmap='gray')
        plt.plot(keypoints1[:, 1], keypoints1[:, 0], '.r')
        plt.savefig(os.path.join(debugfolder, "img1_plus_keypoints.jpg"))
        plt.close()
        plt.figure()
        plt.imshow(img2, cmap='gray')
        plt.plot(keypoints2[:, 1], keypoints2[:, 0], '.r')
        plt.savefig(os.path.join(debugfolder, "img2_plus_keypoints.jpg"))
        plt.close()
        plt.figure()
        ax1 = plt.gca()
        plot_matches(ax1, img1, img2, keypoints1, keypoints2, matches12)
        plt.savefig(os.path.join(debugfolder, "matches.jpg"))
        plt.close()
        logger.critical("Failed to estimate affine transform!")
        logger.critical("Debugging images saved to ", debugfolder)


def welltracking(wellim1, wellim2, trans):
    """
    Takes two well images and a frame shift and re labels the wells correctly

    Parameters
    ------
    wellim1 : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    wellim2 : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    trans : array [-dy,-dx]
        The frame shift between wellim1 and wellim2

    Returns
    ------
    newwellim2 : ndarray (2D) of dtype int
        A labelled image showing the tracked wells
    well_map : dictionary
        Key is the current well label value is the previous well label
    """
    newwellim2 = np.zeros(wellim2.shape, dtype=wellim2.dtype)
    well_map = {}

    initial_regions = regionprops(wellim1)
    initial_centroids = np.array([r.centroid for r in initial_regions])
    initial_labels = np.array([r.label for r in initial_regions])
    detected_regions = regionprops(wellim2)
    detected_centroids = [r.centroid for r in detected_regions]
    detected_labels = np.array([r.label for r in detected_regions])
    detected_centroids = detected_centroids + trans
    # TODO: Can potentially just use 5 lines of source code from skimage.feature.match_descriptors
    # in case skimage change the functions
    matches = match_descriptors(
        initial_centroids,
        detected_centroids,
        metric='euclidean',
        cross_check=True)
    for match in matches:
        last_well = initial_labels[match[0]]
        current_well = detected_labels[match[1]]
        well_map[current_well] = last_well
        newwellim2[wellim2 == current_well] = last_well
    return newwellim2, well_map


def bacteria_tracking(last_wells, current_wells, bacteria_lineage):
    """
    Takes a dictionary of wells from the previous frame and a second
    dictionary with the corresponding wells for the new frame and
    determines the most likely way the bacteria within them have
    moved, died, divided then relabels them accordingly

    Parameters
    ------
    last_wells : Dictionary
        The previous timepoint. The key is the well coordinates and the value
        is a labelled image of detected bacteria
    current_wells : Dictionary
        The current timepoint. The key is the well coordinates and the value
        is a labelled image of detected bacteria
    bacteria_lineage : dictionary
        A dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage

    Returns
    ------
    out_wells : Dictionary
        The current timepoint. The key is the well coordinates and the value
        is a labelled image of tracked bacteria
    bacteria_lineage : dictionary
        Updated dictionary that links the physical unique label of a bacteria
        to one which shows information on its lineage
    """
    out_wells = {}
    for num, well in last_wells.items():
        if num not in current_wells.keys():
            continue
        new_well = current_wells[num]
        in_list = []
        #option_list = []
        for region in regionprops(well):
            # list the bacteria labels from the current frame
            in_list.append(region.label)
        # create a list of all the possible combinations
        logger.debug("Creating combination of %d items repeated %d times" % (
            len(in_list), len(regionprops(new_well))))
        options = itertools.combinations_with_replacement(
            in_list, len(regionprops(new_well)))

        if not in_list:
            if len(regionprops(new_well)) > 0:
                # if there is now a new bacteria we don't want to ignore as it may have just
                # been missed in previous frame
                smax = max(bacteria_lineage, key=int)
                newwell = np.zeros(new_well.shape, dtype=new_well.dtype)
                for new_bac in regionprops(new_well):
                    # so lets give each "new" bacteria a new label
                    smax += 1
                    newwell[new_well == new_bac.label] = smax
                    bacteria_lineage[smax] = str(smax)
            elif not next(options, None):
                # if the out well is also empty then there is nothing to track
                # and simply return an empty well
                newwell = np.zeros(new_well.shape, dtype=new_well.dtype)
        else:  # determine probabilities and label matching/new bacteria
            #options_dict[n] = [in_list,option_list]
            change_options = bactrack.find_changes(
                in_list, options, well, new_well)
            best_option = None
            best_prob = 0
            for option, probs in change_options:
                probs_ = bactrack.find_probs(probs)
                if probs_ > best_prob:
                    best_prob = probs_
                    best_option = option
            newwell, bacteria_lineage = bactrack.label_most_likely(
                best_option, new_well, bacteria_lineage)
        out_wells[num] = newwell

    return out_wells, bacteria_lineage
