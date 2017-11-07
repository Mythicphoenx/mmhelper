# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:38:00 2016

@author: as624
"""

from skimage.feature import (
    match_descriptors,
    plot_matches,
    ORB,
    match_template,
    )
import skimage.util as skutil
from skimage.measure import ransac
import numpy as np
from skimage.transform import AffineTransform
from skimage.measure import regionprops
from momanalysis.utility import logger
import matplotlib.pyplot as plt
import tempfile
import os
import momanalysis.bacteria_tracking as bactrack
import itertools


def frametracker(*args, **kwargs):
    """
    Wrap the specific frametracker we're going to use
    """
    return frametracker_template(*args, **kwargs)

def frametracker_template(
        img1,
        img2,
        border = 100,
        debug=False,
        ):
    """
    Determine overall frame shift using template
    matching (normalized cross-correlation)
    """
    template = img1[border:-border, border:-border]
    xcorr = match_template(
        img2,       # where to look
        template,   # the template to look for
        )
    y, x = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    dy = y - border
    dx = x - border

    if debug:
        plt.figure()
        plt.imshow(img2, cmap='gray')
        rect = plt.Rectangle((x,y), template.shape[1], template.shape[0],
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
        imcol[y:y+template.shape[0],x:x+template.shape[1], 0] = template
        imcol[...,1] = img2
        imcol -= imcol.min()
        imcol /= imcol.max()
        plt.imshow(imcol)
        plt.savefig("DEBUG_FRAMETRACKING_OVERLAY.jpg")
        plt.close()

    return np.array([-dy,-dx])

def frametracker_keypoints(
        img1,
        img2,
        nk = 50,
        fn = 9,
        ft = 0.001,
        hk = 0.1,
        min_samples=10,
        xchange = 300,
        ychange = 30,
        debug=False,
        ):
    """
    Determine overall frame shift using ORB detection
    and keypoint matching
    """
    img1 = skutil.img_as_float(img1)
    img2 = skutil.img_as_float(img2)


    descriptor_extractor = ORB(
        n_keypoints= nk,
        fast_n= fn,
        fast_threshold= ft,
        harris_k= hk,
    )

    #determine keypoints and extract coordinates for first image
    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    #determine keypoints and extract coordinates for second image
    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    #determine matching coordinates
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    #create empty lists
    src = []
    dst = []

    for matches in matches12:
        #find index of the match from original image and image being compared
        a = matches[0]
        b = matches[1]
        #use the index from above to find the original coordinates from the images
        a1 = keypoints1[a]
        b1 = keypoints2[b]
        #Create a list of the matched coordinates
        a1x = a1[1]
        a1y = a1[0]
        b1x = b1[1]
        b1y = b1[0]
        xc = abs(a1x - b1x)
        yc = abs(a1y - b1y)
        #Create a list of the matched coordinates
        if (xc < xchange) & (yc < ychange):
            src.append(a1)
            dst.append(b1)

    src = np.array(src)
    dst = np.array(dst)

    if debug:
        plt.figure()
        plt.imshow(img1, cmap='gray')
        plt.plot(keypoints1[:,1], keypoints1[:,0], '.r')
        plt.savefig("DEBUG_WELLTRACKING_frame_a_plus_keypoints.jpg")
        plt.close()
        plt.figure()
        plt.imshow(img2, cmap='gray')
        plt.plot(keypoints2[:,1], keypoints2[:,0], '.r')
        plt.savefig("DEBUG_WELLTRACKING_frame_b_plus_keypoints.jpg")
        plt.close()
        plt.figure()
        ax = plt.gca()
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
        plt.savefig("DEBUG_WELLTRACKING_matches.jpg")
        plt.close()

    try:
        # estimate affine transform model using all coordinates
        model = AffineTransform()
        model.estimate(src, dst)

        # robustly estimate affine transform model with RANSAC
        model_robust, inliers = ransac((dst, src), AffineTransform, min_samples=min_samples,
                                       residual_threshold=2, max_trials=100)
        trans = model_robust.translation
        return trans
    except:
        debugfolder = tempfile.mkdtemp()
        plt.figure()
        plt.imshow(img1, cmap='gray')
        plt.plot(keypoints1[:,1], keypoints1[:,0], '.r')
        plt.savefig(os.path.join(debugfolder, "img1_plus_keypoints.jpg"))
        plt.close()
        plt.figure()
        plt.imshow(img2, cmap='gray')
        plt.plot(keypoints2[:,1], keypoints2[:,0], '.r')
        plt.savefig(os.path.join(debugfolder, "img2_plus_keypoints.jpg"))
        plt.close()
        plt.figure()
        ax = plt.gca()
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
        plt.savefig(os.path.join(debugfolder, "matches.jpg"))
        plt.close()
        logger.critical("Failed to estimate affine transform!")
        logger.critical("Debugging images saved to ", debugfolder)

def welltracking(wellim1, wellim2, trans, pixls = 11):

    newwellim2 = np.zeros(wellim2.shape, dtype=wellim2.dtype)
    newwellim1 = np.zeros(wellim1.shape, dtype=wellim2.dtype)
    for i,r in enumerate(regionprops(wellim1), start=1):
        #find coordinates of the regions and select first coordinates
        co = r.coords
        p = co[0]
        #subtract the frame shift
        init = p - trans
        #determine the "new" coordinates
        xp1 = init[1]
        yp1 = init[0]
        for j, r2 in enumerate(regionprops(wellim2), start=1):
            #find coordinates of the 2nd image regions and select first coordinates
            co2 = r2.coords
            p2 = co2[0]
            xp2 = p2[1]
            yp2 = p2[0]
            #subtract "new" coordinates away from 2nd image coordinates
            revshiftx = abs(xp2-xp1)
            revshifty = abs(yp2-yp1)
            #if they are within one well width (11 pixels) then assign them to the same label
            if (revshiftx < pixls) & (revshifty < pixls):
                newwellim2[wellim2 == j] = r.label
                """work out the change in wells for bacteria tracking"""
                #newwellim1[wellim1 == i] = r.label
            #return new matching labelled well images
    """
    plt.figure()
    plt.imshow(wellim1)
    plt.figure()
    plt.imshow(wellim2)
    plt.figure()
    plt.imshow(newwellim2)
    plt.show()
    """
    return newwellim2
    
def bacteria_tracking(last_wells, current_wells, label_dict_string):

    """start of tracking"""

    options_dict = {}
    well_dict = {}
    bac_info = {}
    out_wells = {}
    for n, well in last_wells.items():
        for n2, new_well in current_wells.items():
            if n == n2: #if well numbers match then we can try and track the bacteria
                in_list = []    
                option_list = []
                for i, region in enumerate(regionprops(well)):
                    in_list.append(region.label) #list the bacteria labels from the current frame
                #create a list of all the possible combinations
                output_options = list(itertools.product(in_list, repeat=(len(regionprops(new_well))))) 
                for option in output_options:
                #the bacteria have to stay in the same order so we can filter out those that aren't in order
                    if sorted(option) == list(option): 
                        if len(option) > 0: #if no options it is an empty list - filter this out
                            option_list.append(option)
                if not in_list:                
                    if len(regionprops(new_well)) > 0: 
                        #if there is now a new bacteria we don't want to ignore as it may have just
                        #been missed in previous frame 
                        smax = max(label_dict_string, key=int)
                        newwell = np.zeros(new_well.shape, dtype=new_well.dtype)
                        for k, new_bac in enumerate(regionprops(new_well)):
                            #so lets give each "new" bacteria a new label
                            smax+=1
                            newwell[new_well==new_bac.label] = smax
                            label_dict_string[smax] = str(smax)
                    elif not option_list: #
                    #if the out well is also empty then there is nothing to track and simply return an empty well
                        newwell = np.zeros(new_well.shape, dtype=new_well.dtype)
                else: #determine probabilities and label matching/new bacteria
                    options_dict[n] = [in_list,option_list] 
                    change_options = bactrack.find_changes(in_list,option_list, well, new_well)
                    probs_ = bactrack.find_probs(change_options)
                    newwell, label_dict_string = bactrack.label_most_likely(probs_, new_well, label_dict_string)
                out_wells[n] = newwell
    
    return out_wells, label_dict_string

