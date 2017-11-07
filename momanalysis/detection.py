#
# FILE        : detection.py
# CREATED     : 27/09/16 13:08:52
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Detection functions
#

import math
import skimage.filters as skfilt
import momanalysis.skimage_future as skfuture
import skimage.morphology as skmorph
import skimage.measure as skmeas
import numpy as np
import scipy.spatial as scispat
import scipy.ndimage as ndi
from momanalysis.utility import logger
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.filters import sobel
from skimage.measure import regionprops
import skimage.exposure as exp
from statistics import median
from scipy.interpolate import InterpolatedUnivariateSpline

def detect_channel(image,
        scale_range=[5.0, 10.0],
        bright=False,
        minwidth=5,
        ):
    """
    Detect channel which is characterised by a long dark/bright band within the image.

    Inputs
    ------

    image - the image to analyse
    scale_range - scale range for channel ridge detection
    """
    # Use ridge filter to detect... ridges.
    # NOTE: frangi detects DARK ridges by default
    if bright:
        filt = skfuture.frangi(
            image.max()-image,
            scale_range=scale_range,
            black_ridges=True,
            )
    else:
        filt = skfuture.frangi(
            image,
            scale_range=scale_range,
            black_ridges=True,
            )

    #ridges = filt > skfilt.threshold_otsu(filt)
    ridges = filt > skfuture.threshold_li(filt)
    # Clean up to separate any branches
    if minwidth:
        ridges = skmorph.opening(ridges, selem=skmorph.disk(minwidth))

    # Pull out the largest region
    lbl2,N2 = skmeas.label(ridges, return_num=True)
    if N2 == 0:
        logger.warning("Unable to detect ridge!")
        return np.zeros(lbl2.shape, dtype=bool)
    lchannel = np.argmax([ (lbl2==l).sum() for l in range(1,N2+1)])+1
    channel = lbl2==lchannel

    return channel


def detect_wells(
        image,
        scale_range=[2.0,6.0],
        maxd=300,
        mind=100,
        maxperp=22,
        minwidth=3,
        ):
    """
    Detect the wells using a ridge filter, and the classifying
    based on their periodic property
    """
    # NOTE: frangi detects DARK ridges
    filt = skfuture.frangi(
        image,
        scale_range=scale_range,
        )

    ridges = filt > skfilt.threshold_li(filt)

    # Classify using simple measures
    lbl,N = skmeas.label(ridges, return_num=True)
    lblgood = np.zeros(lbl.shape, dtype='int16')

    smax = 0
    for region in regionprops(lbl):
        #print(region.label, region.area)
        if region.area > 2500:
            smax+=1
            lblgood[lbl==region.label] = smax
    lblgood = lblgood > 0
    lbl_filled = ndi.morphology.binary_fill_holes(lblgood)
    wells_middle = lbl_filled - lblgood
    wells_middle = ndi.morphology.binary_closing(wells_middle, iterations=5)

    # Classify using simple measures
    lbl_wells,N = skmeas.label(wells_middle, return_num=True)
    #props = skmeas.regionprops(lbl)
    bwnew = np.zeros(lbl_wells.shape, "bool")
    coms = []
    ngood = 0
    lbl_final= np.zeros(lbl_wells.shape, dtype='int16')
    """
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(filt)
    plt.figure()
    plt.imshow(ridges)
    plt.figure()
    plt.imshow(lblgood)
    plt.figure()
    plt.imshow(wells_middle)
    plt.show()
    """
    for l in range(1, N+1):
        bw = lbl_wells == l
        # Size
        area = bw.sum()
        if area > (maxd*maxd/2) or area < mind:
            continue
        perim = bw ^ ndi.binary_erosion(bw)
        pts = np.array(perim.nonzero(), dtype=float).T
        # Length = max separation between any two points in each region
        maxdist = scispat.distance.cdist(pts, pts)
        dnow = maxdist.max()
        imax = (maxdist == dnow).nonzero()
        vmax = pts[imax[0][0]] - pts[imax[1][0]]
        vmax /= np.sqrt(np.sum(vmax**2))
        vperp = np.array([-1*vmax[1], vmax[0]])
        pperp = [np.dot(vperp, p) for p in pts]
        distperp = np.max(pperp) - np.min(pperp)
        if (dnow> mind) & (dnow < maxd) & (distperp < maxperp):
            #maxperp is causing some "well loss" on detection - values occassionally
            #higher than 20 but less than 22 (2 well widths) so will change to 22
            #May be due to knew detection method so need to revise!!!
            bwnew[bw] = True
            ngood +=1
            lbl_final[bw] = ngood
    if ngood < 2:
        logger.warning("Less than 2 wells detected, this function has probably failed")
        
    ## Lastly let's see if we can detect anomolous things
    #propsgood = skmeas.regionprops(lblgood)
    #good = np.zeros(lblgood.shape)
    #bad = np.zeros(lblgood.shape)
    #mus = np.array([p.moments_hu for p in propsgood])
    #mus2 = mus - mus.mean(axis=0)

    return lbl_final, ridges # For debugging


def extract_well_profiles(
        image,
        channel,
        wells,
        wellwidth = 11,
        debug=False,
        min_well_sep_factor=2.5,
        max_well_sep_factor=6,
        tracked = False):
    """
    * Extend wells down to channel and up to maximal height
    * Add in missing wells (using interpolation of positions)
    * Extract well profiles

    TODO: Add full docstring with returns etc
    """

    #-------------------
    # A bit of pre-processing on the wells
    propsgood = skmeas.regionprops(wells)
    # Make sure we have at least 2 wells... at least this many for the
    # interpolation...
    # TODO: Decide if we need more
    if len(propsgood) < 2:
        logger.error("Less than 2 wells detected; unable to extract well profiles")
        blank_wellimage = np.zeros(image.shape, dtype="uint16")
        return [], blank_wellimage, []

    coms, oris, uvec_para, uvec_perp = _get_wells_and_unit_vectors(
        propsgood,
        debug=debug,
    )

    normseps, posperp_sorted = _get_well_spacing_and_separations(
        coms,
        uvec_perp,
        wellwidth,
        debug=debug,
        min_well_sep_factor=min_well_sep_factor,
        max_well_sep_factor=max_well_sep_factor,
    )

    images, wellimage, coords = interpolate_positions_and_extract_profiles(
        normseps,
        posperp_sorted,
        propsgood,
        uvec_para,
        uvec_perp,
        wellwidth,
        image,
        debug=debug,
        tracked=tracked
    )

    return images, wellimage, coords


#------------------------------
# Subfunctions for extract_well_profiles
#------------------------------

def get_channel_orientation_and_line(channel):
    """
    Determine channel orientation and get line coordinates

    Returns
    -------
    orientation : float
        Angle between x-axis and major axis of ellipse that
        has the same second-moments as the channel.
        Ranges from -pi/2 to pi/2 in counter-clockwise direction
    line : (2,2) tuple
        ((x1,y1), (x2,y2)) for the line end-points;
        uses the major axis length to determine the length
    """
    channelprop = skmeas.regionprops(1*channel)[0]
    yc0,xc0 = channelprop.centroid
    orientation = channelprop.orientation
    xc1 = xc0 + math.cos(orientation) * 0.5 * channelprop.major_axis_length
    yc1 = yc0 + math.sin(orientation) * 0.5 * channelprop.major_axis_length
    xc2 = xc0 - math.cos(orientation) * 0.5 * channelprop.major_axis_length
    yc2 = yc0 - math.sin(orientation) * 0.5 * channelprop.major_axis_length
    return orientation, ((xc1, yc1), (xc2, yc2))


def _get_wells_and_unit_vectors(props, debug=False):
    # In perpendicular direction, get positions,
    # and in parallel direction, get min and max coords
    coms = []
    oris = []
    lens = []

    for i,prop in enumerate(props):
        # Convert to simple line using
        # centroid and orientation information
        y0,x0 = prop.centroid
        orientation = prop.orientation
        # Make sure angle is in range 0, pi
        ori2 = orientation % np.pi

        # Find maximal well length (separation of points in binary image)
        dist = scispat.distance.cdist(prop.coords, prop.coords)
        length = dist.max()
        x1 = x0 + math.cos(ori2) * 0.5 * length
        y1 = y0 - math.sin(ori2) * 0.5 * length
        x2 = x0 - math.cos(ori2) * 0.5 * length
        y2 = y0 + math.sin(ori2) * 0.5 * length
        coms.append(np.array((x0,y0)))
        oris.append(ori2)
        lens.append(length)
    coms = np.array(coms)
    oris = np.array(oris)
    # Get the median well orientation
    ori = np.median(oris)
    if debug:
        print("\nWell orientations")
        print(oris)
        print("median orientation:", ori)
    # Generate parallel and perpendicular unit vectors.
    uvec_para = np.array([math.cos(ori), -math.sin(ori)])
    uvec_perp = np.array([math.sin(ori), math.cos(ori)])
    if debug:
        print("Parallel vector", uvec_para)
        print("Perpendicular vector", uvec_perp)
    # filter out any positions that shouldn't be here
    pospara = [ np.dot(uvec_para, p) for p in coms ]
    pospara_med = np.median(pospara)
    pospara_dif = np.abs(np.array(pospara)-pospara_med)
    pospara_bad = pospara_dif > (0.5*np.median(lens))
    coms = coms[~pospara_bad]
    oris = oris[~pospara_bad]
    if debug:
        print("After filtering bad CoMs")
        print("Orientations")
        print(oris)
    ori = np.median(oris)
    # Generate parallel and perpendicular unit vectors.
    uvec_para = np.array([math.cos(ori), -math.sin(ori)])
    uvec_perp = np.array([math.sin(ori), math.cos(ori)])
    if debug:
        print("Parallel vector", uvec_para)
        print("Perpendicular vector", uvec_perp)
    return coms, oris, uvec_para, uvec_perp


def _get_well_spacing_and_separations(
        coms,
        uvec_perp,
        wellwidth,
        min_well_sep_factor=2.5,
        max_well_sep_factor=6,
        debug=False):
    # Let's get the spacing
    posperp = [ np.dot(uvec_perp, p) for p in coms ]
    if debug:
        print("Perpendicular positions")
        print(posperp)

    posperp_sorted = np.sort(posperp)
    seps = np.diff( posperp_sorted )
    seps2 = np.diff(posperp_sorted) #create a duplicate for determination of median
    # Remove any separations that are less than the well-width,
    # these are likely due to fragmentation of a well
    goodseps = seps > wellwidth
    seps = seps[goodseps]
    posperp_sorted = posperp_sorted[np.concatenate([[True,], goodseps])]
    seps_sorted = np.sort(seps)

    #use only "realistic" separations for determining median
    #I.e. bigger than one well width but less than 6
    goodseps2 = (
        (seps2 >= (wellwidth*min_well_sep_factor))
        & (seps2 <(wellwidth*max_well_sep_factor)))
    seps2 = seps2[goodseps2]
    seps2_sorted = np.sort(seps2)

    if debug:
        print("\nSaving SEPARATIONS.jpg...")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(seps)
        plt.savefig('SEPARATIONS.jpg')
        plt.close()

    # Median well separation...
    # wellsep0 = np.median(seps)
    # It's possible that we have a few gaps; if the most frequent gap
    # is 1 or more, then the mean and median will be unusable
    # Check the median of the lowest few seps
    # and compare with the "wellsep"

    if len(seps2_sorted) > 5:
        lowerseps = seps2_sorted
    else:
        # Too few seps for stats, take lowest and hope for best!
        lowerseps = seps2_sorted[0]
    # 2017/02/01 13:20:06 (GMT):[JM] Surely this should be median...
    #lowersep = np.mean(lowerseps)
    lowersep = np.median(lowerseps) #determine median from "realistic values"

    normseps = seps/lowersep #normalise separations to the median
    normseps_temp = normseps.round() #round to the nearest whole number
    normseps_temp = abs(normseps_temp - normseps) #subtract actual normalised values
    wrong_seps = []
    for i, j in enumerate(normseps_temp):
        if j > 0.1: #"correct" values will be close to whole numbers when divided by median
            wrong_seps.append(i) #append the index value of "incorrect" values
    posperp_sorted = np.delete(posperp_sorted, wrong_seps) #delete these "incorrect" values
    seps3 = np.diff( posperp_sorted )
    normseps = seps3/lowersep
    normseps_sorted = np.sort(normseps)

    if len(seps_sorted) > 5:
        #2017/02/01 13:21:26 (GMT)[JM] As above, should be median here??
        #lowernormed = np.mean(normseps_sorted[:5])
        lowernormed = np.median(normseps_sorted[:5])
    else:
        lowernormed = normseps_sorted[0]
    if debug:
        print("normseps_sorted")
        print(normseps_sorted)
        print("LOWERNORMED")
        print(lowernormed)

    if (0.95 < lowernormed) and (1.05 > lowernormed):
        # We're pretty sure we've got a good estimator
        # here
        pass
    else:
        raise Exception("Wells not extracted properly - mean normalized separation\n"
            + "not close to 1: %f [lowersep = %f, lowerseps = %s]"%(lowernormed, lowersep, str(lowerseps)))

    if debug:
        print("NORMSEPS")
        print(normseps)

    return normseps, posperp_sorted


def interpolate_positions_and_extract_profiles(
        normseps,
        posperp_sorted,
        props,
        uvec_para,
        uvec_perp,
        wellwidth,
        image,
        debug=False,
        tracked=False):

    wellimage = np.zeros(image.shape, dtype="uint16")
    numseps = normseps.round()
    # corrected = seps/numseps
    # wellsep = np.mean(corrected)
    #corrected_normed = corrected / wellsep
    
    if tracked is False:
        #find the median well separations
        seps = np.diff( posperp_sorted )
        #the separations have already been filtered so we can take the min here
        #separations can still be >1 so mean and median may not work!!!
        min_sep = np.min(seps)

        #find how many extra wells we can fit in at either end
        low_extra = int((posperp_sorted[0]/min_sep))
        high_bound = image.shape[1]-(posperp_sorted[len(posperp_sorted)-1])
        high_extra = int((high_bound/min_sep))

        # Now whereever we have a sep of > 1, we need to
        # interpolate in additional wells!

        #if we use the above method simply replace intpos with posp_numbered
        intpos = [0,] + list(np.cumsum(numseps))
        spline = InterpolatedUnivariateSpline(intpos, posperp_sorted, k=1)
        int_interp = np.arange(0-low_extra, np.max(intpos)+1+high_extra)

        posperp_interp = spline(int_interp)
    else:
        # corrected = seps/numseps
        # wellsep = np.mean(corrected)
        #corrected_normed = corrected / wellsep

        # Now whereever we have a sep of > 1, we need to
        # interpolate in additional wells!

        #if we use the above method simply replace intpos with posp_numbered
        intpos = [0,] + list(np.cumsum(numseps))
        posperp_interp = np.interp(
            np.arange(0, np.max(intpos)+1),
            intpos,
            posperp_sorted,
        )
            
    if debug:
        print("INTERPOLATED WELL POSITIONS ALONG AXIS")
        print(posperp_interp)

    # Let's look at the max and min extents along parallel
    # direction
    maxparallel = []
    minparallel = []
    for i,prop in enumerate(props):
        posp = np.array([ np.dot( uvec_para, p[[1,0]]) for p in prop.coords])
        maxparallel.append(posp.max())
        minparallel.append(posp.min())

    # Use "average" end positions?
    #medmax = np.median(maxparallel)
    #medmin = np.median(minparallel)
    # Use extremal end positions
    medmax = np.max(maxparallel)
    medmin = np.min(minparallel)
    p00 = medmin * uvec_para
    p01 = medmax * uvec_para
    p10 = p00 + 1000*uvec_perp
    p11 = p01 + 1000*uvec_perp

    if debug:
        print("Median maximal parallel position", medmax)
        print("Median minimal parallel posotion", medmin)
        print("Basline")
        print(p00, p01)
        print(p10, p11)
    # Extract profiles
    #profiles = []
    #for pperp in posperp_interp:
    #
    #    p0 = p00 + pperp * uvec_perp
    #    p1 = p01 + pperp * uvec_perp
    #
    #    # Now lets get a good line-scan for each well
    #    profiles.append(skmeas.profile_line(
    #        image,
    #        p0[[1,0]],
    #        p1[[1,0]],
    #        linewidth=wellwidth,
    #    ))

    # Lets see the images instead of the profiles
    images = {}
    #images = []
    coords = []
    #we need to take away "low_extra" (the number of new wells we can extrapolate)
    #from the minimum label - so it assigns them properly when tracked!
    for numnow, pperp in enumerate(posperp_interp, min([p.label for p in props])):
        p0 = p00 + pperp * uvec_perp
        p1 = p01 + pperp * uvec_perp
        # TODO: Propose removal : Is this useful for debug?
        #if debug:
        #    print("Well %d"%numnow)
        #    print("Line coordinates:", p0, p1)
        perp_lines = skmeas.profile._line_profile_coordinates(
            p0[[1,0]], p1[[1,0]],
            linewidth=wellwidth)
        # TODO: Propose removal : Is this useful for debug?
        #if debug:
        #    print("Coordinates from skmeas")
        #    print(perp_lines)
        #    print("as int")
        #    print(perp_lines.round().astype(int))
        pixels = ndi.map_coordinates(image, perp_lines,
                order=1, mode='constant', cval=0.0)
        #images.append(pixels)
        images[numnow]=pixels #creates a dictionary of wells with each well linked to a number

        perp_line_coords = perp_lines.round().astype(int).squeeze()
        for ndim in range(wellimage.ndim):
            perp_line_coords[ndim][perp_line_coords[ndim] < 0] = 0
            perp_line_coords[ndim][perp_line_coords[ndim]
                >= wellimage.shape[ndim]] = wellimage.shape[ndim]-1
        perp_line_coords = tuple(perp_line_coords.tolist())
        wellimage[perp_line_coords] = numnow
        coords.append(perp_line_coords)
    return images, wellimage, coords

#------------------------------
# End of extract_well_profiles subfunctions
#------------------------------


def remove_background_max(profiles):
    """
    Assumes background well image is maximum value
    across each well image
    """
    bglevel = np.max(list(profiles.values()), axis=0)
    newprofiles = {}
    for k2, image2 in profiles.items():
        p = bglevel-image2
        p2 = p*(p>0)
        newprofiles[k2] = p2
    return newprofiles



def remove_background(profiles, radius=20, light_background=True):
    """
    Uses port of ImageJ rolling ball background subtraction
    to estimate background
    """
    # Make "spherical" structuring element
    sz = 2*radius  + (radius+1)%2
    X,Y = np.meshgrid(range(sz), range(sz))
    ballheight = float(radius**2) - (X-radius)**2 - (Y-radius)**2
    ballheight[ballheight<0] = 0
    ballheight = np.ma.masked_where(ballheight < 0, ballheight)
    ballheight = np.sqrt(ballheight)
    newprofiles = {}
    for k, im in profiles.items():
        # Run background subtraction
        if light_background:
            imax = im.max()
            im2 = imax-im
            bg = ndi.grey_opening(im2, structure=ballheight, mode="reflect")
            im2 -= bg
            newprofiles[k] = im2-imax
        else:
            bg = ndi.grey_opening(im, structure=ballheight, mode="reflect")
            newprofiles[k] = bg-im
    return newprofiles

def detect_bacteria_in_wells(
    wellimages,
    timepoint = None,
    label_dict_string = None, #dictionary of string labels
    maxsize = 1500, # maximum area (in pixels) of an object to be considered a bacteria
    minsize = 20, # maximum area (in pixels) of an object to be considered a bacteria
    absolwidth = 1, #width (in pixels) at which something is definitely a bacteria
    distfrombottom = 30, #ignores anything labeled this distance from the bottom of the well (prevents channel border being labelled)
    topborder = 3, # Distance to exclude from top due to "shadow"
    toprow = 12,#number of pixels along the top row for a label to be discarded
    thresh_perc = 0.75, #percentage of threshold to use 1=100% - long term just change filter
    ):

    segs = {}
    segs2 = []
    segs3 = {}

    #sigma_list = np.arange(0.5, 3.5, 0.1)
    sigma_list = np.arange(2.0, 6.0)
    for n, im in wellimages.items():
        #find the well shape so we can get the coordinates of the top of the well
        a = im.shape
        ylen = a[0]
        #the top few pixels of the well is often just shadow and is sometimes mistakenly
        #labelled as bacteria - so will use this value later to filter it out
        toppixofwell = ylen - topborder
        # Basic filtering

        #using scale space
        gl_images = [-ndi.gaussian_laplace(im, s, mode="nearest") * s ** 2
            for s in sigma_list]
        newwell =np.max(gl_images, axis=0)

        segs[n]= newwell
        segs2.append(newwell)
    #can we "normalise" the numbers after the filter so it's between a set value?
    #or just change filter
    thresh = (skfilt.threshold_otsu(np.concatenate([s.flatten() for s in segs2]))*thresh_perc)
    smax = 0
    for i,f in segs.items():
        bw = f > thresh
        bw1 = ndi.morphology.binary_erosion(bw)
        bw2 = ndi.morphology.binary_dilation(bw)
        bw3 = bw2^bw1

        #perform distance transform on filtered image
        dist = ndi.distance_transform_edt(bw1)

        #create markers for watershed
        markers = np.zeros_like(f)


        markers[dist >= absolwidth] = 2
        markers[dist < absolwidth] = 0
        markers = ndi.label(markers)[0]
        markers = markers+1
        markers[bw3]=0

        #Perform watershed
        segmentation = watershed(f, markers)
        segmentation = ndi.binary_fill_holes(segmentation - 1)

        #label image
        labeled_bacteria, Nbac = ndi.label(segmentation)
        newbac = np.zeros(labeled_bacteria.shape, dtype=labeled_bacteria.dtype)

        for labnow, region in enumerate(regionprops(labeled_bacteria),start=1):
            # TODO: REMOVE - POINTLESS
            #newbac2 = np.zeros(labeled_bacteria.shape, dtype=labeled_bacteria.dtype)
            if(maxsize > region.area > minsize):
                ycoords = []
                for coords in region.coords:
                    y = coords[0]
                    ycoords.append(y)
                toppix = 0
                for coords in ycoords:
                    if coords >= toppixofwell:
                        toppix += 1
                if ((all(i <= distfrombottom for i in ycoords)) == False): #may need to change to any - move to split!!!
                    if toppix <= toprow:
                        #if they have passed all of these then we can label them in "newbac"
                        smax += 1
                        newbac[labeled_bacteria==labnow] = smax

        segs3[i] = newbac

    return segs3

def split_bacteria(
    segs,
    label_dict_string,
    timepoint=None,
    relative_width = 5, #how narrow bacteria must become to be split e.g. 4 = 1/4 of the median width
    min_area = 20, #the minimum area of a new "split" bacteria in pixels
    change_in_centroid = 5, #minimum change in centroid - unsplit bacteria will be similar/same
    area_change = 20, #percentage change in area - unsplit bacteria shouldn't change much
    distfrombottom = 20, #ignores anything labeled this distance from the bottom of the well (prevents channel border being labelled)
    absolwidth = 2, #average width of a bacteria (determined along the skeleton)
    ):

    """NOTE!!! We may want to make some sort of loop for multiple splits. But at the moment it uses the whole skeleton
    and directly references coordinates so should be okay"""
    split_bac = {}
    if timepoint==0:
        label_dict_string = {}
        smax=0
    else:
        smax = max(label_dict_string, key=int)
    for num, well in segs.items():
        tempwell = np.copy(well)
        widths = []
        max_label = np.max(tempwell)
        max_label2 = np.max(tempwell)
        comparison_well =np.zeros(well.shape, dtype=well.dtype)
        if len(regionprops(tempwell)) == 0:
            comparison_well = np.copy(well)
        else:
            for i, region in enumerate(regionprops(tempwell)):
                widths.append(region.minor_axis_length)
            w = median(widths) #find median width
            """

                maxy = max(region.coords[0:,0]) #find max and min y coordinates in region
                miny = min(region.coords[0:,0])
                for c in  region.coords:
                    if (maxy-w)<=c[0]<=(maxy) or (miny+w>=c[0]>=miny):
                        #if the y coordinate falls within the width from the top/bottom set the value to 0 in tempwell
                        tempwell[c[0]][c[1]] = 0
            """
            skel,dist_temp = skmorph.medial_axis(tempwell, return_distance=True)
            skel = ndi.morphology.binary_dilation(skel)
            skel = dist_temp*skel #distance transform of skeleton
            markers = np.zeros_like(skel)
            skel[skel==0] = 100 #set the background to a high number so it doesn't get included in the next step
            markers[skel<(w*(1/relative_width))] = 1 #1/4 of width
            markers, N = ndi.label(markers)
            for j, r in enumerate(regionprops(markers)):
                for coords in r.coords:
                    tempwell[coords[0]][coords[1]] = 0 #set any of these points to 0 in the temporary image (copy of original)
            new_tempwell, N = ndi.label(tempwell) #label this image
            for k, r_new in enumerate(regionprops(new_tempwell)):
                if r_new.area > min_area: #remove small areas
                    max_label += 1 #give it a temporary "new label" - unique
                    comparison_well[new_tempwell==r_new.label] = max_label
            if len(regionprops(well)) == len(regionprops(comparison_well)):
                comparison_well = np.copy(well) #if the number of regions hasn't changed we will simply use the original image
            elif len(regionprops(well)) < len(regionprops(comparison_well)):#if we have "split" some bacteria we need to do a bit more
                for r_old in regionprops(well):
                    for lbl in regionprops(comparison_well):
                        if (abs(r_old.centroid[0]-lbl.centroid[0])<change_in_centroid) & ((1-(area_change/100))<(r_old.area/lbl.area)<(1+(area_change/100))):
                            #if the centroid and area hasn't changed much then use old label to prevent erosions in loops
                            comparison_well[well==r_old.label] = r_old.label
                            break
        newwell=np.zeros(comparison_well.shape, dtype=comparison_well.dtype)
        #We need to make sure the labels are in an order so that they are assigned an "smax" value in a nice way
        comparison_well, N = ndi.label(comparison_well)
        for r2 in regionprops(comparison_well):
            ycoords = []
            for coords in r2.coords:
                ycoords.append(coords[0])
            if ((all(i <= distfrombottom for i in ycoords)) == False): #may need to change to any - move to split!!!
                #If we simply use "minor_axis_length" then it has to be large enough to remove wrong "fake" bacteria
                #this means we lose some "real" bacteria as well - oftem the "fake bacteria" are long skinny segments
                #from the center of the well - so we should instead find the average width along the skeleton
                tempwell=np.zeros(comparison_well.shape, dtype=comparison_well.dtype)
                tempwell[comparison_well==r2.label] = 1
                skel,dist_temp = skmorph.medial_axis(tempwell, return_distance=True)
                skel = dist_temp*skel #distance transform of skeleton
                av_width = skel[skel>0].mean()
                if av_width > absolwidth:
                    smax +=1
                    newwell[comparison_well==r2.label] = smax
                    if timepoint == 0:
                        label_dict_string[smax] = str(smax)
        split_bac[num]= newwell

    return split_bac, label_dict_string
