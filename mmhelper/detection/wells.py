"""
Well detection functions
"""
import math
import skimage.filters as skfilt
import skimage.measure as skmeas
from skimage.filters import sobel
from skimage.measure import regionprops
import mmhelper.skimage_future as skfuture
from mmhelper.utility import logger
import scipy.spatial as scispat
import scipy.ndimage as ndi
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import numpy as np

def detect_wells(
        image,
        debug="",
        phase=False,
        scale_factor=1,
    ):
    """
    Detect the wells using a ridge filter, and the classifying
    based on their periodic property

    Parameters
    ------
    image : ndarray (2D)
        The image to analyse
    debug   : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)
    phase : Boolean, optional
        Whether the image is brightfield or phase (default : False)
    scale_factor : float, optional
        Used to scale other parameters depending on the image magnification (default: 1)

    Returns
    ------
    wells : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    detected_wellimg : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    wellcoords : Dictionary
        Key is the well number and the value is an array of coordinates for the respective well
    """

    labelled_wellimg = detect_initial_well_masks(
        image,
        phase=phase,
        debug=debug,
        scale_factor=scale_factor)[0]
    wells, detected_wellimg, wellcoords = extract_well_profiles(
        image, labelled_wellimg, scale_factor=scale_factor)
    return wells, detected_wellimg, wellcoords

def detect_initial_well_masks(
        image,
        scale_range=[4.0, 7.0],
        maxd=400,
        mind=100,
        maxperp=30,
        min_outline_area=2500,
        merge_length=8,
        debug="",
        phase=False,
        scale_factor=1,
):
    """
    Detect the wells using a ridge filter, and the classifying
    based on their periodic property

    Parameters
    ------
    image : ndarray (2D)
        The image to analyse
    scale_range : 2-tuple of floats, optional
        The range of sigmas used (default: [4.0,.7.0])
    maxd : float, optional
        maximum length of a detected well (default: 400)
    mind : float, optional
        minimum length of a detected well (default: 100)
    maxperp : float, optional
        maximum width of a detected well (default: 30)
    min_outline_area : float, optional
        minimum area in pixels of the outline of the wells (default: 2500)
    merge_length : int, optional
        number of iterations for binary closing of the wells (default: 5)
    debug   : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)
    phase : Boolean, optional
        Whether the image is brightfield or phase (default : False)
    scale_factor : float, optional
        Used to scale other parameters depending on the image magnification (default: 1)

    Returns
    ------
    lbl_final : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    ridges : ndarray (2D) of dtype int
        A labelled image showing the detected ridges
    """
    mind = (mind * scale_factor)
    maxd = (maxd * scale_factor)
    maxperp = (maxperp * scale_factor)
    # multiply the scale range by the square root of the scale factor
    #scale_range = list(np.array(scale_range) * (scale_factor**(1/2.0)))
    logger.debug("Detecting well with parameters:")
    logger.debug("  scale range : %s" % str(scale_range))
    logger.debug("  maxd      : %s" % str(maxd))
    logger.debug("  mind    : %s" % str(mind))
    # NOTE: frangi detects DARK ridges
    #skewness = scistats.skew(image.flat)
    # if image.max() > 10000: #if really high it is a dark image so probably phase
    #    phase = True
    if phase is True:
        filt = sobel(
            image,
        )
    else:
        filt = skfuture.frangi(
            image,
            scale_range=scale_range,
        )

    ridges = filt > skfilt.threshold_li(filt)
    # if minwidth:
    #    ridges = skmorph.opening(ridges, selem=skmorph.disk(minwidth))
    if debug:
        plt.figure(figsize=(16, 12))
        plt.imshow(filt, cmap='gray')
        plt.contour(ridges, levels=[0.5], colors=["r"], linewidths=[4, ],
                    label="Initial threshold")
    # Classify using simple measures
    lbl = skmeas.label(ridges, return_num=True)[0]
    lblgood = np.zeros(lbl.shape, dtype='int16')
    smax = 0
    for region in regionprops(lbl):
        #print(region.label, region.area)
        if region.area > min_outline_area:
            smax += 1
            lblgood[lbl == region.label] = smax
    lblgood = lblgood > 0
    best_ngood = 0

    best_lbl_final = np.zeros(lbl.shape, dtype='int16')
    for num_int in range(1, 6):  # TODO: review this for loop
        # dilate the labels to ensure the wells "attach" to the channel
        lblgood2 = ndi.morphology.binary_dilation(lblgood)#, iterations=num_int)
        # fill the wells
        lbl_filled = ndi.morphology.binary_fill_holes(lblgood2)
        # extract only the middles
        wells_middle = lbl_filled ^ lblgood2
        # have to redilate wells_middle as we narrowed it by 2 pixels earlier
        wells_middle = ndi.morphology.binary_dilation(
            wells_middle, iterations=num_int)
        if merge_length > 0:
            wells_middle = ndi.morphology.binary_closing(
                wells_middle, iterations=merge_length)
        if debug:
            # plt.contour(lblgood, levels=[0.5], colors=["m"], linewidths=[3,],
            #    label="'Good' wells")
            # plt.contour(lbl_filled, levels=[0.5], colors=["c"], linewidths=[2,],
            #    label="'Filled' wells")
            # plt.contour(wells_middle, levels=[0.5], colors=["y"], linewidths=[1,],
            #    label="Well middles")
            pass
        # Classify using simple measures
        lbl_wells, n_wells = skmeas.label(wells_middle, return_num=True)
        #props = skmeas.regionprops(lbl)
        bwnew = np.zeros(lbl_wells.shape, "bool")
        ngood = 0
        lbl_final = np.zeros(lbl_wells.shape, dtype='int16')
        for lbl in range(1, n_wells + 1):
            bw_ = lbl_wells == lbl
            # Size
            area = bw_.sum()
            if area > (maxd * maxd / 2) or area < mind:
                continue
            perim = bw_ ^ ndi.binary_erosion(bw_)
            pts = np.array(perim.nonzero(), dtype=float).T
            # Length = max separation between any two points in each region
            maxdist = scispat.distance.cdist(pts, pts)
            dnow = maxdist.max()
            imax = (maxdist == dnow).nonzero()
            vmax = pts[imax[0][0]] - pts[imax[1][0]]
            vmax /= np.sqrt(np.sum(vmax**2))
            vperp = np.array([-1 * vmax[1], vmax[0]])
            pperp = [np.dot(vperp, p) for p in pts]
            distperp = np.max(pperp) - np.min(pperp)
            if (dnow > mind) & (dnow < maxd) & (distperp < maxperp):
                # maxperp is causing some "well loss" on detection - values occassionally
                # higher than 20 but less than 22 (2 well widths) so will change to 22
                # May be due to knew detection method so need to revise!!!
                bwnew[bw_] = True
                ngood += 1
                lbl_final[bw_] = ngood
        # if ngood doesn't reach 5 we still need it to return the
        # best result so we will store the best
        if ngood > best_ngood:
            best_ngood = ngood
            best_lbl_final = lbl_final
        if ngood > 5:
            break
    ngood = best_ngood
    if ngood < 2:
        logger.warning(
            "Less than 2 wells detected, this function has probably failed")
    lbl_final = best_lbl_final
    if debug:
        plt.contour(lbl_final, levels=[0.5], colors=["g"], linewidths=[1, ],
                    labels=["Final wells (after length filtering)", ])
        plt.legend()
        plt.savefig(debug)
        plt.close()
    # Lastly let's see if we can detect anomolous things
    #propsgood = skmeas.regionprops(lblgood)
    #good = np.zeros(lblgood.shape)
    #bad = np.zeros(lblgood.shape)
    #mus = np.array([p.moments_hu for p in propsgood])
    #mus2 = mus - mus.mean(axis=0)
    return lbl_final, ridges  # For debugging


def extract_well_profiles(
        image,
        wells,
        wellwidth=11,
        debug=False,
        min_well_sep_factor=2.5,
        max_well_sep_factor=6,
        scale_factor=1,
):
    """
    * Extend wells down to channel and up to maximal height
    * Add in missing wells (using interpolation of positions)
    * Extract well profiles

    Parameters
    ------
    image : ndarray (2D)
        The image to analyse
    lbl_final : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    wellwidth : float, optional
        Width of the wells (default : 11)
    debug   : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)
    min_well_sep_factor : float, optional
        Minimum distance between wells as a factor of the well width (default : 2.5)
    max_well_sep_factor : float, optional
        Maximum distance between wells as a factor of the well width (default : 6)
    scale_factor : float, optional
        Used to scale other parameters depending on the image magnification (default: 1)

    Returns
    ------
    images : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    wellimage : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    coords : Dictionary
        Key is the well number and the value is an array of coordinates for the respective well
    """
    wellwidth = (wellwidth * scale_factor)
    # -------------------
    # A bit of pre-processing on the wells
    propsgood = skmeas.regionprops(wells)
    # Make sure we have at least 2 wells... at least this many for the
    # interpolation...
    # TODO: Decide if we need more
    if len(propsgood) < 2:
        logger.error(
            "Less than 2 wells detected; unable to extract well profiles")
        blank_wellimage = np.zeros(image.shape, dtype="uint16")
        return {}, blank_wellimage, {}

    coms, oris, uvec_para, uvec_perp = get_wells_and_unit_vectors(
        propsgood,
        debug=debug,
    )

    normseps, posperp_sorted = well_spacing_and_seps(
        coms,
        uvec_perp,
        wellwidth,
        debug=debug,
        min_well_sep_factor=min_well_sep_factor,
        max_well_sep_factor=max_well_sep_factor,
    )

    images, wellimage, coords = interpolate_pos_extract_profs(
        normseps,
        posperp_sorted,
        propsgood,
        uvec_para,
        uvec_perp,
        wellwidth,
        image,
        debug=debug,
    )

    images = remove_background(images)
    return images, wellimage, coords


# ------------------------------
# Subfunctions for extract_well_profiles
# ------------------------------

def get_wells_and_unit_vectors(props, debug=False):
    """
    Determines the perpendicular positions of the wells
    and the min and max coordinates in the parallel direction

    Parameters
    ------
    props : list of RegionProperties
        Each item describes one labeled region
    debug : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)

    Returns
    ------
    coms : array
        An array of the center of mass
    oris :
        An array of orientations
    uvec_para : array
        An array of parallel unit vectors
    uvec_perp : array
        An array of perpendicular unit vectors
    """
    coms = []
    oris = []
    lens = []

    for prop in props:
        # Convert to simple line using
        # centroid and orientation information
        y0_, x0_ = prop.centroid
        orientation = prop.orientation
        # Make sure angle is in range 0, pi
        ori2 = orientation % np.pi

        # Find maximal well length (separation of points in binary image)
        dist = scispat.distance.cdist(prop.coords, prop.coords)
        length = dist.max()
        #x1 = x0_ + math.cos(ori2) * 0.5 * length
        #y1 = y0_ - math.sin(ori2) * 0.5 * length
        #x2 = x0_ - math.cos(ori2) * 0.5 * length
        #y2 = y0_ + math.sin(ori2) * 0.5 * length
        coms.append(np.array((x0_, y0_)))
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
    pospara = [np.dot(uvec_para, p) for p in coms]
    pospara_med = np.median(pospara)
    pospara_dif = np.abs(np.array(pospara) - pospara_med)
    pospara_bad = pospara_dif > (0.5 * np.median(lens))
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


def well_spacing_and_seps(
        coms,
        uvec_perp,
        wellwidth,
        min_well_sep_factor=2.5,
        max_well_sep_factor=8,
        debug=False):
    """
    Determines the separations between the wells and makes sure they are consistent

    Parameters
    ------
    coms : array
        An array of the center of mass
    uvec_perp : array
        An array of perpendicular unit vectors
    wellwidth : float, optional
        Width of the wells (default : 11)
    min_well_sep_factor : float, optional
        Minimum distance between wells as a factor of the well width (default : 2.5)
    max_well_sep_factor : float, optional
        Maximum distance between wells as a factor of the well width (default : 8)
    debug : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)

    Returns
    ------
    normseps : array
        An array of normalised separations between wells
    posperp_sorted : array
        An array of separations between wells sorted from the smallest distance
    """
    # Let's get the spacing
    posperp = [np.dot(uvec_perp, p) for p in coms]
    if debug:
        print("Perpendicular positions")
        print(posperp)
    posperp_sorted = np.sort(posperp)
    seps = np.diff(posperp_sorted)
    # create a duplicate for determination of median
    seps2 = np.diff(posperp_sorted)
    # Remove any separations that are less than the well-width,
    # these are likely due to fragmentation of a well
    goodseps = seps > wellwidth
    seps = seps[goodseps]
    posperp_sorted = posperp_sorted[np.concatenate([[True, ], goodseps])]
    seps_sorted = np.sort(seps)
    # use only "realistic" separations for determining median
    # I.e. bigger than one well width but less than 6
    goodseps2 = (
        (seps2 >= (wellwidth * min_well_sep_factor))
        & (seps2 < (wellwidth * max_well_sep_factor)))
    seps2 = seps2[goodseps2]
    seps2_sorted = np.sort(seps2)

    if debug:
        print("\nSaving SEPARATIONS.jpg...")
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
    lowersep = np.median(lowerseps)  # determine median from "realistic values"
    normseps = seps / lowersep  # normalise separations to the median
    normseps_temp = normseps.round()  # round to the nearest whole number
    # subtract actual normalised values
    normseps_temp = abs(normseps_temp - normseps)
    wrong_seps = []
    for i, j in enumerate(normseps_temp):
        if j > 0.1:  # "correct" values will be close to whole numbers when divided by median
            # append the index value of "incorrect" values
            wrong_seps.append(i)
    # delete these "incorrect" values
    posperp_sorted = np.delete(posperp_sorted, wrong_seps)
    seps3 = np.diff(posperp_sorted)
    normseps = seps3 / lowersep
    normseps_sorted = np.sort(normseps)

    if len(seps_sorted) > 5:
        # 2017/02/01 13:21:26 (GMT)[JM] As above, should be median here??
        #lowernormed = np.mean(normseps_sorted[:5])
        lowernormed = np.median(normseps_sorted[:5])
    else:
        lowernormed = normseps_sorted[0]

    if debug:
        print("normseps_sorted")
        print(normseps_sorted)
        print("LOWERNORMED")
        print(lowernormed)

    if (lowernormed > 0.95) and (lowernormed < 1.05):
        # We're pretty sure we've got a good estimator
        # here
        pass
    else:
        raise Exception(
            "Wells not extracted properly - mean normalized separation\n" +
            "not close to 1: %f [lowersep = %f, lowerseps = %s]" %
            (lowernormed,
             lowersep,
             str(lowerseps)))

    if debug:
        print("NORMSEPS")
        print(normseps)

    return normseps, posperp_sorted


def interpolate_pos_extract_profs(
        normseps,
        posperp_sorted,
        props,
        uvec_para,
        uvec_perp,
        wellwidth,
        image,
        debug=False,
):
    """
    Uses the determined spacing to interpolate any missing wells and then
    extracts their profiles

    Parameters
    ------
    normseps : array
        An array of normalised separations between wells
    posperp_sorted : array
        An array of separations between wells sorted from the smallest distance
    props : list of RegionProperties
        Each item describes one labeled region
    uvec_para : array
        An array of parallel unit vectors
    uvec_perp : array
        An array of perpendicular unit vectors
    wellwidth : float
        Width of the wells
    image : ndarray (2D)
            The image to analyse
    debug : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)

    Returns
    ------
    images : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    wellimage : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    coords : Dictionary
        Key is the well number and the value is an array of coordinates for the respective well
    """

    wellimage = np.zeros(image.shape, dtype="uint16")
    numseps = normseps.round()
    # corrected = seps/numseps
    # wellsep = np.mean(corrected)
    #corrected_normed = corrected / wellsep

    # find the median well separations
    seps = np.diff(posperp_sorted)
    # the separations have already been filtered so we can take the min here
    # separations can still be >1 so mean and median may not work!!!
    min_sep = np.min(seps)

    # find how many extra wells we can fit in at either end
    low_extra = int((posperp_sorted[0] / min_sep))
    high_bound = image.shape[1] - posperp_sorted[-1] - 1
    high_extra = int((high_bound / min_sep))

    # Now whereever we have a sep of > 1, we need to
    # interpolate in additional wells!

    # if we use the above method simply replace intpos with posp_numbered
    intpos = [0, ] + list(np.cumsum(numseps))
    spline = InterpolatedUnivariateSpline(intpos, posperp_sorted, k=1)
    int_interp = np.arange(0 - low_extra, np.max(intpos) + 1 + high_extra)

    posperp_interp = spline(int_interp)

    if debug:
        print("INTERPOLATED WELL POSITIONS ALONG AXIS")
        print(posperp_interp)

    # Let's look at the max and min extents along parallel
    # direction
    maxparallel = []
    minparallel = []
    for prop in props:
        posp = np.array([np.dot(uvec_para, p[[1, 0]]) for p in prop.coords])
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
    p10 = p00 + 1000 * uvec_perp
    p11 = p01 + 1000 * uvec_perp

    if debug:
        print("Median maximal parallel position", medmax)
        print("Median minimal parallel posotion", medmin)
        print("Basline")
        print(p00, p01)
        print(p10, p11)
    # Extract profiles
    #profiles = []
    # for pperp in posperp_interp:
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
    coords = {}
    # we need to take away "low_extra" (the number of new wells we can extrapolate)
    # from the minimum label - so it assigns them properly when tracked!
    for numnow, pperp in enumerate(
            posperp_interp, min([p.label for p in props])):
        pos0 = p00 + pperp * uvec_perp
        pos1 = p01 + pperp * uvec_perp
        # TODO: Propose removal : Is this useful for debug?
        # if debug:
        #    print("Well %d"%numnow)
        #    print("Line coordinates:", p0, p1)
        perp_lines = np.array(skmeas.profile._line_profile_coordinates(
            pos0[[1, 0]], pos1[[1, 0]],
            linewidth=wellwidth))
        # TODO: Propose removal : Is this useful for debug?
        # if debug:
        #    print("Coordinates from skmeas")
        #    print(perp_lines)
        #    print("as int")
        #    print(perp_lines.round().astype(int))
        pixels = ndi.map_coordinates(image, perp_lines,
                                     order=1, mode='constant', cval=0.0)
        # images.append(pixels)
        # creates a dictionary of wells with each well linked to a number
        images[numnow] = pixels

        perp_line_coords = perp_lines.round().astype(int).squeeze()
        for ndim in range(wellimage.ndim):
            perp_line_coords[ndim][perp_line_coords[ndim] < 0] = 0
            perp_line_coords[ndim][perp_line_coords[ndim] >=
                                   wellimage.shape[ndim]] = wellimage.shape[ndim] - 1
        perp_line_coords = tuple(perp_line_coords.tolist())
        wellimage[perp_line_coords] = numnow
        coords[numnow] = perp_line_coords
    return images, wellimage, coords

# ------------------------------
# End of extract_well_profiles subfunctions
# ------------------------------

def remove_background(profiles, radius=20, light_background=True):
    """
    Uses port of ImageJ rolling ball background subtraction
    to estimate background and removes the background from the image

    Parameters
    ------
    profiles : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    radius : float, optional
        The radius of the rolling ball (default : 20)
    light_background : Boolean
        Whether the background is light or not (default : True)

    Returns
    ------
    newprofiles : Dictionary
        Key is the well number and the value is a ndarray (2D) of the
        background subtracted well
    """
    # Make "spherical" structuring element
    sz_ = 2 * radius + (radius + 1) % 2
    xco, yco = np.meshgrid(range(sz_), range(sz_))
    ballheight = float(radius**2) - (xco - radius)**2 - (yco - radius)**2
    ballheight[ballheight < 0] = 0
    ballheight = np.ma.masked_where(ballheight < 0, ballheight)
    ballheight = np.sqrt(ballheight)
    newprofiles = {}
    for k, im1 in profiles.items():
        # Run background subtraction
        if light_background:
            imax = im1.max()
            im2 = imax - im1
            bg1 = ndi.grey_opening(im2, structure=ballheight, mode="reflect")
            im2 -= bg1
            newprofiles[k] = im2 - imax
        else:
            bg1 = ndi.grey_opening(im1, structure=ballheight, mode="reflect")
            newprofiles[k] = bg1 - im1
    return newprofiles
