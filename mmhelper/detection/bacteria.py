"""
Bacteria detection functions
"""
import math
import numpy as np
from mmhelper.utility import logger
import skimage.measure as skmeas
from skimage.morphology import watershed, medial_axis, skeletonize
from skimage.filters import sobel
import skimage.filters as skfilt
from skimage.measure import regionprops
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import scipy.ndimage as ndi


def run_bacteria_detection(
        wells,
        debug="",
        scale_factor=1,
        ):
    """
    Runs the bacteria detection function then splits bacteria_image

    Parameters
    ------
    wells: Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    debug : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)
    scale_factor : Float, optional
        Used to scale other parameters depending on the image magnification (default : 1)

    Returns
    ------
    bacteria : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """

    initial_bacteria = detect_bacteria_in_all_wells(
        wells,
        debug=debug,
        scale_factor=scale_factor,
    )
    # should this be part of bacteria detection?
    unlabelled_bacteria = split_bacteria_in_all_wells(initial_bacteria, wells)
    bacteria = relabel_bacteria(unlabelled_bacteria)
    return bacteria


def detect_bacteria_in_all_wells(
        wellimages,
        maxsize=1500,
        minsize=20,
        absolwidth=1,
        min_av_width=3,
        sigma_list=(2.0, 6.0),
        scale_factor=1,
        debug="",
    ):
    """
    Takes a dictionary of wells, detects any bacteria in each well and returns
    a new dictionary with each value containing a labelled image of detected bacteria

    Parameters
    ------
    wellimages : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well
    maxsize : float, optional
        Maximum area (in pixels) of an object to be considered a bacteria (default : 1500)
    minsize : float, optional
        Maximum area (in pixels) of an object to be considered a bacteria (default : 20)
    absolwidth : float, optional
        Width (in pixels) at which something is definitely a bacteria (default : 1)
    min_av_width : float, optional
        Minimum average width (in pixels) of a bacteria
    sigma_list : scalar arguments, optional
        Sigma from ndi.gaussian_laplace, list of the standard deviations of the Gaussian filter
        Scaled with the scale factor (default : (2, 6))
    scale_factor : Float, optional
        Used to scale other parameters depending on the image magnification (default : 1)
    debug : Boolean, optional
        Whether to add debugging outputs, save debug images with this basename (default : False)

    returns
    ------
    segs2 : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """
    maxsize = scale_factor * maxsize
    minsize = scale_factor * minsize
    absolwidth = scale_factor * absolwidth
    # divide by 2 first as its the half width (because of the skeleton)
    min_av_width = scale_factor * (min_av_width / 2)

    segs = {}
    segs2 = {}
    logger.debug("Detecting bacteria in wells")
    #sigma_list = np.arange(0.5, 3.5, 0.1)
    sigma_list = np.arange(*sigma_list) * math.sqrt(scale_factor)
    for well_num, im1 in wellimages.items():
        # using scale space
        gl_images = [-(ndi.gaussian_laplace(im1, s, mode="nearest")) * s ** 2
                     for s in sigma_list]
        newwell = np.max(gl_images, axis=0)

        segs[well_num] = newwell

    # can we "normalise" the numbers after the filter so it's between a set value?
    # or just change filter
    thresh = skfilt.threshold_li(np.concatenate(
        [s.flatten() for s in segs.values()]))
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        imfiltdebug = np.hstack(segs.values())
        plt.imshow(imfiltdebug, cmap='gray')
        maskdebug = imfiltdebug > thresh
        plt.imshow(np.ma.masked_where(~maskdebug, maskdebug), cmap='hsv')
        plt.savefig(debug)
        plt.close()
        #outpath_debug_tif = "_FILTERED".join(os.path.splitext(debug))+".tif"
        #skio.imsave(outpath_debug_tif, imfiltdebug)

    logger.debug(
        "  Calculated threshold for bacteria detection : %s" % str(thresh))
    smax = 0
    allstats = {}
    for well_number, img0 in segs.items():
        initialbac = detect_bacteria_in_well(
            img0,
            thresh,
            absolwidth,
            well_number=well_number,
            debug=debug,
        )
        filteredbac, stats = filter_bacteria(
            initialbac,
            min_av_width,
            minsize,
            maxsize,
        )

        for k,v in stats.items():
            allstats.setdefault(k, 0)
        allstats[k] += v
        segs2[well_number] = filteredbac

    logger.debug(
        "  Final number of detected bacteria following morphological property filtering : %d" %
        (smax))

    return segs2


def detect_bacteria_in_well(
        img0,
        thresh,
        absolwidth,
        well_number=-1,
        debug="",
    ):
    """
    Detects bacteria in a single well image `img0`.
    This is usually the result of a filtering step (e.g. `LoG`).

    Parameters
    ------
    img0 : ndarray (2D)
        Image of a single well
    Thresh : float
        Threshold to use (determined from all wells)
    absolwidth : float
        Width (in pixels) at which something is definitely a bacteria (default : 1)
    min_av_width : float
        Minimum average width (in pixels) of a bacteria
    minsize : float
        Maximum area (in pixels) of an object to be considered a bacteria (default : 20)
    maxsize : float
        Maximum area (in pixels) of an object to be considered a bacteria (default : 1500)
    well_number : int
        Label of current well for logging purposes
    smax : int
        Maximum label of full well image (for this frame)
    debug : boolean (or equivalent), (optional)
        Equated to True/False to determine if debug logging output should be generated

    returns
    ------
    newbac : ndarray (2D, ints)
        Labelled array of bacteria
    smax : int
        Maximum label of full well image (updated)
    """
    bw0 = img0 > thresh
    bw1 = ndi.morphology.binary_erosion(bw0)
    if debug:
        nbac0 = ndi.label(bw1)[1]
        logger.debug(
            "    Well %d - bacteria in initial binary image: %d" %
            (well_number, nbac0))
    bw2 = ndi.morphology.binary_dilation(bw0)
    bw3 = bw2 ^ bw1

    # perform distance transform on filtered image
    dist = ndi.distance_transform_edt(bw1)

    # create markers for watershed
    markers = np.zeros(img0.shape, dtype=bool)

    markers[dist >= absolwidth] = True
    markers = ndi.label(markers)[0]
    markers = markers + 1
    markers[bw3] = 0

    # Perform watershed
    segmentation = watershed(img0, markers)
    segmentation = ndi.binary_fill_holes(segmentation != 1)

    # label image
    labeled_bacteria, nbac = ndi.label(segmentation)
    logger.debug(
        "    Well %d - bacteria detected after initial filtering : %d" %
        (well_number, nbac))
    return labeled_bacteria


def filter_bacteria(
        labeled_bacteria,
        min_av_width,
        minsize,
        maxsize,
    ):
    """
    Filters bacteria (i.e. removes bacteria failing criteria) based on
    area and width.
    """

    dist = ndi.distance_transform_edt(labeled_bacteria > 0)

    newbac = np.zeros(labeled_bacteria.shape, dtype=labeled_bacteria.dtype)

    stats = dict(
        num_rejected_av_width=0,
        num_rejected_area=0,
    )

    label = 0
    for region in regionprops(labeled_bacteria):
        masked_bac = labeled_bacteria == region.label
        skel = skeletonize(masked_bac)
        av_width_skel = np.mean(dist[skel])
        if av_width_skel < min_av_width:
            stats["num_rejected_av_width"] += 1
            continue
        if maxsize > region.area > minsize:
            label += 1
            newbac[labeled_bacteria == region.label] = label
        else:
            stats["num_rejected_area"] += 1
    return newbac, stats


def split_bacteria_in_all_wells(
        bacteria,
        well_images,
        #min_break_size=2,
        min_skel_length=5,
    ):
    """
    Takes a dictionary containing a labelled image of detected bacteria
    and attempts to 'split' any labels which may have been detected as
    multiple bacteria instead of one

    Parameters
    ------
    bacteria : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    well_images : Dictionary
        Key is the well number and the value is a ndarray (2D) of the well

    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """
    dists = []
    intensities = []
    for well_label, bacteria_label_image in bacteria.items():
        distnow, intnow = get_bacteria_stats_in_well(
            bacteria_label_image,
            well_images[well_label],
            min_skel_length=min_skel_length
        )
        dists.extend(distnow)
        intensities.extend(intnow)

    dists = np.array(dists)
    intensities = np.array(intensities)
    import matplotlib.pyplot as plt

    iqr25, iqr75 = np.percentile(dists, [25, 75])
    dists2 = dists[(dists>=iqr25) & (dists<=iqr75)]
    iqr25, iqr75 = np.percentile(intensities, [25, 75])
    intensities2 = intensities[(intensities>=iqr25) & (intensities<=iqr75)]

    medd, madd = get_med_and_mad(dists)
    medd2, madd2 = get_med_and_mad(dists2)
    medi, madi = get_med_and_mad(intensities)
    medi2, madi2 = get_med_and_mad(intensities2)

    #plt.show()

    for well_label, bacteria_label_image in bacteria.items():
        newbacteria = split_bacteria_in_well(
            bacteria_label_image,
            well_images[well_label],
            min_skel_length=min_skel_length,
            threshold_stat=medd2-madd2,
            debug=dict(
                med_dist=medd, mad_dist=madd,
                med_dist2=medd2, mad_dist2=madd2,
                med_int=medi, mad_int=madi,
                med_int2=medi2, mad_int2=madi2,
            )
        )
        bacteria[well_label] = newbacteria
    return bacteria


def get_med_and_mad(data):
    """
    Return median and MAD of data
    """
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return med, mad


def threshold_mad(data, factor):
    """
    Determine Median Absolute Deviation based threshold
    """
    med, mad = get_med_and_mad(data)
    threshold = med - factor * mad
    return threshold


def get_bacteria_stats_in_well(
        bacteria_label_image,
        intensity_image,
        min_skel_length=6,
    ):
    """
    Get bacteria width stats in a single well image using
    distance transform and skeletonization

    Parameters
    ------
    bacteria_label_image : ndarray (2D)
        Labelled image of detected bacteria
    intensity_image : ndarray (2D)
        Intensity image for the current well
    min_skel_length : float (optional)
        Minimum skeleton length for a region to be considered for splitting
    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """
    mask = bacteria_label_image > 0
    dist = ndi.distance_transform_edt(mask)
    skel = skmorph.skeletonize(mask)
    skellabel = ndi.label(skel)[0]
    for prop in regionprops(skellabel):
        if prop.area < min_skel_length:
            skel[skellabel == prop.label] = False
        #skelmask = skellabel == prop.label

    return dist[skel], intensity_image[skel]


def split_bacteria_in_well(
        bacteria_label_image,
        intensity_image,
        min_skel_length=6,
        threshold_stat=None,
        debug=None,
    ):
    """
    Split bacteria in a single well image based on
    statistical thresholding of width and intensity along
    the skeleton

    Parameters
    ------
    bacteria_label_image : ndarray (2D)
        Labelled image of detected bacteria
    intensity_image : ndarray (2d)
        Intensity image of current well
    min_skel_length : float (optional)
        Minimum skeleton length for a region to be considered for splitting
    returns
    ------
    split_bac : Dictionary
        The key is the well coordinates and the value is a labelled image of detected bacteria
    """

    max_label = 0
    stats = []
    dist = ndi.distance_transform_edt(bacteria_label_image > 0)
    output_labels = bacteria_label_image.copy()
    for prop in regionprops(bacteria_label_image):
        region_mask = bacteria_label_image == prop.label
        #skel, dist = skmorph.medial_axis(region_mask, return_distance=True)
        skel = skmorph.skeletonize(region_mask)
        if skel.sum() < min_skel_length:
            max_label += 1
            output_labels[region_mask] = max_label
            continue

        skelprop = regionprops(skel.astype(int))[0]
        posy, posx = skelprop.coords.T

        skel_intensity = intensity_image[skel]
        skel_dist = dist[skel]

        threshold_int = threshold_mad(skel_intensity, 2)
        threshold_dist = threshold_mad(skel_dist, 1)

        inds = np.argsort(posy)
        for i, (name, vals, name2) in enumerate((
                ("Distance", skel_dist, "dist"),
                ("Intensity", skel_intensity, "int"),
                ("Combined", skel_dist*(-1*skel_intensity), None),
                )):
            med, mad = get_med_and_mad(vals)

        breaks = skel_intensity < (debug["med_int"] - debug["mad_int"])#threshold_int

        breaks_dist = skel_dist < debug["med_dist"] #threshold_dist
        #breaks_dist = skel_dist < np.median(skel_dist) #debug["med_dist"] #threshold_dist

        skel_dist_breaks = skel.copy()
        skel_dist_breaks[skel] = breaks_dist

        breaks[~breaks_dist] = False

        ### TODO: added this in as potential width ratio method
        ### neeed to rethink with Jeremy
        #width_ratio = 0.5
        #threshold_dist_ratio = width_ratio * np.max(skel_dist)
        #threshold_dist_ratio = np.median(skel_dist) - 1*threshold_stat
        threshold_dist_ratio = threshold_stat


        breaks_dist_hard = skel_dist < threshold_dist_ratio
        #breaks[breaks_dist_hard] = True

        skel_breaks = skel.copy()
        skel_breaks[skel] = breaks

        break_labels, n_breaks = ndi.label(skel_breaks)

        sizes = ndi.sum(
            skel_breaks, break_labels,
            index=np.arange(1, n_breaks + 1)
        )
        for label, break_size in enumerate(sizes, start=1):
            # if break_size < min_break_size:
            if break_size < 0:
                skel_breaks[break_labels == label] = False
        bacteria_markers, n_bacteria_markers = ndi.label(
            skel ^ skel_breaks,
            structure=np.ones((3, 3), dtype=bool)
            )

        num_removed = 0
        for label in range(1, n_bacteria_markers+1):
            bwnow = bacteria_markers == label
            if bwnow.sum() >= min_skel_length:
                continue
            bacteria_markers[bwnow] = 0
            num_removed += 1

        if num_removed == n_bacteria_markers:
            max_label += 1
            output_labels[region_mask] = max_label
            continue


        bacteria_markers, num_final = ndi.label(
            bacteria_markers>0,
            structure=np.ones((3,3), dtype=bool)
        )

        bacterianow = watershed(-dist, bacteria_markers, mask=region_mask)
        bacterianow[bacterianow > 0] += max_label
        output_labels[region_mask] = bacterianow[region_mask]
        max_label += num_final

    return output_labels


def relabel_bacteria(bacteria):
    """
    Relabel bacteria in dictionary to be
    sequential
    """
    relabelled = {}
    maxlabel = 0
    for well_label, bacnow in bacteria.items():
        bacnow[bacnow > 0] += maxlabel
        relabelled[well_label] = bacnow.copy()
        maxlabel = max(maxlabel, bacnow.max())
    return relabelled
