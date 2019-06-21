"""Input/Output functions

"""
import os
import glob
from collections import defaultdict
import numpy as np
import scipy.ndimage as ndi
import skimage.io as skio
from skimage.external import tifffile
from mmhelper.utility import logger

# =========================================================
# Data IO (wrappers around skimage functions)
# =========================================================


def input_folder(inputfiles, ext="[tT][iI][fF]"):
    """
    Determines if the input is a file or a folder.
    If it's a folder it returns a list of the tif files inside

    Parameters:
    ------
    inputfiles : List or str
        A list of files or a string which is the path to a folder containing
        the files
    ext : str, optional
        extension to match; NB using e.g. [tT][iI][fF] matches
        lower and uppercase extensions

    Returns
    ------
    A list of files to analyse
    """
    if isinstance(inputfiles, str):
        inputfiles = [inputfiles, ]
    # If input is a file-list, do nothing
    if os.path.isfile(inputfiles[0]):
        logger.debug("First input is a file; assuming file(s) given")
        return inputfiles
    logger.debug("First input not a file; assuming folder(s) given")
    # Should have paths
    filelist = []
    for path in inputfiles:
        if os.path.isdir(path):
            filelist.extend(glob.glob(
                os.path.join(path, "*.{}".format(ext))
            ))
        else:
            filelist.extend(glob.glob(path))

    return sorted(filelist)


def folder_all_areas(mainfolder):
    """
    Takes our recommended input of a folder with all time/area stamped images
    e.g. 'area_date_time_info.tif', subfolders them by area number

    Parameters
    ------
    mainfolder : str
        string which is a path to the folder holding the required files

    Returns
    ------
    area : dictionary
        keys are area numbers and the values are a list of files linked
        to that area
    """
    area = defaultdict(list)
    for filename in os.listdir(mainfolder):
        if filename.endswith(".tif"):
            basename = os.path.splitext(filename)[0]
            area_num = basename.split('_')[0]
            basename_split = basename.split('_')
            # this (above) may need to change slightly if our labelling system
            # changes
            if 'PI' not in basename_split:  # if it is a PI image don't include it in analysis
                area[area_num].append(os.path.join(mainfolder, filename))
    return area


def load_data(filenames):
    """Load a single file or sequence of files using skimage.io"""
    filenames = [filenames, ] if isinstance(filenames, str) else filenames
    loadfunc = tifffile.imread if all(f.lower().endswith("tif")
                                      for f in filenames) else skio.imread
    if len(filenames) > 1:
        return np.array([loadfunc(f) for f in filenames], dtype=float)
    elif len(filenames) == 1:
        return loadfunc(filenames[0]).astype(float)
    else:
        raise Exception("load_data received an empty list")

# =========================================================
# Sample data
# =========================================================


def _generate_well(
        width=11,
        height=200,
        num_bacteria_lambda=1,
        signal_to_noise=5,
        length_mean=20,
        length_std=10,
        bg_offset=50,
):
    """
    Generates a single well

    Parameters
    ------
    width : float, optional
        The width of the well to generate (default : 11)
    height : float, optional
        Height of the well to generate (default : 200)
    num_bacteria_lambda : float, optional
        Expectation of interval for np.random.poisson (default : 1)
    signal_to_noise : int, optional
        The signal to noise for the background (default : 1)
    length_mean : float, optional
        The mean length of bacteria to generate (default : 20)
    length_std : float, optional
        The standard deviation of bacteria lengths (default : 10)
    bg_offset : float, optional
        How far the background should be offset (default : 50)

    Returns
    ------
    ndarray (2D)
        A randomly generate well containing
    lbl : ndarray (2D)
        The well with the addition of the bacteria
    """
    # Background; choose 10 so should be no -ve pixels
    bkground = bg_offset + np.random.randn(width, height)

    # How many bacteria?
    num_bac = np.random.poisson(num_bacteria_lambda, 1)
    lbl = np.zeros((width, height), dtype="uint16")

    if num_bac == 0:
        return bkground, lbl

    # Distribute randomly along height, making sure there are no "collisions"
    signal = np.zeros((width, height), dtype=bool)
    nums = 0
    while nums < num_bac:  # for n in range(num_bac):
        length = max((length_mean + length_std * np.random.randn(), 5))
        pos = width / 2 + length / 2 + \
            (height - width - length) * np.random.rand()
        # Create skeleton
        x_wid = int(width / 2)
        y_wid = np.arange(pos - length / 2, pos + length / 2 + 1, dtype=int)

        if np.any(signal[x_wid, y_wid]):
            continue

        signal[x_wid, y_wid] = 1
        # Use distance transform to dilate the skeletons
        lbl[ndi.distance_transform_edt(~signal) < ((width / 2) - 2)] = nums + 1
        nums += 1
    signal = 1.0 * (lbl > 0)

    # Blur edges
    signal = ndi.gaussian_filter(signal, 2.0)
    signal -= signal.min()
    signal /= signal.max()

    return bkground - signal_to_noise * signal, lbl


def _generate_well_with_border(
        border_multiplier=1.2,
        border_width=2,
        **well_kwargs
):
    """
    Generates a well with a border

    Parameters
    ------
    border_multiplier : float, optional
        Multiple of SNR that border is darker
    border_width : float, optional
        Width of the border

    Returns
    ------
    signal : ndarray (2D)
        A randomly generated numpy array for a well
    lbl : ndarray (2D)
        The generated well the same as above but containing bacteria
    """
    well_width = well_kwargs['width']
    well_height = well_kwargs['height']
    width = 2 * border_width + well_width
    height = 2 * border_width + well_height
    bkgr = well_kwargs.get("bg_offset", 50) - border_multiplier * \
        well_kwargs.get("signal_to_noise", 5)
    signal = bkgr + np.random.randn(width, height)
    well_im, well_lbl = _generate_well(**well_kwargs)
    signal[border_width:border_width + well_width,
           border_width:border_width + well_height] = well_im
    lbl = np.zeros((width, height))
    lbl[border_width:border_width + well_width,
        border_width:border_width + well_height] = well_lbl
    return signal, lbl


def load_sample_well_data(
        num_wells=20,
        seed=None,
):
    """
    Generates simple sample well data, i.e. columns which sometimes contain a bacteria or so

    Returns:
        row of wells (as an image)
        row of labelled data ( as label image )
    """
    np.random.seed(seed)
    wells = []
    labels = []
    for well_num in range(num_wells):
        well, lbl = _generate_well()
        wells.append(well)
        labels.append(lbl)
    return np.vstack(wells).T, np.vstack(labels).T


def load_sample_full_frame(
        size=(600, 1000),
        well_vertical_offset=100,
        well_seperation=50,
        well_width=11,
        border_width=3,
        well_height=200,
        bg_offset=50,
        signal_to_noise=3,
        channel_dark_width=8,
        channel_light_width=8,
        channel_dark_line=-5,
        channel_light_line=5,
        seed=None,
        **well_kwargs
    ):
    """
    Creates a sample full frame

    Parameters
    ------
    size : tuple, optional
        Size of the whole frame (default : (600, 1000))
    well_vertical_offset : float, optional
        Distance between the bottom of the frame and the bottom of the wells (default : 100)
    well_seperation : float, optional
        Distance between wells (default : 50)
    well_width : float, optional
        The width of each well (default : 11)
    border_width : float, optional
        The width of the well borders (default : 3)
    well_height : float, optional
        Height of the wells (default : 200)
    bg_offset : float, optional
        How offset the background is (default : 50)
    signal_to_noise : int, optional
        The signal to noise for the background (default : 1)
    channel_dark_width : float, optional
        Width of the dark channel at the bottom of the wells (default : 8)
    channel_light_width : float, optional
        Width of the light channel at the bottom of the wells (default : 8)
    channel_dark_line=-5,
    channel_light_line=5,
    seed=None,

    Returns
    ------
    image : ndarray(2D)
        An array of the final created image
    lbl_wells : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    lbl_bacteria : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    """
    np.random.seed(seed)

    # Create a bigger image so that rotation doesn't cause any edge effects;
    sizebig = [int(1.4 * max(size)) for d in range(2)]

    # Generate the bg image
    image = bg_offset + np.random.randn(*sizebig)
    lbl_wells = np.zeros(sizebig, dtype='uint16')
    lbl_bacteria = np.zeros(sizebig, dtype='uint16')

    bords = [(sbig - s) // 2 for sbig, s in zip(sizebig, size)]
    # Add in the wells
    for lab, x_num in enumerate(
            range(
                10, sizebig[0] - well_width, well_seperation +
                well_width),
            start=1):
        well_im = _generate_well_with_border(
            border_width=border_width,
            width=well_width,
            height=well_height,
            bg_offset=bg_offset + signal_to_noise,
            signal_to_noise=signal_to_noise,
        )[0]
        image[well_vertical_offset +
              bords[0]:well_vertical_offset +
              well_height +
              2 *
              border_width +
              bords[0], x_num:x_num +
              well_width +
              2 *
              border_width] = well_im.T
        lbl_wells[well_vertical_offset +
                  bords[0]:well_vertical_offset +
                  well_height +
                  2 *
                  border_width +
                  bords[0], x_num:x_num +
                  well_width +
                  2 *
                  border_width] = lab
    # Add in the channel; a dark line then bright

    image[well_vertical_offset + bords[0]:well_vertical_offset +
          channel_dark_width + bords[0]] += channel_dark_line
    image[well_vertical_offset + bords[0]:well_vertical_offset -
          channel_light_width + bords[0]:-1] += channel_light_line

    # Invert Y
    image = image[::-1, :]
    lbl_wells = lbl_wells[::-1, :]

    # Now pull out the properly sized image!
    slices = [slice(b, -b) for b in bords]
    image = image[slices]
    lbl_wells = lbl_wells[slices]
    lbl_bacteria = lbl_bacteria[slices]

    image -= image.min()
    image /= image.max()

    return image, lbl_wells, lbl_bacteria


def split_fluorescence(data, num_fluo=1):
    """Splits the into brightfield and fluorescence

    Parameters
    ------
    data : ndarray
        An array containing all of the data brightfield and (if required) fluorescence
    num_fluo : int, optional
        How many fluorescent channels there are per brightfield image (default : 1)

    Returns
    ------
    newdata : ndarray
        The brightfield images
    fluo_data : ndarray or list
        The more than 1 fluo channel, these are returned as a list of ndarrays
        otherwise just an array
    """
    # this is currently causing issues - opening them as [([fluo_type1_t1, fluo_type1_t2)]
    # need it to be temporal e.g. [(fluo_type1_t1),(fluo_type1_t2)] - change here or line 92 where
    # fluo_data is accessed by time
    fluo_data = []
    if data.ndim == 4:
        newdata = data[:, 0, :, :]
        fluo_data.append(data[:, 1, :, :])
        return newdata, fluo_data
    data = [data[n::num_fluo + 1] for n in range(num_fluo + 1)]
    newdata = data[0]
    fluo_data = data[1:]
    return newdata, fluo_data
