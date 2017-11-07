"""Input/Output functions

"""
import numpy as np
import scipy.ndimage as ndi
import skimage.io as skio
from skimage.external import tifffile
import os
from collections import defaultdict
import shutil

##=========================================================
## Data IO (wrappers around skimage functions)
##=========================================================

def input_folder(inputfiles):
    """determines if the input is a file or a folder.
    If it's a folder it returns a list of the tif files inside
    """
    original_dir = os.getcwd()
    image_list = []
    for f in inputfiles:
        f = os.path.realpath(f)
        if os.path.isfile(f)==True:
            return inputfiles
        if os.path.isdir(f)==True:
            os.chdir(f)
            for file in os.listdir(f):
                if file.endswith(".tif"):
                    image = os.path.realpath(file)
                    image_list.append(image)
    os.chdir(original_dir)
    return image_list

def folder_all_areas(mainfolder):
    """takes our recommended input of a folder with all time/area stamped images e.g. 'area_date_time_info.tif',
    subfolders them by area number"""
    original_dir = os.getcwd()
    area = defaultdict(list)
    arealist = []
    mainfolder = mainfolder[0]
    os.chdir(mainfolder)
    for file in os.listdir(mainfolder):
        if file.endswith(".tif"):
            basename = os.path.splitext(file)[0]            
            area_num = basename.split('_')[0]
            basename_split = basename.split('_')            
            #this (above) may need to change slightly if our labelling system changes
            if not basename == 'Thumbs': #windows produces a Thumbs.db file so this stops this interfering
                if not 'PI' in basename_split: #if it is a PI image don't include it in analysis
                    area[area_num].append(file)
    return area


def load_data(filenames):
    """Load a single file or sequence of files using skimage.io"""
    filenames = [filenames,] if isinstance(filenames, str) else filenames
    loadfunc = tifffile.imread if all(f.lower().endswith("tif")
        for f in filenames) else skio.imread
    if len(filenames) > 1:
        return np.array([ loadfunc(f) for f in filenames ], dtype=float)
    elif len(filenames) == 1:
        return loadfunc(filenames[0]).astype(float)
    else:
        raise Exception("load_data received an empty list")

##=========================================================
## Sample data
##=========================================================

def _generate_well(
        width=11,
        height=200,
        num_bacteria_lambda=1,
        signal_to_noise=5,
        length_mean=20,
        length_std=10,
        bg_offset = 50,
        ):
    """
    Generate a single well
    """
    # Background; choose 10 so should be no -ve pixels
    bg = bg_offset + np.random.randn(width, height)

    # How many bacteria?
    N_bac = np.random.poisson(num_bacteria_lambda, 1)
    lbl = np.zeros((width, height), dtype="uint16")

    if N_bac == 0:
        return bg, lbl

    # Distribute randomly along height, making sure there are no "collisions"
    signal = np.zeros((width, height), dtype=bool)
    n = 0
    while n < N_bac: # for n in range(N_bac):
        length = max(( length_mean + length_std * np.random.randn(), 5))
        pos = width/2 + length/2 + (height-width-length)*np.random.rand()
        # Create skeleton
        x = int(width/2)
        y = np.arange(pos-length/2, pos+length/2+1, dtype=int)

        if np.any(signal[x, y]):
            continue

        signal[x, y] = 1
        # Use distance transform to dilate the skeletons
        lbl[ ndi.distance_transform_edt(~signal) < ((width/2)-2)] = n+1
        n += 1
    signal = 1.0*(lbl > 0)

    # Blur edges
    signal = ndi.gaussian_filter(signal, 2.0)
    signal -= signal.min()
    signal /= signal.max()

    return bg - signal_to_noise * signal, lbl


def _generate_well_with_border(
        border_multiplier = 1.2, # Multiple of SNR that border is darker
        border_width=2,
        **well_kwargs
        ):
    well_width = well_kwargs['width']
    well_height = well_kwargs['height']
    width = 2*border_width + well_width
    height = 2*border_width + well_height
    bg = well_kwargs.get("bg_offset", 50) - border_multiplier * well_kwargs.get("signal_to_noise", 5)
    signal = bg + np.random.randn(width, height)
    well_im, well_lbl = _generate_well(**well_kwargs)
    signal[border_width:border_width + well_width, border_width:border_width+well_height] = well_im
    lbl = np.zeros((width, height))
    lbl[border_width:border_width+well_width, border_width:border_width+well_height] = well_lbl
    return signal, lbl


def load_sample_well_data(
        num_wells = 20,
        seed = None,
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
    for n in range(num_wells):
        well, lbl = _generate_well()
        wells.append(well)
        labels.append(lbl)
    return np.vstack(wells).T, np.vstack(labels).T


def load_sample_full_frame(
        size=(600, 1000),
        well_vertical_offset=100,
        well_seperation=50,
        well_width = 11,
        border_width = 3,
        well_height = 200,
        bg_offset = 50,
        signal_to_noise = 3,
        channel_dark_width = 8,
        channel_light_width = 8,
        channel_dark_line = -5,
        channel_light_line = 5,
        seed = None,
        **well_kwargs
        ):
    np.random.seed(seed)

    # Create a bigger image so that rotation doesn't cause any edge effects;
    sizebig = [ int(1.4*max(size)) for d in range(2)]

    # Generate the bg image
    image = bg_offset + np.random.randn(*sizebig)
    lbl_wells = np.zeros(sizebig, dtype='uint16')
    lbl_bacteria = np.zeros(sizebig, dtype='uint16')

    bords = [ (sbig - s)//2 for sbig, s in zip(sizebig, size)]
    # Add in the wells
    for lab, x in enumerate(range(10, sizebig[0]-well_width, well_seperation + well_width), start=1):
        well_im, well_lbl = _generate_well_with_border(
            border_width=border_width,
            width=well_width,
            height=well_height,
            bg_offset=bg_offset+signal_to_noise,
            signal_to_noise =signal_to_noise,
            )
        image[well_vertical_offset+bords[0]:well_vertical_offset+well_height+2*border_width + bords[0],
              x:x+well_width+2*border_width] = well_im.T
        lbl_wells[well_vertical_offset+bords[0]:well_vertical_offset+well_height+2*border_width + bords[0],
              x:x+well_width+2*border_width] = lab
    # Add in the channel; a dark line then bright

    image[well_vertical_offset+bords[0]:well_vertical_offset+channel_dark_width+bords[0]] += channel_dark_line
    image[well_vertical_offset+bords[0]:well_vertical_offset-channel_light_width+bords[0]:-1] += channel_light_line

    # Invert Y
    image = image[::-1, :]
    lbl_wells = lbl_wells[::-1, :]

    # Now pull out the properly sized image!
    slices = [ slice(b, -b) for b in bords]
    image = image[slices]
    lbl_wells = lbl_wells[slices]
    lbl_bacteria = lbl_bacteria[slices]

    image -= image.min()
    image /= image.max()

    return image, lbl_wells, lbl_bacteria

def split_fluorescence(data):
    newdata = data[::2]
    fluodata = data[1::2]

    return newdata, fluodata

