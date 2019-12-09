"""
Main function to run all detection
"""
import numpy as np
from mmhelper.detection.bacteria import (
    run_bacteria_detection,
)
from mmhelper.detection.modality import detect_phase
from mmhelper.detection.wells import (
    detect_wells,
)


def run_detection(image,
                  brightchannel,  # pylint: disable=unused-argument
                  debug="",
                  scale_factor=1,
                  ):
    """
    Runs all of the image detection

    Parameters
    ------
    image : ndarray (2D)
        The image to analyse
    brightchannel : Boolean, optional
        Whether the image has a bright channel or not (default : False)
    scale_factor : Float, optional
        Used to scale other parameters depending on the image
        magnification (default : 1)
    debug : Boolean, optional
        WWhether to add debugging outputs, save debug images with this
        basename (default : False)

    Returns
    ------
    detected_wellimg : ndarray (2D) of dtype int
        A labelled image showing the detected wells
    bacteria_image : ndarray (2D) of dtype int
        A labelled image showing the detected bacteria
    wellcoords : Dictionary
        Key is the well number and the value is an array of coordinates
        for the respective well
    bacteria : Dictionary
        The key is the well coordinates and the value is a labelled image
        of detected bacteria

    """
    debugwell = "%s_well_initial.png" % debug if debug else ""
    debugbacteria = "%s_bacteria_debug.png" % debug if debug else ""

    phase = detect_phase(image)

    wells, detected_wellimg, wellcoords = detect_wells(
        image,
        phase=phase,
        debug=debugwell,
        scale_factor=scale_factor)
    bacteria_image = np.zeros(image.shape, dtype="int16")
    if not wells:
        return detected_wellimg, bacteria_image, wellcoords, {}
    bacteria = run_bacteria_detection(
        wells, debug=debugbacteria, scale_factor=scale_factor)
    for key, value in bacteria.items():
        bacteria_image[wellcoords[key]] = value
    return detected_wellimg, bacteria_image, wellcoords, bacteria
