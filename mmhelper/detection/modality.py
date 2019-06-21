"""
Determines whether phase or not
"""
import scipy.stats as scistats

def detect_phase(image):
    """
    Determines if the image being analysed is brightfield or phase.

    Parameters
    ------
    image : ndarray (2D)
        The image to analyse

    Returns
    ------
    Boolean
        True for phase or False for brightfield
    """
    #return scistats.skew(image.flat) > 0

    # Switch to using alternative skewness measure,
    # Pearson's median skewness:
    # which is techically 3*(mean-median)/std
    # return 3*(np.mean(image) - np.median(image)) > np.std(image)

    # Pearsons second skewness also didn't work... resorting to
    # empirically determined threshold on skewness :/

    return scistats.skew(image.flat) > 0.25
