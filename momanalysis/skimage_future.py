#
# FILE        : skimage_future.py
# CREATED     : 27/09/16 13:40:48
# AUTHOR      : J. Metz <metz.jp@gmail.com>
# DESCRIPTION : Skimage functions that are unavailable in stable
#               release but present in development version
#

import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def _frangi_hessian_common_filter(image, scale_range, scale_step,
                                  beta1, beta2):
    """This is an intermediate function for Frangi and Hessian filters.
    Shares the common code for Frangi and Hessian functions.
    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta1 : float, optional
        Frangi correction constant.
    beta2 : float, optional
        Frangi correction constant.
    Returns
    -------
    filtered_list : list
        List of pre-filtered images.
    """
    # Import has to be here due to circular import error

    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    beta1 = 2 * beta1 ** 2  # `ad` in imagej code L137
    beta2 = 2 * beta2 ** 2  # `bd` in imagej code L138

    filtered_array = np.zeros(sigmas.shape + image.shape)
    lambdas_array = np.zeros(sigmas.shape + image.shape)

    # Filtering for all sigmas
    for i, sigma in enumerate(sigmas):
        # Make 2D hessian
        # Whole routine for this in imagej L158-1282, but pre-calculates
        # gaussian-blurring, here thats part of hessian_matrix
        (Dxx, Dxy, Dyy) = hessian_matrix(image, sigma)


        # Correct for scale
        Dxx = (sigma ** 2) * Dxx
        Dxy = (sigma ** 2) * Dxy
        Dyy = (sigma ** 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        # NOTE: According to _hessian_matrix_det code, this is only valid for sigma >= 3!!
        (lambda1, lambda2) = hessian_matrix_eigvals(Dxx, Dxy, Dyy)

        # Compute some similarity measures
        lambda1[lambda1 == 0] = 1e-10
        rb = (lambda2 / lambda1) ** 2
        s2 = lambda1 ** 2 + lambda2 ** 2

        # Compute the output image
        filtered = np.exp(-rb / beta1) * (np.ones(np.shape(image)) -
                                          np.exp(-s2 / beta2))

        # Store the results in 3D matrices
        filtered_array[i] = filtered
        lambdas_array[i] = lambda1
    return filtered_array, lambdas_array


def frangi(image, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15,
           black_ridges=True):
    """Filter an image with the Frangi filter.
    This filter can be used to detect continuous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.
    Calculates the eigenvectors of the Hessian to compute the similarity of
    an image region to vessels, according to the method described in _[1].
    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta1 : float, optional
        Frangi correction constant.
    beta2 : float, optional
        Frangi correction constant.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).
    Notes
    -----
    Written by Marc Schrijver, 2/11/2001
    Re-Written by D. J. Kroon University of Twente (May 2009)
    References
    ----------
    .. [1] A. Frangi, W. Niessen, K. Vincken, and M. Viergever. "Multiscale
           vessel enhancement filtering," In LNCS, vol. 1496, pages 130-137,
           Germany, 1998. Springer-Verlag.
    .. [2] Kroon, D.J.: Hessian based Frangi vesselness filter.
    .. [3] http://mplab.ucsd.edu/tutorials/gabor.pdf.
    """
    filtered, lambdas = _frangi_hessian_common_filter(image,
                                                      scale_range, scale_step,
                                                      beta1, beta2)
    if black_ridges:
        filtered[lambdas < 0] = 0
    else:
        filtered[lambdas > 0] = 0

    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    return np.max(filtered, axis=0)


def threshold_li(image):
    """Return threshold value based on adaptation of Li's Minimum Cross Entropy method.

    Parameters
    ----------
    image : array
        Input image.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels intensities more than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           http://citeseer.ist.psu.edu/sezgin04survey.html
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_li(image)
    >>> binary = image > thresh
    """
    # Copy to ensure input image is not modified
    image = image.copy()
    # Requires positive image (because of log(mean))
    immin = np.min(image)
    image -= immin
    imrange = np.max(image)
    tolerance = 0.5 * imrange / 256

    # Calculate the mean gray-level
    mean = np.mean(image)

    # Initial estimate
    new_thresh = mean
    old_thresh = new_thresh + 2 * tolerance

    # In case while starts off false
    threshold = new_thresh
    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    while abs(new_thresh - old_thresh) > tolerance:
        old_thresh = new_thresh
        threshold = old_thresh + tolerance   # range
        # Calculate the means of background and object pixels
        mean_back = image[image <= threshold].mean()
        mean_obj = image[image > threshold].mean()

        temp = (mean_back - mean_obj) / (np.log(mean_back) - np.log(mean_obj))

        if temp < 0:
            new_thresh = temp - tolerance
        else:
            new_thresh = temp + tolerance

    return threshold + immin
