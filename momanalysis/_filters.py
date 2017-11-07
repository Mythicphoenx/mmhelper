import numpy as np
import scipy.ndimage as ndi

def gaussian_kernel(sigma, window_factor=4):
    # Method suggested on scipy cookbook
    """ Returns a normalized 2D gauss kernel array for convolutions """


    linrange = slice(-int(window_factor*sigma), int(window_factor*sigma)+1)
    x, y = np.mgrid[linrange,
                    linrange]
    g = np.exp(-(0.5*x**2/sigma**2 + 0.5*y**2/sigma**2))
    return g / g.sum()

    """
    # Method I used in the 1d gaussian filter
    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(window_factor * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0

    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
            tmp = np.exp(-0.5 * float(ii * ii) / sd)
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
            wsum += 2.0 * tmp
    for ii in range(2 * lw + 1):
            weights[ii] /= wsum
    """

def _eig2image(Lxx,Lxy,Lyy):
    """
    TODO: Acknowldege here
    """

    tmp = np.sqrt((Lxx - Lyy)**2 + 4*Lxy**2)
    v2x = 2*Lxy
    v2y = Lyy - Lxx + tmp

    # Normalize
    mag = np.sqrt(v2x**2 + v2y**2)
    i = (mag != 0)
    v2x[i] = v2x[i]/mag[i]
    v2y[i] = v2y[i]/mag[i]

    # The eigenvectors are orthogonal
    v1x = -v2y
    v1y = v2x

    # Compute the eigenvalues
    mu1 = 0.5*(Lxx + Lyy + tmp)
    mu2 = 0.5*(Lxx + Lyy - tmp)

    # Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
    check=np.abs(mu1)>np.abs(mu2)

    Lambda1=mu1
    Lambda1[check]=mu2[check]
    Lambda2=mu2
    Lambda2[check]=mu1[check]

    Ix=v1x
    Ix[check]=v2x[check]
    Iy=v1y
    Iy[check]=v2y[check]

    return Lambda1,Lambda2,Ix,Iy
