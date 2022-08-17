import numpy as np
import scipy
import os
import warnings
from KDEpy import FFTKDE

def comp_pca(usnap, n_trunc, detrend=True):
    """Perform Principal Component Analysis on data. 

    Parameters
    ----------
    usnap : array
        Data array, size (n_pts, n_dim).
    n_trunc : integer
        The number of PCA dimensions to be retained.
    detrend : boolean, optional
        Whether or not to deduct the mean from the data.

    Returns
    -------
    lam : array
        The first n_trunc PCA eigenvalues.
    phi : array
        The first n_trunc PCA eigenfunctions.
    usnap_mean : array
        The mean of the data.

    """
    m = usnap.shape[0]
    usnap_mean = np.mean(usnap, axis=0)
    if detrend:
        usnap = usnap - usnap_mean[np.newaxis,:]
    R = np.einsum('ij,kj->ik', usnap, usnap) / m
    lam, phi = np.linalg.eigh(R)
    idx = lam.argsort()[::-1]
    lam = lam[idx]
    phi = phi[:,idx] / np.sqrt(m)
    lam_inv = np.diag(1.0/np.sqrt(lam))
    psi = np.dot(usnap.T, np.dot(phi,lam_inv))
    return lam[0:n_trunc], psi[:,0:n_trunc], usnap_mean


def grid_nint(pts, fmu, ngrid):
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    fmug = fmu.reshape( (ngrid,)*ndim ).T
    axes = [ grd[ (i,) + (0,)*i + (slice(None),) + (0,)*(ndim-i-1) ] \
             for i in range(ndim) ]
    I = fmug
    for ii in range(ndim):
        I = np.trapz(I, x=axes[ii], axis=0)
    return I


def fix_dim_gmm(gmm, matrix_type="covariance"):
 
    if matrix_type == "covariance":
        matrix = gmm.covariances_
    elif matrix_type == "precisions":
        matrix = gmm.precisions_
    elif matrix_type == "precisions_cholesky":
        matrix = gmm.precisions_cholesky_

    n_components, n_features = gmm.means_.shape
    m = np.empty((n_components, n_features, n_features))

    for n in range(gmm.n_components):
        if gmm.covariance_type == "full":
            m[n] = matrix[n]
        elif gmm.covariance_type == "tied":
            m[n] = matrix
        elif gmm.covariance_type == "diag":
            m[n] = np.diag(matrix[n])
        elif gmm.covariance_type == "spherical":
            m[n] = np.eye(gmm.means_.shape[1]) * matrix[n]

    return m


def process_parameters(dim, mean, cov):
    """
    Infer dimensionality from mean or covariance matrix, ensure that
    mean and covariance are full vector resp. matrix.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if dim == 1:
        mean.shape = (1,)
        cov.shape = (1, 1)

    if mean.ndim != 1 or mean.shape[0] != dim:
        raise ValueError("Array 'mean' must be vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    else:
        if cov.shape != (dim, dim):
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                                 " but cov.ndim = %d" % cov.ndim)

    return mean, cov

def custom_KDE(data, weights=None, bw=None):
    data = data.flatten()
    if bw is None:
        try:
            sc = scipy.stats.gaussian_kde(data, weights=weights)
            bw = np.sqrt(sc.covariance).flatten()
            # Ensure that bw is a scalar value
            if np.size(bw) == 1:
                bw = np.asscalar(bw)
            else:
                raise ValueError("The bw must be a number.")
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0           
    return FFTKDE(bw=bw).fit(data, weights)
