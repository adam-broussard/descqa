from __future__ import division, print_function
import numpy as np
from scipy.stats import chi2
from past import L2Diff, L1Diff, KS_test, write_summary_details


def get_subvolume_indices(x, y, z, box_size, n_side):
    side_size = box_size/n_side
    return np.ravel_multi_index(np.floor(np.vstack((x, y, z))/side_size).astype(int), (n_side,)*3, 'wrap')


def jackknife(data, jack_indices, n_jack, func, full_args=(), full_kwargs={}, jack_args=(), jack_kwargs={}):
    if len(data) != len(jack_indices):
        raise ValueError('`data` and `jack_indices` must have the same length')
    if not np.in1d(jack_indices, np.arange(n_jack)).all():
        raise ValueError('`jack_indices` must be an array of int between 0 to n_jack-1')

    full = np.array(func(data, *full_args, **full_kwargs), dtype=np.float)

    jack = []
    for i in xrange(n_jack):
        jack.append(func(data[jack_indices != i], *jack_args, **jack_kwargs))
    jack = np.array(jack, dtype=np.float)

    bias = (jack.mean(axis=0) - full)*(n_jack-1)
    return full-bias, bias, np.cov(jack, rowvar=False, bias=True)*float(n_jack-1)


def chisq(difference, covariance):
    d = np.asarray(difference)
    cov = np.asarray(covariance)
    if cov.ndim == 1:
        cov = np.diag(cov)
    return np.dot(d, np.dot(np.linalg.inv(cov), d))


def chisq_threshold(dof, conf_level=0.95):
    return chi2.isf(1.0-conf_level, dof)


def Lp_norm(difference, p=2.0):
    d = np.asarray(difference)
    d **= p
    return d.sum() ** (1.0/p)

def AD_statistic(y1, y2, threshold):
    '''
    Calculate Anderson-Darling statistic from two CDFs;
    y1, y2: CDF y-values of the two distribution; they should have the 
    same x-axis.
    '''
    # compute CvM statistic
    inv_weight = (y2*(1-y2))[:-1]
    # remove infinities in the weight function
    mask = inv_weight==0
    inv_weight[mask] = 1
    ads = np.sum(((y2 - y1)[:-1])**2*(y2[1:]-y2[:-1])/inv_weight)

    if ads<threshold:
        success = True
    else:
        success = False
    
    return ads, success
