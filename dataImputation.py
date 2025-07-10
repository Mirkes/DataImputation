# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:43:20 2025

This module contains three functions to work with data with gaps:
    degap - recursively removes feature or record with the most fraction of
            missed values
    kNNImpute - inpute data on base of k nearest methods method.
    svdWithGaps - impute data on base of singular value decomposition
            analogue of Principal components).

A detailed description of each function is provided in the functions themselves.

@author: Evgeny Mirkes, em322
"""
# pylint: disable=C0103

import numpy as np

def degap (data, col):
    '''
    degap removed the most incomplete rows and columns up to complete table
    forming. This method can be used only if data collected independently
    (rows/records/observations are observed independently) and data missed
    completely at random (property to be missed is independent of values of
    other attributes and missing value itself).

    Parameters
    ----------
    data : 2D ndarray or 2D array like structure
        n-by-m matrix of double. Missing values are denoted by NaN.
    col : 1D ndarray
        1-by-m vector with names (identifiers) of columns.

    Returns
    -------
    data : 2D ndarray
        The same data as was input argument but without removed rows and columns.
    col : 1D ndarray
        Vector with names (identifiers) of saved columns.

    '''

    # Convert data to ndarray
    data = np.asarray(data)

    # Check data
    if len(data.shape) != 2:
        raise ValueError('Array data must be 2D')
    if data.shape[1] != col.shape[0]:
        raise ValueError('Number of columns in data must be the same as number of elements in col')

    #Start degupping
    while True:
        mat = np.isnan(data)
        (n , m) = mat.shape
        # Calculate fraction of missed for records
        mr = np.sum(mat, axis=1) / m
        # Calculate fraction of missed for features
        mf = np.sum(mat, axis=0) / n
        # Check stop condition
        if sum(mr>0) + sum(mf>0) == 0:
            break

        #Search maximally gapped
        imr = np.argmax(mr)
        imf = np.argmax(mf)

        # What should be removed?
        if mr[imr] > mf[imf]:
            data = np.delete(data, imr, 0)
        else:
            data = np.delete(data, imf, 1)
            col = np.delete(col, imf)

    return (data, col)

def svdWithGaps(data, tol=0.05, tolConv=0.0001,
                interval='3Sigma', verbose=False):
    '''
    svdWithGaps returns complete version of matrix data

    svdWithGaps imputes data by decomposition of data matrix 'data' into
    singular vectors and later reconstruct all values.

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    tol : float, optional
        tol is tolerance level: stop of PCs calculation if the sum
        residual variances of all attributes is less than specified
        fraction of sum of variances of original data. The default is 0.05.
    tolConv : float, optional
        tolConv is tolerance level of PC search: PC is considered as
        found if 1 minus dot product of old PC and new PC is less than
        specified value. The default is 0.0001 that which corresponds to
        difference 0.81 of degree or 0.014 radian.
    interval : string or ndarray, optional
        interval specifies the type of intervals to use

            'infinite' : means that infinite intervals are used.

            'minMax' : means that the same intervals are used for each missed
            values. This intervals are defined as
                    [min(data[:, i]), max(data[:, i])]

            '3Sigma' means that the same intervals are used for each missed
            values. This intervals are defined as
                    [np.mean(data[:, i]) - 3 * np.std(data[:, i]),
                    np.mean(data[:, i]) + 3 * np.std(data[:, i])]

            2 element vector A : means that for each missing value in any
            attribute the interval [A[0], A[1]] is used.

            2-by-m matrix A : means that for each missing value in i-th
            attribute the interval [A[0, i], A[1, i]] is used.

            n-by-m-by-2 3D ndarray A : means that for each missing value
            data(i ,j) indivdual interval [A[i, j, 0], A[i, j, 1]] is used.
        The default is '3Sigma'.
    verbose : Boolean, optional
        verbos switchs on information of itersation accuracy: False to switch
        out and True to switch on. The default is False.

    Returns
    -------
    complete : 2D ndarray
        Completed (imputed) version of array data.

    '''


    # Sanity check of inputs
    err = False
    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            if not np.issubdtype(data.dtype, np.number):
                err = True
        else:
            err = True
    else:
        err = True
    if err:
        raise ValueError('Array "data" must be 2D ndarray of numerical type')

    # Calculate number of NaNs
    (n , m) = data.shape
    nans = np.isnan(data)
    if np.sum(nans) == 0:
        if verbose:
            print('There is no missing values in data. Nothing to do.')
        return data

    # Sanity check of arguments
    if (not isinstance(tol, float)) or (tol <= 0) or (tol >= 1):
        raise ValueError('Wrong value of "tol" argument: tol must be between 0 and 1')
    if (not isinstance(tolConv, float)) or (tolConv <= 0) or (tolConv >= 1):
        raise ValueError('Wrong value of "tolConv" argument: tol must be between 0 and 1')

    # Check intervals
    if isinstance(interval, str):
        interval = interval.lower()
        match interval:
            case 'infinite':
                # Unrestricted case
                restored = inifinitSVD(data, nans, tol, tolConv, verbose)
                complete = data.copy()
                complete[nans] = restored[nans]
                return complete

            case 'minmax':
                # Form min-max intervals
                #interval = [min(data, [], 'omitnan'); max(data, [], 'omitnan')];
                interval = np.concatenate((np.nanmin(data, axis=0, keepdims=True),
                                           np.nanmax(data, axis=0, keepdims=True)))

            case '3sigma':
                tmp = np.nanmean(data, axis=0, keepdims=True)
                st = np.nanstd(data, axis=0, keepdims=True)
                interval = np.concatenate((tmp - 3 * st, tmp + 3 * st))

            case _:
                raise ValueError('Incorrect value for interval argument')

    # Complete interval selection and distances caclulation
    if interval.ndim == 1:
        # 2 element vector A : means that for each missing value in any
        # attribute the interval [A[0], A[1]] is used.
        interval = np.transpose(np.tile(interval, (m, 1)))


    if interval.shape[0] == 2:
        # 2-by-m matrix A : means that for each missing value in i-th
        # attribute the interval [A[0, i], A[1, i]] is used.
        lo = np.tile(interval[0,:], (n, 1))
        hi = np.tile(interval[1,:], (n, 1))
    else:
        lo = np.squeeze(interval[:, :, 0])
        hi = np.squeeze(interval[:, :, 1])

    if (lo.shape[0] != n) or (lo.shape[1] != m):
        raise ValueError('Wrong size of specified intervals')

    # Final calculations
    restored = restrictedSVD(data, nans, tol, tolConv, lo, hi, verbose)
    complete = data.copy()
    complete[nans] = restored[nans]

    return complete

def inifinitSVD(data, nans, tol, tolConv, verbose):
    '''
    inifinitSVD implements unrestricted SVD for data with gaps. It is internal
    function and it does not control correctness of attributes. It is
    recommended to use svdWithGaps.

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    nans : 2D ndarray of boolean
        data is n-by-m matrix of boolean. Missing values are denoted by True,
        known values by False.
    tol : float
        tol is tolerance level: stop of PCs calculation if the sum
        residual variances of all attributes is less than specified
        fraction of sum of variances of original data. The default is 0.05.
    tolConv : float
        tolConv is tolerance level of PC search: PC is considered as
        found if 1 minus dot product of old PC and new PC is less than
        specified value. The default is 0.0001 that which corresponds to
        difference 0.81 of degree or 0.014 radian.
    verbose : boolean
        verbose is True to output accuracy of each itersation.

    Returns
    -------
    recovered : 2D ndarray
        Completed (imputed) version of array data.
    '''
    # Create array for results
    (n , m) = data.shape
    res = np.zeros_like(data)

    # Calculate residual variance
    base = np.sum(np.nanvar(data, axis=0))
    cutVar = base * tol
    comp = 1
    while True:
        # Create duplicate of data for fast calculation
        zData = data.copy()
        zData[nans] = 0
        # Initialise PC
        iters = 0
        b = np.nanmean(data, axis=0, keepdims=True)

        # Furthest from mean
        tmp = np.subtract(data, b)
        ind = np.argmax(np.nanvar(tmp, axis=1))
        y = tmp[ind, :]
        y[np.isnan(y)] = 0
        y = y / np.sqrt(np.sum(np.square(y)))

        # Create zero oldY is guarantee of non stop at the first itersation
        oldY = np.zeros((1, m))

        # Main loop of PC calculation
        while True:
            iters = iters + 1
            # Calculate/Recalculate x!
            tmp = np.tile(y, (n, 1))
            tmp[nans] = 0
            x = np.divide(np.sum(np.multiply(np.subtract(zData, b),
                    tmp), axis=1), np.sum(np.square(tmp), axis=1))

            # Check of convergence
            if 1 - abs(np.dot(oldY, y)) < tolConv:
                break

            oldY = y
            # Recalculate b!
            b = np.nanmean(data - np.expand_dims(x, axis=1) *
                           np.expand_dims(y, axis=0), axis=0, keepdims=True)
            # Recalculate y!
            tmp = np.transpose(np.tile(x, (m, 1)))
            tmp[nans] = 0
            y = np.divide(np.sum(np.multiply(np.subtract(zData, b), tmp),
                                 axis=0), np.sum(np.square(tmp), axis=0))
            y = y / np.sqrt(np.sum(np.square(y)))

        # Recalculate result and residuals
        tmp = b + np.expand_dims(x, axis=1) * np.expand_dims(y, axis=0)
        res = res + tmp
        data = data - tmp
        curr = np.sum(np.nanvar(data, axis=0))
        if verbose:
            print(f"Component {comp}: Fraction of unexplained variance is" +
                  f" {curr/base:.5f}, used itersations {iters}")
        # Check stop condition
        if curr < cutVar:
            break
        comp = comp + 1

    return res


def restrictedSVD(data, nans, tol, tolConv, lo, hi, verbose):
    '''
    restrictedSVD implements SVD for data with gaps with individual intervals
    for each missed value. It is internal function and it does not control
    correctness of attributes. It is recommended to use svdWithGaps.

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    nans : 2D ndarray of boolean
        data is n-by-m matrix of boolean. Missing values are denoted by True,
        known values by False.
    tol : float
        tol is tolerance level: stop of PCs calculation if the sum
        residual variances of all attributes is less than specified
        fraction of sum of variances of original data. The default is 0.05.
    tolConv : float
        tolConv is tolerance level of PC search: PC is considered as
        found if 1 minus dot product of old PC and new PC is less than
        specified value. The default is 0.0001 that which corresponds to
        difference 0.81 of degree or 0.014 radian.
    lo : 2D ndarray of float
        lo is n-by-m matrix of low boundaries of intervals.
    hi : 2D ndarray of float
        hi is n-by-m matrix of high boundaries of intervals.
    verbose : boolean
        verbose is True to output accuracy of each itersation.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''

    # Create array for results
    (n , m) = data.shape
    res = np.zeros_like(data)

    # Calculate residual variance
    base = np.sum(np.nanvar(data, axis=0))
    cutVar = base * tol
    comp = 1
    while True:
        # Create duplicate of data for fast calculation
        zData = data.copy()
        zData[nans] = 0
        # Initialise PC
        iters = 0
        xiters = 0
        b = np.nanmean(data, axis=0, keepdims=True)

        # Furthest from mean
        tmp = np.subtract(data, b)
        ind = np.argmax(np.nanvar(tmp, axis=1))
        y = tmp[ind, :]
        y[np.isnan(y)] = 0
        y = y / np.sqrt(np.sum(np.square(y)))

        # Create zero oldY is guarantee of non stop at the first itersation
        oldY = np.zeros((1, m))

        # Main loop of PC calculation
        while True:
            iters = iters + 1
            # Calculate/Recalculate x!
            # Solve unrestricted problem.
            tmp = np.tile(y, (n, 1))
            tmp[nans] = 0
            x = np.divide(np.sum(np.multiply(np.subtract(zData, b),
                    tmp), axis=1), np.sum(np.square(tmp), axis=1))

            # Search point closest to projection
            while True:
                xiters = xiters + 1
                tmp = b + np.expand_dims(x, axis=1) * np.expand_dims(y, axis=0)
                tmp[~nans] = data[~nans]
                ind = nans & (tmp < lo)
                cnt = np.sum(ind)
                tmp[ind] = lo[ind]
                ind = nans & (tmp > hi)
                cnt = cnt + np.sum(ind)
                tmp[ind] = hi[ind]

                # cnt is number of points that were corrected. If cnt == 0 then
                # we have all projections inside intervals and it is not
                # necessary to continue calculations!
                if cnt == 0:
                    break
                oldX = x
                # Recalculate x
                x = np.sum(np.multiply(np.subtract(tmp, b), y), axis=1)
                # x = sum(bsxfun(@times, bsxfun(@minus, tmp, b), y), 2);
                if np.sqrt(np.sum(np.square(oldX - x))) < tol:
                    break

            # Check of convergence
            if 1 - abs(np.dot(oldY, y)) < tolConv:
                break

            oldY = y
            # Recalculate b!
            b = np.nanmean(data - np.expand_dims(x, axis=1) *
                           np.expand_dims(y, axis=0), axis=0, keepdims=True)
            # Recalculate y!
            tmp = np.transpose(np.tile(x, (m, 1)))
            tmp[nans] = 0
            y = np.divide(np.sum(np.multiply(np.subtract(zData, b), tmp),
                                 axis=0), np.sum(np.square(tmp), axis=0))
            y = y / np.sqrt(np.sum(np.square(y)))

        # Recalculate result and residuals
        tmp = b + np.expand_dims(x, axis=1) * np.expand_dims(y, axis=0)
        res = res + tmp
        data = data - tmp
        curr = np.sum(np.nanvar(data, axis=0))
        if verbose:
            print(f"Component {comp}: " +
                  f"Fraction of unexplained variance is {curr/base:.5f}, " +
                  f"used itersations {iters}, used x itersations {xiters}")
        # Check stop condition
        if curr < cutVar:
            break
        comp = comp + 1

    return res

def kNNImpute(data, k=1, interval='3Sigma', kernel='uniform'):
    '''
    kNNImpute imputes data by weighted mean of k nearest neighbour. Nearest
    neighbours are defined by known values and intervals of distribution of
    unknown values.

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    k : int, optional
        Number of nearest neighbours. The default is 1.
    interval : string or ndarray, optional
        interval specifies the type of intervals to use

            'infinite' : means that infinite intervals are used.

            'minMax' : means that the same intervals are used for each missed
            values. This intervals are defined as
                    [min(data[:, i]), max(data[:, i])]

            '3Sigma' means that the same intervals are used for each missed
            values. This intervals are defined as
                    [np.mean(data[:, i]) - 3 * np.std(data[:, i]),
                    np.mean(data[:, i]) + 3 * np.std(data[:, i])]

            2 element vector A : means that for each missing value in any
            attribute the interval [A[0], A[1]] is used.

            2-by-m matrix A : means that for each missing value in i-th
            attribute the interval [A[0, i], A[1, i]] is used.

            n-by-m-by-2 3D ndarray A : means that for each missing value
            data(i ,j) indivdual interval [A[i, j, 0], A[i, j, 1]] is used.
        The default is '3Sigma'.
    kernel : string or function, optional
            function for neighbour weights estimation. Weights are
            function of distnace from the target point d divided by
            distance from target point to furthest of k neighbours D. Value
            can be one of the following
            (https://en.wikipedia.org/wiki/Kernel_(statistics))

            'uniform' is uniform weights: return 1 / k

            'Triangular' is function return 1 - (d / D)

            'Epanechnikov' is function return 1 - (d / D) ^ 2

            'Biweight' is function return (1 - (d / D) ^ 2) ^ 2

            'Triweight' is function return (1 - (d / D) ^ 2) ^ 3

            'Tricube' is function return (1 - (d / D) ^ 3) ^ 3

            'Gaussian' is function return exp( - 0.5 * (d / D) ^ 2)

            'Cosine' is function return cos(0.5 * pi * d / D)

            'Logistic' is function return 1 / (exp(d / D) + 2 + exp(- d / D))

            'Sigmoid' is function return 1 / (exp(d / D) + exp(- d / D))

            'Silverman' is function return sin((d / D) / sqrt(2) + pi / 4)

            fName is name of function with following definition
            def fName(dD):
            where fName is name of function, dD = d / D is parameter.
        The default is 'uniform'.

    Returns
    -------
    complete : 2D ndarray of float
        complete is an n-by-m matrix of double without missing values.
    uncertainty : 2D ndarray of float
        uncertainty is an n-by-m matrix of double with value of kNN uncertainty
        (unbiased variance of attribute of weighted k nearest values) for each
        value missing in the data. For k=1 this value is undefined and equal to
        NaN. For small k this value is not very accurate.

    '''
    # Sanity check of inputs
    err = False
    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            if not np.issubdtype(data.dtype, np.number):
                err = True
        else:
            err = True
    else:
        err = True
    if err:
        raise ValueError('Array "data" must be 2D ndarray of numerical type')
    nans = np.isnan(data)
    complete = data.copy()
    # Calculate number of NaNs
    (n , m) = data.shape
    nans = np.isnan(data)
    if np.sum(nans) == 0:
        # There is no missing values in data. Nothing to do.
        return data
    uncertainty = np.zeros_like(data)
    # Check value of k
    if not (isinstance(k, int) and k>0):
        raise ValueError('k must positive integer')
    # Weight function (kernel)
    if callable(kernel):
        # It is function
        wFunc = kernel
    else:
        wFunc = uniform
        match kernel.lower():
            case 'uniform':
                wFunc = uniform
            case 'triangular':
                wFunc = triangular
            case 'epanechnikov':
                wFunc = epanechnikov
            case 'biweight':
                wFunc = biweight
            case 'triweight':
                wFunc = triweight
            case 'tricube':
                wFunc = tricube
            case 'gaussian':
                wFunc = gaussian
            case 'cosine':
                wFunc = cosine
            case 'logistic':
                wFunc = logistic
            case 'sigmoid':
                wFunc = sigmoid
            case 'silverman':
                wFunc = silverman
            case _:
                raise ValueError('Incorrect value for kernel argument')

    # Check intervals, Form intervals for specified option
    dist = None
    if isinstance(interval, str):
        interval = interval.lower()
        match interval:
            case 'infinite':
                # Unrestricted case
                dist = infinitDist(data, nans)

            case 'minmax':
                # Form min-max intervals
                #interval = [min(data, [], 'omitnan'); max(data, [], 'omitnan')];
                interval = np.concatenate((np.nanmin(data, axis=0, keepdims=True),
                                           np.nanmax(data, axis=0, keepdims=True)))

            case '3sigma':
                tmp = np.nanmean(data, axis=0, keepdims=True)
                st = np.nanstd(data, axis=0, keepdims=True)
                interval = np.concatenate((tmp - 3 * st, tmp + 3 * st))

            case _:
                raise ValueError('Incorrect value for interval argument')

    # Complete interval selection and distances caclulation
    if dist is None:
        if interval.ndim == 1:
            # 2 element vector A : means that for each missing value in any
            # attribute the interval [A[0], A[1]] is used.
            interval = np.transpose(np.tile(interval, (m, 1)))

        if interval.shape[0] == 2:
            # 2-by-m matrix A : means that for each missing value in i-th
            # attribute the interval [A[0, i], A[1, i]] is used.
            dist = oneForAllDist(data, nans, interval)
        else:
            dist = individualDist(data, nans, interval)

    # We have distance matrix and can continue calculation
    # impute one gap per iteration
    for r in range(n):
        inds = np.nonzero(nans[r, :])[0]
        for c in inds:
            dis = dist[:, r]
            # put Inf to instances with unknown value attribute c
            dis[nans[:,c]] = np.inf
            # Select k neighbours and calculate mean and variance
            ind = np.argsort(dis)
            ind = ind[0:k]
            dis = dis[ind]
            (complete[r, c], uncertainty[r, c]) = impute(data[:, c], dis, ind, wFunc)

    return complete, uncertainty

def infinitDist(data, nans):
    '''
    Calculate dictance matrix without restriction

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    nans : 2D ndarray of boolean
        data is n-by-m matrix of boolean. Missing values are denoted by True,
        known values by False.

    Returns
    -------
    dist : 2D ndarray of float
        n-by-n matrix of distances between points.

    '''
    n = data.shape[0]
    dist = np.inf * np.ones((n, n))
    # Calculate distance for each pair of points
    for k in range(n - 1):
        for kk  in range(k + 1, n):
            # Select set of coordinates which is known for both objects
            ind = np.logical_not(nans[k, :] | nans[kk, :])
            dist[k, kk] = np.sqrt(np.sum(
                np.square(data[k, ind] - data[kk, ind])))
            dist[kk, k] = dist[k, kk]

    return dist

def oneForAllDist(data, nans, interval):
    '''
    Calculate dictance matrix in case when restriction are defined for
    attributes but not for objects

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    nans : 2D ndarray of boolean
        data is n-by-m matrix of boolean. Missing values are denoted by True,
        known values by False.
    interval : 2D ndarray of float
        2-by-m matrix A : means that for each missing value in i-th
        attribute the interval [A[0, i], A[1, i]] is used.
.
    Returns
    -------
    dist : 2D ndarray of float
        n-by-n matrix of distances between points.

    '''
    n = data.shape[0]
    dist = np.inf * np.ones((n, n))
    # Calculate distance for each pair of points
    for k in range(n - 1):
        for kk  in range(k + 1, n):
            v1 = data[k, :].copy()
            v2 = data[kk, :].copy()
            # Select attributes which are undefined for both objects and
            # set corresponding attributes to zero
            ind = nans[k, :] & nans[kk, :]
            v1[ind] = 0
            v2[ind] = 0
            # Select set of attributes which are known fro one object inly
            ind1 = nans[k, :] & ~nans[kk, :]
            ind2 = ~nans[k, :] & nans[kk, :]
            # Complete missed values in v1 as it is required
            v1[ind1] = v2[ind1]
            ind = v2 < interval[0, :]
            v1[ind1 & ind] = interval[0, ind1 & ind]
            ind = v2 > interval[1, :]
            v1[ind1 & ind] = interval[1, ind1 & ind]
            # Complete missed values in v2 as it is required
            v2[ind2] = v1[ind2]
            ind = v1 < interval[0, :]
            v2[ind2 & ind] = interval[0, ind2 & ind]
            ind = v1 > interval[1, :]
            v2[ind2 & ind] = interval[1, ind2 & ind]
            # Calculate distance
            dist[k, kk] = np.sqrt(np.sum(np.square(v1 - v2)))
            dist[kk, k] = dist[k, kk]

    return dist

def individualDist(data, nans, interval):
    '''
    Calculate dictance matrix in case when restriction are defined for
    each missed value individually

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    nans : 2D ndarray of boolean
        data is n-by-m matrix of boolean. Missing values are denoted by True,
        known values by False.
    interval : 2D ndarray of float
        n-by-m-by-2 3D array with one interval for each missed
        value in position i, j:  [interval(i, j, 1), interval(i, j, 2)]

    Returns
    -------
    dist : 2D ndarray of float
        n-by-n matrix of distances between points.

    '''
    n = data.shape[0]
    dist = np.inf * np.ones((n, n))
    # Calculate distance for each pair of points
    for k in range(n - 1):
        for kk  in range(k + 1, n):
            v1 = data[k, :].copy()
            v2 = data[kk, :].copy()
            # Select attributes which are undefined for both objects and
            # set corresponding attributes to zero
            ind = nans[k, :] & nans[kk, :]
            v1[ind] = 0
            v2[ind] = 0
            # Correct for intervals without overlapping
            ind1 = interval[kk, :, 1] < interval[k, :, 0]
            v1[ind & ind1] = interval[k, ind & ind1, 0]
            v2[ind & ind1] = interval[kk, ind & ind1, 1]
            ind1 = interval[k, :, 1] < interval[kk, :, 0]
            v1[ind & ind1] = interval[k, ind & ind1, 1]
            v2[ind & ind1] = interval[kk, ind & ind1, 0]
            # Select set of attributes which are known for one object inly
            ind1 = nans[k, :] & ~nans[kk, :]
            ind2 = ~nans[k, :] & nans[kk, :]
            # Complete missed values in v1 as it is required
            v1[ind1] = v2[ind1]
            ind = v2 < interval[k, :, 0]
            v1[ind1 & ind] = interval[k, ind & ind1, 0]
            ind = v2 > interval[k, :, 1]
            v1[ind1 & ind] = interval[k, ind & ind1, 1]
            # Complete missed values in v2 as it is required
            v2[ind2] = v1[ind2]
            ind = v1 < interval[kk, :, 0]
            v2[ind2 & ind] = interval[kk, ind2 & ind, 0]
            ind = v1 > interval[kk, :, 1]
            v2[ind2 & ind] = interval[kk, ind2 & ind, 1]
            # Calculate distance
            dist[k, kk] = np.sqrt(np.sum(np.square(v1 - v2)))
            dist[kk, k] = dist[k, kk]

    return dist

def impute(data, dis, ind, wFunc):
    '''
    impute calculated inputation on base of attribute values in data, distances
    to and indices of nearest neightbours, and weight function

    Parameters
    ----------
    data : 2D ndarray of float
        data is n-by-m matrix of double. Missing values are denoted by NaN.
        data is n-by-m matrix of double. Missing values are denoted by NaN.
    dis : 2D ndarray of float
        distances to the k nearest neighbours.
    ind : 2D ndarray of int
        indices of the k nearest neighbours in data.
    wFunc : function
        function to calcuate weights of nearest neighbours.

    Returns
    -------
    val : float
        value to impute gap.
    unc : float
        kNN uncertainty of imputation.

    '''
    # Normalise distances and calculate weights
    dis = wFunc(dis / max(dis))
    # Normalise weights
    dis = dis / np.sum(dis)
    # Calculate mean and variance
    val = np.dot(dis, data[ind])
    unc = np.dot(dis, ((data[ind] - val) ** 2)) / (1 - np.sum(dis ** 2))

    return val, unc

# Kernel functions

def uniform(dD):
    '''
    'uniform' is uniform weights: return 1 / k

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return np.ones_like(dD)

def triangular(dD):
    '''
    'Triangular' is function return 1 - (d / D)

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return 1 - dD

def epanechnikov(dD):
    '''
    'Epanechnikov' is function return 1 - (d / D) ^ 2

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return 1 - np.square(dD)

def biweight(dD):
    '''
    'Biweight' is function return (1 - (d / D) ^ 2) ^ 2

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return np.square(1 - np.square(dD))

def triweight(dD):
    '''
    'Triweight' is function return (1 - (d / D) ^ 2) ^ 3

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return (1 - np.square(dD)) ** 3

def tricube(dD):
    '''
    'Tricube' is function return (1 - (d / D) ^ 3) ^ 3

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return (1 - dD ** 3) ** 3
def gaussian(dD):
    '''
    'Gaussian' is function return exp( - 0.5 * (d / D) ^ 2)

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return np.exp(- 0.5 * np.square(dD))

def cosine(dD):
    '''
    'Cosine' is function return cos(0.5 * pi * d / D)

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return np.cos(0.5 * np.pi * dD)

def logistic(dD):
    '''
    'Logistic' is function return 1 / (exp(d / D) + 2 + exp(- d / D))

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return 1 / (np.exp(dD) + 2 + np.exp(- dD))

def sigmoid(dD):
    '''
    'Sigmoid' is function return 1 / (exp(d / D) + exp(- d / D))
    

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return 1 / (np.exp(dD) + np.exp(- dD))

def silverman(dD):
    '''
    'Silverman' is function return sin((d / D) / sqrt(2) + pi / 4)

    Parameters
    ----------
    dD : 1D ndarray of float
        normalised distances from neighbours to target point.

    Returns
    -------
    1D ndarray of float
        weights of neightbours.

    '''
    return np.exp(dD / np.sqrt(2)) * np.sin(dD / np.sqrt(2) + np.pi / 4)
