function [complete, uncertainty] = kNNImpute(data, k, varargin)
%kNNImpute imputes data by weighted mean of k nearest neighbour. Nearest
%neighbours are defined by known values and intervals of distribution of
%unknown values.
%
%Inputs:
%   data is n-by-m matrix of double. Missing values are denoted by NaN.
%   k is number of neighbours to use.
%   Name value pairs can be one of the following list
%       'interval' specifies the type of intervals to use:
%           'infinit' means that infinite intervals are used.
%           'minMax' means that the same intervals are used for each missed
%               values. This intervals are defined as 
%                   [min(data(:, i)), max(data(:, i))]
%           '3Sigma' means that  the same intervals are used for each missed
%               values. This intervals are defined as 
%                   [mean(data(:, i)) - 3 * std(data(:, i)), 
%                    mean(data(:, i)) + 3 * std(data(:, i))]
%           2-by-1 matrix A means that for each missing value in any
%               attribute the interval [A(1), A(2)] is used.
%           2-by-m matrix A means that for each missing value in i-th
%               attribute the interval [A(1, i), A(2, i)] is used.
%           n-by-m-by-2 3D array A means that for each missing value 
%               data(i ,j)indivdual interval [A(i, j, 1), A(i, j, 2)] is
%               used.
%           default value is '3Sigma'.
%       'kernel' is function for neighbour weights estimation. Weights are
%           function of distnace from the target point d divided by
%           distance from target point to furthest of k neighbours D. Value
%           can be one of the following 
%           (https://en.wikipedia.org/wiki/Kernel_(statistics)):
%           'uniform' is uniform weights: w = 1 / k;
%           'Triangular' is function w = 1 - (d / D);
%           'Epanechnikov' is function w = 1 - (d / D) ^ 2;
%           'Biweight' is function w = (1 - (d / D) ^ 2) ^ 2;
%           'Triweight' is function w = (1 - (d / D) ^ 2) ^ 3;
%           'Tricube' is function w = (1 - (d / D) ^ 3) ^ 3;
%           'Gaussian' is function w = exp( - 0.5 * (d / D) ^ 2);
%           'Cosine' is function w = cos(0.5 * pi * d / D);
%           'Logistic' is function w = 1 / (exp(d / D) + 2 + exp(- d / D));
%           'Sigmoid' is function w = 1 / (exp(d / D) + exp(- d / D));;
%           'Silverman' is function w = sin((d / D) / sqrt(2) + pi / 4);
%           function handler for function with header 
%               function w = fName(dD)
%           where dD = d / D;
%           default value is 'uniform'.
%
%Outputs:
%   complete is n-by-m matrix of double without missing values
%   uncertainty is n-by-m matrix of double with value of kNN uncertainty
%       for each value which is missed in data.

    % Sanity check of inputs
    if nargin<2 
        error('At least data and k must be specified');
    end
    if ~ismatrix(data) || ~isnumeric(data)
        error('Data must be numeric matrix');
    end
    nans = isnan(data);
    complete = data;
    [n, m] = size(data);
    uncertainty = zeros(n, m);
    if sum(nans(:)) == 0
        warning('There is no missing values in data. Nothing to do.');
        return;
    end
    if ~isnumeric(k) || k < 1
        error('k must be positive integer number');
    end

    wFunc = @uniform;
    interval = '3Sigma';
    
    % Parse varargin
    for i=1:2:length(varargin)
        if strcmpi(varargin{i}, 'interval')
            interval = varargin{i + 1};
        elseif strcmpi(varargin{i}, 'kernel')
            tmp = varargin{i + 1};
            if isa(tmp, 'function_handle')
                wFunc = tmp;
            else
                switch lower(varargin{i + 1})
                    case 'uniform'
                        wFunc = @uniform;
                    case 'triangular'
                        wFunc = @triangular;
                    case 'epanechnikov'
                        wFunc = @epanechnikov;
                    case 'biweight'
                        wFunc = @biweight;
                    case 'triweight'
                        wFunc = @triweight;
                    case 'tricube'
                        wFunc = @tricube;
                    case 'gaussian'
                        wFunc = @gaussian;
                    case 'cosine'
                        wFunc = @cosine;
                    case 'logistic'
                        wFunc = @logistic;
                    case 'sigmoid'
                        wFunc = @sigmoid;
                    case 'silverman'
                        wFunc = @silverman;
                    otherwise
                        error('Incorrect value for kernel argument');
                end
            end;
        else
            if ischar(varargin{i})
                error(['Wrong name of argument "', varargin{i}, '"']);
            else
                error(['Wrong name of argument "', num2str(varargin{i}), '"']);
            end
        end
    end
        
    dist = [];
    % Form intervals for specified option
    if ~isnumeric(interval)
        switch lower(interval)
            case 'infinit'
                % Unrestricted case
                dist = inifinitDist(data, nans);
            case 'minmax'
                % Form intervals
                interval = [min(data, [], 'omitnan'); max(data, [], 'omitnan')];
            case '3sigma'
                % Calculate mean and std
                tmp = mean(data, 'omitnan');
                st = std(data, 'omitnan');
                interval = [tmp - 3 * st; tmp + 3 * st];
            otherwise
                error('Incorrect value for interval argument');
        end
    end
    
    % Complete interval selection and distances caclulation
    if isempty(dist)
        if size(interval, 1) == 2 
            if size(interval, 2) == 1
                % 2-by-1 matrix A means that for each missing value in any
                % attribute the interval [A(1), A(2)] is used.
                interval = repmat(interval, 1, m);
            end
            % 2-by-m matrix A means that for each messing value in i-th
            % attribute the interval [A(1, i), A(2, i)] is used.
            dist = oneForAllDist(data, nans, interval);
        else
            % n-by-m-by-2 3D array A means that for each missing value
            % data(i ,j)indivdual interval [A(i, j, 1), A(i, j, 2)] is used.
            dist = individualDist(data, nans, interval);
        end
    end
    
    % We have distance matrix and can continue calculation
    % impute one gap per iteration
    for r = 1:n
        inds = find(nans(r, :));
        for c = inds
            dis = dist(:, r);
            % put Inf to instances with unknown value attribute c
            dis(nans(:,c)) = Inf;
            % Select k neighbours and calculate mean and variance
            [dis, ind] = sort(dis);
            [complete(r, c), uncertainty(r, c)]...
                = impute(data(:, c), dis(1:k), ind(1:k), wFunc);
        end
    end
end

%'uniform' is uniform weights: w = 1 / k;
function w = uniform(dD)
    w = ones(size(dD));
end

%'Triangular' is function w = 1 - (d / D);
function w = triangular(dD)
    w = 1 - dD;
end

%'Epanechnikov' is function w = 1 - (d / D) ^ 2;
function w = epanechnikov(dD)
    w = 1 - dD .^ 2;
end

%'Biweight' is function w = (1 - (d / D) ^ 2) ^ 2;
function w = biweight(dD)
    w = (1 - dD .^ 2) .^ 2;
end

%'Triweight' is function w = (1 - (d / D) ^ 2) ^ 3;
function w = triweight(dD)
    w = (1 - dD .^ 2) .^ 3;
end

%'Tricube' is function w = (1 - (d / D) ^ 3) ^ 3;
function w = tricube(dD)
    w = (1 - dD .^ 3) .^ 3;
end

%'Gaussian' is function w = exp( - 0.5 * (d / D) ^ 2);
function w = gaussian(dD)
    w = exp(- 0.5 * dD .^ 2);
end

%'Cosine' is function w = cos(0.5 * pi * d / D);
function w = cosine(dD)
    w = cos(0.5 * pi * dD);
end

%'Logistic' is function w = 1 / (exp(d / D) + 2 + exp(- d / D));
function w = logistic(dD)
    w = 1 / (exp(dD) + 2 + exp(- dD));
end

%'Sigmoid' is function w = 1 / (exp(d / D) + exp(- d / D));;
function w = sigmoid(dD)
    w = 1 / (exp(dD) + exp(- dD));
end

%'Silverman' is function w = sin((d / D) / sqrt(2) + pi / 4);
function w = silverman(dD)
    w = exp(dD / sqrt(2)) * sin(dD / sqrt(2) + pi / 4);
end

function  dist = inifinitDist(data, nans)
% Calculate dictance matrix without restriction
%Inputs:
%   data is n-by-m matrix of data points with one object per row
%   nans is n-by-m matrix of logical with true for missed value
%Outputs:
%   dist is n-by-n distance matrix

    n = size(data, 1);
    dist = Inf(n);
    for k = 1:n - 1
        for kk = k + 1:n
            % Select set of coordinates which is known for both objects
            ind = ~(nans(k, :) | nans(kk, :));
            dist(k, kk) = sqrt(sum((data(k, ind) - data(kk, ind)) .^ 2));
            dist(kk, k) = dist(k, kk);
        end
    end
end

function dist = oneForAllDist(data, nans, interval)
% Calculate dictance matrix in case when restriction are defined for
% attributes but not for objects
%Inputs:
%   data is n-by-m matrix of data points with one object per row
%   nans is n-by-m matrix of logical with true for missed value
%   intervals is 2-by-m matrix with one interval for each attribute p:
%       [interval(1, p); interval(2, p)].
%Outputs:
%   dist is n-by-n distance matrix
    n = size(data, 1);
    dist = Inf(n);
    for k = 1:n - 1
        v1 = data(k, :);
        for kk = k + 1:n
            v2 = data(kk, :);
            % Select attributes which are undefined for both objects and
            % set corresponding attributes to zero
            ind = nans(k, :) & nans(kk, :);
            v1(ind) = 0;
            v2(ind) = 0;
            % Select set of attributes which are known for one object only
            ind1 = nans(k, :) & ~nans(kk, :);
            ind2 = ~nans(k, :) & nans(kk, :);
            % Complete missed values in v1 as it is required
            v1(ind1) = v2(ind1);
            ind = v2 < interval(1, :);
            v1(ind1 & ind) = interval(1, ind1 & ind);
            ind = v2 > interval(2, :);
            v1(ind1 & ind) = interval(2, ind1 & ind);
            % Complete missed values in v2 as it is required
            v2(ind2) = v1(ind2);
            ind = v1 < interval(1, :);
            v2(ind2 & ind) = interval(1, ind2 & ind);
            ind = v1 > interval(2, :);
            v2(ind2 & ind) = interval(2, ind2 & ind);
            % Calculate distance
            dist(k, kk) = sqrt(sum((v1 - v2) .^ 2));
            dist(kk, k) = dist(k, kk);
        end
    end
end

function dist = individualDist(data, nans, interval)
% Calculate dictance matrix in case when restriction are defined for
% each missed value individually
%Inputs:
%   data is n-by-m matrix of data points with one object per row
%   nans is n-by-m matrix of logical with true for missed value
%   intervals is n-by-m-by-2 3D array with one interval for each missed
%   value in position i, j:
%       [interval(i, j, 1), interval(i, j, 2)]
%Outputs:
%   dist is n-by-n distance matrix
    n = size(data, 1);
    dist = Inf(n);
    for k = 1:n - 1
        v1 = data(k, :);
        for kk = k + 1:n
            v2 = data(kk, :);
            % Select attributes which are indefinit for both objects and
            % set corresponding attributes to middle value
            ind = nans(k, :) & nans(kk, :);
            % Set zero for all
            v1(ind) = 0;
            v2(ind) = 0;
            % Correct for intervals without overlapping
            ind1 = interval(kk, :, 2) < interval(k, :, 1);
            v1(ind & ind1) = interval(k, ind & ind1, 1);
            v2(ind & ind1) = interval(kk, ind & ind1, 2);
            ind1 = interval(k, :, 2) < interval(kk, :, 1);
            v1(ind & ind1) = interval(k, ind & ind1, 2);
            v2(ind & ind1) = interval(kk, ind & ind1, 1);
            % Select set of attributes which are known fro one object inly
            ind1 = nans(k, :) & ~nans(kk, :);
            ind2 = ~nans(k, :) & nans(kk, :);
            % Complete missed values in v1 as it is required
            v1(ind1) = v2(ind1);
            ind = v2 < interval(k, :, 1);
            v1(ind1 & ind) = interval(k, ind & ind1, 1);
            ind = v2 > interval(k, :, 2);
            v1(ind1 & ind) = interval(k, ind & ind1, 1);
            % Complete missed values in v2 as it is required
            v2(ind2) = v1(ind2);
            ind = v1 < interval(kk, :, 1);
            v2(ind2 & ind) = interval(kk, ind2 & ind, 1);
            ind = v1 > interval(kk, :, 2);
            v2(ind2 & ind) = interval(kk, ind2 & ind, 2);
            % Calculate distance
            dist(k, kk) = sqrt(sum((v1 - v2) .^ 2));
            dist(kk, k) = dist(k, kk);
        end
    end
end

function [val, unc] = impute(data, dis, ind, wFunc)
    % Normalise distances and calculate weights
    dis = wFunc(dis / max(dis));
    % Normalise weights
    dis = dis / sum(dis);
    % Calculate mean and variance
    val = dis' * data(ind);
    unc = (dis' * ((data(ind) - val) .^ 2)) / (1 - sum(dis .^ 2));
end
