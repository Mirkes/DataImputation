function complete = svdWithGaps(data, varargin)
%svdWithGaps imputes data by decomposition of data matrix 'data' into
%singular vectors and later reconstruct all values.
%
%Inputs:
%   data is n-by-m matrix of double. Missing values are denoted by NaN.
%   Name value pairs can be one of the following list
%       'tol' is tolerance level: stop of PCs calculation if the sum
%           residual variances of all attributes is less than specified
%           fraction of sum of variances of original data.
%           Default value is 0.05.
%       'tolConv' is tolerance level of PC search: PC is considered as
%           found if 1 minus dot product of old PC and new PC is less than
%           specified value.
%           Default value is 0.0001 which corresponds to difference 0.81 of
%           degree.
%       'interval' specifies the type of intervals to use:
%           'infinite' means that infinite intervals are used.
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
%       'verbose' switchs on information of iteration accuracy: 0 to switch
%           out (default) and any other value to switch on.
%
%Outputs:
%   complete is n-by-m matrix of double without missing values

    % Sanity check of inputs
    if nargin < 1 
        error('At least data must be specified');
    end
    if ~ismatrix(data) || ~isnumeric(data)
        error('Data must be numeric matrix');
    end
    nans = isnan(data);
    [n, m] = size(data);
    if sum(nans(:)) == 0
        warning('There is no missing values in data. Nothing to do.');
        complete = data;
        return;
    end
    
    tol = 0.05;
    tolConv = 0.0001;
    interval = '3Sigma';
    verbose = false;
    
    % Parse varargin
    for i=1:2:length(varargin)
        if strcmpi(varargin{i}, 'interval')
            interval = varargin{i + 1};
        elseif strcmpi(varargin{i}, 'tol')
            tol = varargin{i + 1};
        elseif strcmpi(varargin{i}, 'tolConv')
            tolConv = varargin{i + 1};
        elseif strcmpi(varargin{i}, 'verbose')
            verbose = varargin{i + 1} ~= 0;
        else
            if ischar(varargin{i})
                error(['Wrong name of argument "', varargin{i}, '"']);
            else
                error(['Wrong name of argument "', num2str(varargin{i}), '"']);
            end
        end
    end
        
    % Sanity check of arguments
    if ~isnumeric(tol) || tol <= 0 || tol > 1
        error('Wrong value of "tol" argument: tol must be between 0 and 1');
    end
    % Sanity check of arguments
    if ~isnumeric(tolConv) || tolConv <= 0 || tolConv > 1
        error('Wrong value of "tolConv" argument: tolConv must be between 0 and 1');
    end
    
    restored = [];
    % Form intervals for specified option
    if ~isnumeric(interval)
        switch lower(interval)
            case 'infinite'
                % Unrestricted case
                restored = inifinitSVD(data, find(nans), tol, tolConv, verbose);
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
    if isempty(restored)
        if size(interval, 1) == 2 
            if size(interval, 2) == 1
                % 2-by-1 matrix A means that for each missing value in any
                % attribute the interval [A(1), A(2)] is used.
                interval = repmat(interval, 1, m);
            end
            % 2-by-m matrix A means that for each messing value in i-th
            % attribute the interval [A(1, i), A(2, i)] is used.
            lo = repmat(interval(1, :), n, 1);
            hi = repmat(interval(2, :), n, 1);
        else
            % n-by-m-by-2 3D array A means that for each missing value
            % data(i ,j)indivdual interval [A(i, j, 1), A(i, j, 2)] is used.
            lo = interval(:, :, 1);
            hi = interval(:, :, 2);
        end
        if size(lo, 1) ~= n || size(lo, 2) ~= m
            error('Wrong size of specified intervals');
        end
        restored = restrictedSVD(data, nans, tol, tolConv, lo, hi, verbose);
    end
    complete = data;
    complete(nans) = restored(nans);
end

function res = inifinitSVD(data, nans, tol, tolConv, verbose)
%inifinitSVD implements unrestricted SVD for data with gaps.
%Inputs:
%   data is n-by-m matrix of data
%   nans is vector of indices corresponds to messed value.
%   tol is tolerance level to defile required number of PCs
%   tolConv is tolerance level to define PC convergence
%   verbose is indicator to output accuracy of each iteration
%Outputs:
%   res is n-by-m matrix of restored data.
    [n, m] = size(data);
    res = zeros(n, m);
    % Calculate residual variance 
    base = sum(var(data, 'omitnan'));
    cutVar = base * tol;
    comp = 1;
    while true
        % Create duplicate of data for fast calculation
        zData = data;
        zData(nans) = 0;
        % Initialise PC
        iter = 0;
        b = mean(data, 'omitnan');
        % Furthest from mean
        tmp = bsxfun(@minus, data, b);
        [~, ind] = max(var(tmp, 0, 2, 'omitnan'));
        y = tmp(ind, :);
        y(isnan(y)) = 0;
        y = y ./ sqrt(sum(y .^ 2));
        % Zero oldY is guarantee of non stop at the first iteration
        oldY = zeros(1, m);
        %Main loop of PC calculation
        while true
            iter = iter + 1;
            % Recalculate x!
            tmp = repmat(y, n, 1);
            tmp(nans) = 0;
            x = sum(bsxfun(@minus, zData, b) .* tmp, 2) ./ sum(tmp .^ 2, 2);
            % Check of convergence
            if 1 - abs(oldY * y') < tolConv
                break;
            end
            oldY = y;
            % Recalculate b!
            b = mean(data - x * y, 'omitnan');
            % Recalculate y!
            tmp = repmat(x, 1, m);
            tmp(nans) = 0;
            y = sum(bsxfun(@minus, zData, b) .* tmp)...
                ./ sum(tmp .^ 2);
            y = y ./ sqrt(sum(y .^ 2));
        end
        % Recalculate result and residuals
        tmp = bsxfun(@plus, b, x * y);
        res = res + tmp;
        data = data - tmp;
        curr = sum(var(data, 'omitnan'));
        if verbose
            fprintf(['Component %d Fraction of unexplained',...
                ' variance is %f iterations %d\n'], comp, curr/base, iter);
        end
        if curr < cutVar
            break;
        end
        comp = comp + 1;
    end
end

function res = restrictedSVD(data, nans, tol, tolConv, lo, hi, verbose)
%restrictedSVD implements SVD for data with gaps with individual intervals
%for each missed value.
%
%Inputs:
%   data is n-by-m matrix of data
%   nans is n-by-m matrix with true corresponds to messed value.
%   tol is tolerance level to defile required number of PCs
%   tolConv is tolerance level to define PC convergence
%   lo is n-by-m matrix of low boundaries of intervals
%   hi is n-by-m matrix of high boundaries of intervals
%   verbose is indicator to output accuracy of each iteration
%Outputs:
%   res is n-by-m matrix of restored data.

    [n, m] = size(data);
    res = zeros(n, m);
    nanInd = find(nans);
    % Calculate residual variance 
    base = sum(var(data, 'omitnan'));
    cutVar = base * tol;
    comp = 1;
    while true
        % Create duplicate of data for fast calculation
        zData = data;
        zData(nanInd) = 0;
        % Initialise PC
        iter = 0;
        xiter = 0;
        b = mean(data, 'omitnan');
        % Furthest from mean
        tmp = bsxfun(@minus, data, b);
        [~, ind] = max(var(tmp, 0, 2, 'omitnan'));
        y = tmp(ind, :);
        y(isnan(y)) = 0;
        y = y ./ sqrt(sum(y .^ 2));
        % Zero oldY is guarantee of non stop at the first iteration
        oldY = zeros(1, m);
        %Main loop of PC calculation
        while true
            iter = iter + 1;
            % Recalculate x!
            %%%%%%%%%%%%%%%%%%%%%
            % Solve unrestricted problem.
            tmp = repmat(y, n, 1);
            tmp(nanInd) = 0;
            x = sum(bsxfun(@minus, zData, b) .* tmp, 2) ./ sum(tmp .^ 2, 2);
            % Form point closest to projection
            while true
                xiter = xiter + 1;
                tmp = bsxfun(@plus, b, x * y);
                tmp(~nans) = data(~nans);
                ind = nans & tmp < lo;
                cnt = sum(ind(:));
                tmp(ind) = lo(ind);
                ind = nans & tmp > hi;
                cnt = cnt + sum(ind(:));
                % If cnt == 0 then we have all projections inside intervals
                % and it is not necessary to continue calculations!
                if cnt == 0 
                    break;
                end
                tmp(ind) = hi(ind);
                oldX = x;
                x = sum(bsxfun(@times, bsxfun(@minus, tmp, b), y), 2);
                if sqrt(sum((oldX - x) .^ 2)) < tol
                    break;
                end
            end
            
            % Check of convergence
            if 1 - abs(oldY * y') < tolConv
                break;
            end
            oldY = y;
            % Recalculate b!
            b = mean(data - x * y, 'omitnan');
            % Recalculate y!
            tmp = repmat(x, 1, m);
            tmp(nanInd) = 0;
            y = sum(bsxfun(@minus, zData, b) .* tmp)...
                ./ sum(tmp .^ 2);
            y = y ./ sqrt(sum(y .^ 2));
        end
        % Recalculate result and residuals
        tmp = bsxfun(@plus, b, x * y);
        res = res + tmp;
        data = data - tmp;
        lo = lo - tmp;
        hi = hi - tmp;
        curr = sum(var(data, 'omitnan'));
        if verbose
            fprintf(['Component %d Fraction of unexplained',...
                ' variance is %f iterations %d x iterations %d\n'],...
                comp, curr/base, iter, xiter);
        end
        if curr < cutVar
            break;
        end
        comp = comp + 1;
    end
end
