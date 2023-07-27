% Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

function [realp, imgp] = extractData(sparseData, realin, imgin, formatted, i)
    % This function extracts the real and imaginary part in the coo format
    % for every energy point i
    % where sparse data is the data in csc format and realin/imgin are the
    % arrays to save the data into
    % formatted is the final struct and use to assert sparsness properties
    % i is the energy point 

    % copy input data
    realp = realin;
    imgp = imgin;
    
    [rows, columns, data] = find(sparseData); 
    
    % if at energy below/above fermi level some sparse matrices are
    % fully zero. Therefore pad with realmin~e-300
    if isempty(columns)
        realp(i,:) = realmin;
        imgp(i,:) = realmin;
    else
        % more bandaid fixes:
        % since out of reasons P* has a few zero elements less...
        % find these and add data approx e-300

        % the same amount
        if length(rows) == length(formatted.rows)
            assert(isequal(rows, formatted.rows));
            assert(isequal(columns, formatted.columns));
        % lower amount
        else
            assert(length(rows) <= length(formatted.rows));
            assert(length(data) <= length(formatted.rows));
            assert(length(data) == length(rows));
            % inserts missing elements with 10^~300 and sorts into right
            % order
            data = fixData([formatted.rows formatted.columns], [rows columns], data);
            assert(length(data) == length(formatted.rows));
        end
        realp(i,:) = real(data);
        imgp(i,:) = imag(data);
    end

end


