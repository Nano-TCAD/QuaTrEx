function dataOut  = fixData(largerIdx, smallerIdx, dataIn)
% FIXDATA - Adjusts dataIn based on a set of larger and smaller indices
%
%   dataOut = FIXDATA(largerIdx, smallerIdx, dataIn) adjusts dataIn based
%   on the larger and smaller indices. The output dataOut has the same size
%   as largerIdx and is sorted to match smallerIdx. The missing elements in
%   smallerIdx are replaced with realmin.
%
%   Inputs:
%   -------
%   largerIdx: matrix of size NxM representing the larger index set
%   smallerIdx: matrix of size PxM representing the smaller index set
%   dataIn: matrix of size NxK representing the input data
%
%   Outputs:
%   --------
%   dataOut: matrix of size NxK representing the output data with missing
%   elements replaced by realmin and sorted to match smallerIdx

% Check if each row of largerIdx is also present in smallerIdx
[Lia, ~] = ismember(largerIdx, smallerIdx, 'rows');

% Get the missing indices in largerIdx
missingIdx = largerIdx(~Lia,:);

% Ensure that the number of missing indices matches the difference
% between the number of indices in largerIdx and smallerIdx
assert(numel(missingIdx) == numel(largerIdx) - numel(smallerIdx))

% Add the missing indices to smallerIdx
smallerIdxCopy = [smallerIdx; missingIdx];

% Ensure that the number of indices in the new smallerIdxCopy is equal to
% the number of indices in largerIdx
assert(numel(smallerIdxCopy) == numel(largerIdx));

% Get the locations of smallerIdxCopy in largerIdx
[~, Locb] = ismember(smallerIdxCopy, largerIdx, 'rows');

% Ensure that all indices are found in largerIdx
assert(all(Locb > 0));

% Sort the indices in Locb to get the sorting order
[~, idx] = sort(Locb);

% Replace missing elements in dataIn with realmin
dataOut = [dataIn; ones(size(missingIdx,1),1)*(realmin + 1j*realmin)];

% Sort the output data using the sorting order
dataOut = dataOut(idx);

% assert that sorting worked out
assert(isequal(smallerIdxCopy(idx,:),largerIdx));

end

