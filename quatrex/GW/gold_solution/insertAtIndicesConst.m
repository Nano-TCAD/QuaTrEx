% Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

function newData = insertAtIndicesConst(index, data, number)
% This function inserts a number into a data vector at all the elements of the
% index vector.

newData = data;
for i = 1:length(index)
    newData = [newData(1:index(i)-1); number; newData(index(i):end)];
end

end