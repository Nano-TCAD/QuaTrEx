function newData = insertAtIndices(index, data, number)
% This function inserts a number into a data vector at all the elements of the
% index vector.

assert(length(number) == length(index));

newData = data;
for i = 1:length(index)
    newData = [newData(1:index(i)-1) number(i) newData(index(i):end)];
end

end