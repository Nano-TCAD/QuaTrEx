function changeFormatV(pathV)
    % function to change format from matlab sparse
    % to directly save the three vectors independently
    % done to avoid tedious stuff in python
    % not really needed for V since only one matrix
    % but for XR, XL, XG it was needed

    sr = load(pathV);
    
    % V has the full sparsity pattern as all the 
    % other used tensors
    [rows, columns, data] = find(sr.V); 

    % save directly data
    formatted.rows = rows;
    formatted.columns = columns;
    formatted.realvh = real(data);
    formatted.imgvh = imag(data);
    
    % save to file
    save("data_Vh_4.mat", "formatted", "-v7.3", "-nocompression");
end