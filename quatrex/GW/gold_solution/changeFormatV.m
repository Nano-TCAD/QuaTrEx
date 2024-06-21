% Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

function changeFormatV(pathV)
    % function to change format from matlab sparse
    % to directly save the three vectors independently
    % done to avoid tedious stuff in python
    % not really needed for V since only one matrix
    % but for XR, XL, XG it was needed
    path = '/usr/scratch/mont-fort17/dleonard/GW_paper/Si_Nanowire_poisson/';

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
    filename = [path 'data_Vh_PS_Si.mat'];
    save(filename, "formatted", "-v7.3", "-nocompression");

    % V has the full sparsity pattern as all the 
    % other used tensors
    [rows, columns, data] = find(sr.H);
   

    % save directly data
    formatted.rows = rows;
    formatted.columns = columns;
    formatted.realvh = real(data);
    formatted.imgvh = imag(data);
    
    % save to file
    filename = [path 'data_H_PS_SI.mat'];
    save(filename, "formatted", "-v7.3", "-nocompression");

    % V has the full sparsity pattern as all the 
    % other used tensors
    [rows, columns, data] = find(sr.S);
   

    % save directly data
    formatted.rows = rows;
    formatted.columns = columns;
    formatted.realvh = real(data);
    formatted.imgvh = imag(data);
    
    % save to file
    filename = [path 'data_S_PS_SI.mat'];
    save(filename, "formatted", "-v7.3", "-nocompression");

end