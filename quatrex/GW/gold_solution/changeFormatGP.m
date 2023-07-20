% Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

function changeFormatGP(path)
    % load .mat file from path and creates file in format E, rows, cols,
    % real X, img X where E is a vector and the rest are structs of size
    % the energy vector lenght filled with a vector. 
    % data is then stored in coo format....

    % load file
    sr = load(path);
    
    % stick together new struct
    formatted.E = sr.E;
    
    % assume at every energy point same non-zero elements (just take first
    % then)
    % this is sadly not true, at some energy points there are less non zero
    % elements, but sr.PRE_sparse(1) has randomly the max (there are only a
    % few points with less, so most have the same amount)
    [rows, columns, ~] = find(sr.PRE_sparse(1).sparse_matrix); 
    
   
    formatted.rows = rows;
    formatted.columns = columns;
    
    ne = length(formatted.E);
    no = length(formatted.columns);
    assert(length(formatted.columns) == length(formatted.rows));

    % create array to save non-zero elements
    realgg = zeros(ne,no);
    realgl = zeros(ne,no);
    realgr = zeros(ne,no);
    imggg = zeros(ne,no);
    imggl = zeros(ne,no);
    imggr = zeros(ne,no);

    realpg = zeros(ne,no);
    realpl = zeros(ne,no);
    realpr = zeros(ne,no);
    imgpg = zeros(ne,no);
    imgpl = zeros(ne,no);
    imgpr = zeros(ne,no);

    % not good practice to extract data in a for-loop iteratively 
    % btw todo better alternative
    for i = 1:ne
        % extract gg
        [realgg, imggg] = extractData(sr.GGE_sparse(i).sparse_matrix, realgg, imggg, formatted, i);

        % extract gl
        [realgl, imggl] = extractData(sr.GLE_sparse(i).sparse_matrix, realgl, imggl, formatted, i);

        % extract gr
        [realgr, imggr] = extractData(sr.GRE_sparse(i).sparse_matrix, realgr, imggr, formatted, i);
            
        % extract pg
        [realpg, imgpg] = extractData(sr.PGE_sparse(i).sparse_matrix, realpg, imgpg, formatted, i);

        % extract pl
        [realpl, imgpl] = extractData(sr.PLE_sparse(i).sparse_matrix, realpl, imgpl, formatted, i);

        % extract pr
        [realpr, imgpr] = extractData(sr.PRE_sparse(i).sparse_matrix, realpr, imgpr, formatted, i);
    end


    % stick together new struct

    % transpose energy and cols/ros
    formatted.E = formatted.E.';
    formatted.rows = formatted.rows.'; 
    formatted.columns = formatted.columns.'; 

    % todo repeated code blocks 
    assert(isequal(size(realgg),[ne,no]))
    assert(numel(realgg) == numel(realgl))
    assert(numel(realgg) == numel(realgr))
    assert(numel(realgg) == numel(realpg))
    assert(numel(realgg) == numel(realpl))
    assert(numel(realgg) == numel(realpr))

    formatted.realgg = realgg;
    formatted.imggg = imggg;

    formatted.realgl = realgl;
    formatted.imggl = imggl;

    formatted.realgr = realgr;
    formatted.imggr = imggr;

    formatted.realpg = realpg;
    formatted.imgpg = imgpg;

    formatted.realpl = realpl;
    formatted.imgpl = imgpl;

    formatted.realpr = realpr;
    formatted.imgpr = imgpr;

    % save to file
    save("data_GP.mat", "formatted","-v7.3","-nocompression");
end