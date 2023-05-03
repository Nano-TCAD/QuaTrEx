function changeFormatGPWS(path)
    % load .mat file from path and creates file in format E, rows, cols,
    % real X, img X where E is a vector and the rest are structs of size
    % the energy vector lenght filled with a vector. 
    % data is then stored in coo format....

    % load file
    sr = load(path);
    
    % stick together new struct
    formatted.E = sr.E;
    formatted.Bmax = sr.Bmax;
    formatted.Bmin = sr.Bmin;
    
    % assume at every energy point same non-zero elements (just take first
    % then)
    % this is sadly not true, at some energy points there are less non zero
    % elements, but sr.PRE_sparse(1) has randomly the max (there are only a
    % few points with less, so most have the same amount)
    % I (almaeder) honestly don't like to just guess a point, but it is what it is
    [rows, columns, ~] = find(sr.PRE_sparse(1).sparse_matrix); 
    
   
    formatted.rows = rows;
    formatted.columns = columns;
    
    ne = length(formatted.E);
    no = length(formatted.columns);
    assert(length(formatted.columns) == length(formatted.rows));

    % create array to save non-zero elements
    % don't like to create million buffers
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

    realwg = zeros(ne,no);
    realwl = zeros(ne,no);
    realwr = zeros(ne,no);
    imgwg = zeros(ne,no);
    imgwl = zeros(ne,no);
    imgwr = zeros(ne,no);

    realsg = zeros(ne,no);
    realsl = zeros(ne,no);
    realsr = zeros(ne,no);
    imgsg = zeros(ne,no);
    imgsl = zeros(ne,no);
    imgsr = zeros(ne,no);


    % not good practice to extract data in a for-loop iteratively 
    % btw todo better alternative
    % todo remove repeated blocks
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

        % extract wg
        [realwg, imgwg] = extractData(sr.WGE_sparse(i).sparse_matrix, realwg, imgwg, formatted, i);

        % extract wl
        [realwl, imgwl] = extractData(sr.WLE_sparse(i).sparse_matrix, realwl, imgwl, formatted, i);

        % extract wr
        [realwr, imgwr] = extractData(sr.WRE_sparse(i).sparse_matrix, realwr, imgwr, formatted, i);
            
        % extract sg
        [realsg, imgsg] = extractData(sr.Sigma_GWGE_sparse(i).sparse_matrix, realsg, imgsg, formatted, i);

        % extract sl
        [realsl, imgsl] = extractData(sr.Sigma_GWLE_sparse(i).sparse_matrix, realsl, imgsl, formatted, i);

        % extract sr
        [realsr, imgsr] = extractData(sr.Sigma_GWRE_sparse(i).sparse_matrix, realsr, imgsr, formatted, i);
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
    assert(numel(realgg) == numel(realwg))
    assert(numel(realgg) == numel(realwl))
    assert(numel(realgg) == numel(realwr))
    assert(numel(realgg) == numel(realsg))
    assert(numel(realgg) == numel(realsl))
    assert(numel(realgg) == numel(realsr))
    
    % todo repeated code blocks 
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

    formatted.realwg = realwg;
    formatted.imgwg = imgwg;

    formatted.realwl = realwl;
    formatted.imgwl = imgwl;

    formatted.realwr = realwr;
    formatted.imgwr = imgwr;

    formatted.realsg = realsg;
    formatted.imgsg = imgsg;

    formatted.realsl = realsl;
    formatted.imgsl = imgsl;

    formatted.realsr = realsr;
    formatted.imgsr = imgsr;

    % save to file
    save("data_GPWS_04.mat", "formatted","-v7.3","-nocompression");
end