## Authors
- Leonard Deutschle, PHD student in Electrical Engineering and Information Technology at ETH Zurich
- Alexander Maeder, Master's Student in Electrical Engineering and Information Technology at ETH Zurich
## Description
Contains files to read and manipulate a created MATLAB GW solution in Python.

The MATLAB solution is given as a struct with the following fields:
- E: The energy grid vector. 
- GGE_sparse: Greater Green's Function
- GLE_sparse: Lesser Green's Function
- GRE_sparse: Retarded Green's Function
- PGE_sparse: Greater Polarization
- PLE_sparse: Lesser Polarization
- PRE_sparse: Retarded Polarization
- WGE_sparse: Greater W
- WLE_sparse: Lesser W
- WRE_sparse: Retarded W
- Sigma_GWGE_sparse: Greater Sigma
- Sigma_GWLE_sparse: Lesser Sigma
- Sigma_GWRE_sparse: Retarded Sigma
All of the above except the energy vector, are 1 x #energy points structs. 
Where every field named "sparse matrix" is a sparse matrix.
It is assumed that all of them have non-zero elements at the same indexes.

The MATLAB solution is then transformed with `changeFormatGPWS.m`
and can then be read with the Python script `read_solution.py`

`changeFormatGPWS.m` contains a function, which needs the path to the solution as an input.

The main reason for the formatting is to transform the data from a vector of sparse matrices
to a 2D array. Meaning every column is the data of one sparse matrix and there are #energy point columns.
In addition, the MATLAB script fills "missing" values with a vanishing number to achieve uniformity in the data.

The Python script needs the following dependencies:
- h5py
- numpy
- scipy

# Note
The transformation can fail. This is due that it is assumed that `PRE_sparse(1)` is one 
of the sparse matrices with the most non-zero elements. 
This was arbitrary for the two given datasets.
In case the transformation fails, the line
```
[rows, columns, ~] = find(sr.PRE_sparse(1).sparse_matrix); 
```
has to be changed. `sr.PRE_sparse(1)` should be replaced with a different sparse matrix
of which can be assumed to have the most non-zero elements.
In the previously given data, most to all matrices had the most non-zero elements.
