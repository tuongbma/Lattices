import numpy as np

def compute_muy(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / np.dot(vector_b, vector_b)

def compute_vector_magnitude(v):
    return np.linalg.norm(v)

def compute_c_in_reduction_step(vector_a, vector_b):
    return round(np.dot(vector_a, vector_b) / np.dot(vector_b, vector_b))
    
def swap_column(matrix_m, col_i, col_j):
    matrix_m[:, [col_i,col_j]] = matrix_m[:, [col_j,col_i]]
    return matrix_m

def compute_Gram_Schmidt_orthogonalization(lattice_basis):
    m = lattice_basis.shape[0]
    n = lattice_basis.shape[1]

    GS_orth = np.zeros((m,n)) #init a matrix with size m x n
    GS_orth[:,0] = lattice_basis[:,0] # b1 = b1~

    for i in range (n):
        temp_basis_i = lattice_basis[:, i]
        for j in range(0, i):
            muy = compute_muy(lattice_basis[:,i], GS_orth[:,j])
            GS_orth[:,i] = np.subtract(temp_basis_i, muy * GS_orth[:,j])
            temp_basis_i =  GS_orth[:,i] # to subtract in the next inner loop
    return GS_orth
