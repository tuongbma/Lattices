import numpy as np
import util
import lll

def babai(gamma, lattice_basis, t):
    n = lattice_basis.shape[1] #get basis rank

    #Step 1: Run LLL on lattice basis B
    lll_reduced_basis = lll.lll_algorithm(gamma, lattice_basis)

    # Compute Gram-Schmidt orthogonalization of reduced basis
    GS_orth = util.compute_Gram_Schmidt_orthogonalization(lll_reduced_basis)

    #Step 2: Compute x - the CVP
    b = t
    for j in range (n-1, -1, -1): # run from n-1 to 0
        c = util.compute_c_in_reduction_step(b, GS_orth[j])
        b = np.subtract(b, c * lll_reduced_basis[j])
    
    return (np.subtract(t,b))

if __name__ == "__main__":
    gamma = 3/4
    t = np.array([1,2,3])
    lattice_basis = np.array([[0, 1, -1],
                              [1, 1, 0],
                              [0, 1, 2]])
    print(babai(gamma,lattice_basis, t))