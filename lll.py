import numpy as np
import util

def lll_algorithm(gamma, lattice_basis):
    while(True): #loop util break
        stop_loop = True
        n = lattice_basis.shape[1]

        # Step 1: Compute Gram-Schmidt orthogonalization
        GS_orth = util.compute_Gram_Schmidt_orthogonalization(lattice_basis)

        # Step 2: Reduction
        for i in range (1, n):
            for j in range(i-1, -1, -1): #loop backward
                c = util.compute_c_in_reduction_step(lattice_basis[:,i], GS_orth[:,j])
                lattice_basis[:,i] = np.subtract(lattice_basis[:,i], c * lattice_basis[:,j])

        # Step 3: Swap
        for i in range (n-1):
            muy = util.compute_muy(lattice_basis[:,i+1], lattice_basis[:,i])
            x = util.compute_vector_magnitude(np.add(muy * GS_orth[i], GS_orth[:,i+1]))
            if (gamma * (util.compute_vector_magnitude(GS_orth[:,i])) ** 2 > x ** 2): # not satify LLL condition 2
                lattice_basis = util.swap_column(lattice_basis, i, i+1)
                stop_loop = False # if some columns are swapped => back to step 1
                break

        if (stop_loop): # if swap step doesnt swap any columns => stop loop and return result
            return lattice_basis

if __name__ == "__main__":
    gamma = 3/4
    # lattice_basis = np.array([[3,2], 
    #                           [1,1]])
    lattice_basis = np.array([[1, -1, 3],
                              [1, 0, 5],
                              [1, 2, 6]])
    print(lll_algorithm(gamma,lattice_basis))