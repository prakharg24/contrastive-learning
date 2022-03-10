import numpy as np

def sinedistance_eigenvectors(ustar, w):
    upredicted, _, _ = np.linalg.svd(w.T, full_matrices=False)
    sinedistance = np.linalg.norm(np.matmul(ustar, ustar.T) - np.matmul(upredicted, upredicted.T))/(2**0.5)
    return sinedistance
