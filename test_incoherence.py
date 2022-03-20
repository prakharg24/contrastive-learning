import numpy as np
import matplotlib.pyplot as plt

# d = 200
d_array = range(200, 250)
r = 50
# r_array = range(10, 50)

for seed in range(5):
    # for r in r_array:
    for d in d_array:
        u, s, vh = np.linalg.svd(np.random.rand(d, r), full_matrices=False)
        ustar = np.matmul(u, vh)

        max_incoherence = 0
        for i in range(d):
            basis = np.zeros(d)
            basis[i] = 1
            # incoherence is now the i-th row of U
            incoherence = np.matmul(basis.T, ustar)
            # Retain the max squared norm
            max_incoherence = max(max_incoherence, np.linalg.norm(incoherence) ** 2)

        # plt.scatter(r, max_incoherence)
        plt.scatter(d, max_incoherence)

plt.savefig("inc_with_d.png")