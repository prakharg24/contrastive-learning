from data_generator import SpikedCovarianceDataset
import numpy as np
import matplotlib.pyplot as plt
# Check if incoherence increases with r linearly
d = 10
x = np.arange(1, 1000)
print(x.shape)
y = []
for r in x:
    generator = SpikedCovarianceDataset(r, d, 1, 0.5, label_mode="reg")
    y.append(generator.calc_incoherence())
plt.plot(x, y)
plt.show()
r = 10
x = np.arange(1, 1000)
y = []
for d in x:
    generator = SpikedCovarianceDataset(r, d, 1, 0.5, label_mode="reg")
    y.append(generator.calc_incoherence())
plt.plot(x, y)
plt.show()
