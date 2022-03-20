import numpy as np
from test_utils import sinedistance_eigenvectors
import math


class SpikedCovarianceDataset():
    def __init__(self, r, d, sigma, noise_sigma, heteroskedastic_bound, label_mode='cls'):
        ## Setting hyperparameters for data generation
        self.r = r
        self.d = d
        self.sigma = sigma
        self.noise_sigma = noise_sigma
        self.heteroskedastic_bound = heteroskedastic_bound

        ## Creating orthonormal matrix U* through SVD
        u, s, vh = np.linalg.svd(np.random.rand(d, r), full_matrices=False)
        self.ustar = np.matmul(u, vh)

        # Scale U* s.t. incoherence condition is satisfied
        # incoherence = self.calc_incoherence()
        # ideal_incoherence = self.get_ideal_incoherence()
        # self.ustar = self.ustar / math.sqrt(incoherence) * math.sqrt(ideal_incoherence)

        ## Creating Optimal Classifier Vector w*
        self.wstar = np.random.rand(r)
        self.wstar = self.wstar / np.linalg.norm(self.wstar)

        ## Setting label mode to 'classification' or 'regression'
        self.label_mode = label_mode

    def get_ideal_incoherence(self):
        return self.r * math.log(self.d) / self.d

    def calc_incoherence(self):
        max_incoherence = 0
        for i in range(self.d):
            basis = np.zeros(self.d)
            basis[i] = 1
            # incoherence is now the i-th row of U
            incoherence = np.matmul(basis.T, self.ustar)
            # Retain the max squared norm
            max_incoherence = max(max_incoherence, np.linalg.norm(incoherence) ** 2)
        return max_incoherence

    def get_next_batch(self, batch_size=64):
        ## Original Signal z
        signal = np.random.normal(0., self.sigma, self.r*batch_size)
        signal = signal.reshape(self.r, batch_size)

        ## Noise Sigma E
        # noise = np.random.normal(0., self.noise_sigma, self.d*batch_size)
        # noise = noise.reshape(self.d, batch_size)

        # Heteroskedastic noise
        noise = np.zeros((self.d, batch_size))
        ratio_bound = self.heteroskedastic_bound
        for idx in range(len(noise)):
            # For every dimension, decrease the noise_sigma by a constant until we reach ratio bound
            heteroskedastic_noise_sigma = self.noise_sigma - \
                                          (self.noise_sigma - self.noise_sigma/ratio_bound)/len(noise) * idx
            noise[idx] = np.random.normal(0., heteroskedastic_noise_sigma, batch_size)

        ## Input features are U*z + E
        input_features = np.matmul(self.ustar, signal) + noise
        input_features = input_features.T # Make batch size first dimension

        ## Calculate Inner Product Normalized <z, w*>/v
        inner_product = np.matmul(self.wstar, signal)/self.sigma
        if self.label_mode=='cls':
            ## Output labels selected from bernoulli distribution with selected probabilities
            bernoulli_mean = 1/(1 + np.exp(-inner_product))
            output_labels = np.random.binomial(1, bernoulli_mean, size=batch_size)
        elif self.label_mode=='reg':
            ## Output labels with some regression error e
            regression_error = np.random.normal(0., self.sigma/10., batch_size)
            output_labels = inner_product + regression_error
        else:
            raise Exception("Label Mode Not Correct")

        return input_features, output_labels, signal.T

    def get_ustar(self):
        ## To extract the original matrix U* for comparison
        return self.ustar

    def get_wstar(self):
        ## TO extract the original classifier/regressor w*
        return self.wstar


if __name__=="__main__":
    generator = SpikedCovarianceDataset(5, 20, 1., label_mode='classification')
    # generator = SpikedCovarianceDataset(5, 20, 1., label_mode='regression')
    X, y = generator.get_next_batch(batch_size=4)
    print(X)
    print(y)
