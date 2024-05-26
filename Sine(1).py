import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared

sampling_freq = 1000
tstep = 1/sampling_freq
N = [1000,2000]

t={}
for i in N:
    t[i] = np.linspace(0,(i-1)*tstep,i)

signal= {}
for i in t:
    signal[i]= np.sin(2 * np.pi * 3 * t[i])

pred = {}
b = {}
for i in N:
    pred[i] = np.linspace(0,(i-1 + 500)*tstep,i)
    b[i] = np.sin(2 * np.pi * 3 * pred[i])

kernels = {'RBF':1 * RBF(length_scale=1.0, length_scale_bounds=(-1e2, 1e2)), 
           'RQ':1 * RationalQuadratic(length_scale=1.0, alpha=1),
           'ESS':1 * ExpSineSquared(length_scale=1, periodicity=1)}

for n in t:
    for i in kernels:
        gaussian_process = GaussianProcessRegressor(kernel=kernels[i])
        gaussian_process.fit(t[n].reshape(-1, 1), signal[n].reshape(-1, 1))

        mean_prediction, std_prediction = gaussian_process.predict(pred[n].reshape(-1,1), return_std=True)
        
        # Plot the predictions of the gaussian process regressor
        plt.plot(
            pred[n],
            mean_prediction,
            label="Gaussian process regressor",
            linewidth=2,
            linestyle="dotted",
        )
        
        plt.fill_between(
            pred[n].ravel(),
            mean_prediction - std_prediction,
            mean_prediction + std_prediction,
            color="tab:green",
            alpha=0.2,
        )
        
        plt.scatter(
            t[n],
            signal[n],
            color="black",
        )

        plt.plot(pred[n],b[n],linestyle='dashed')

        plt.title(f'Fitting {n} with kernel {i}')

        plt.legend(['Prediction','Uncertainity','Training Data','Actual Curve'],loc='upper right')

        plt.show()


