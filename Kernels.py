import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared
from scipy.stats import norm
from scipy import signal

x = np.arange(-50,51,1)
signals = {'sin':np.sin(x), 'Gaussian':norm.pdf(x,50,1), 'Square':signal.square(x/4)}
y = np.arange(-50,101,1)
original = {'sin':np.sin(y), 'Gaussian':norm.pdf(y,50,1), 'Square':signal.square(y/4)}

kernels = {'RBF' :1 * RBF(length_scale=1.0, length_scale_bounds=(-1e2, 1e2)),'RQ': 1 * RationalQuadratic(length_scale=1.0, alpha=1.5)
           , 'ESS':1 *  ExpSineSquared(length_scale=7, periodicity=1)}

for j in signals:
    for i in kernels:
        gaussian_process = GaussianProcessRegressor(kernel=kernels[i])
        gaussian_process.fit(x.reshape(-1,1), signals[j].reshape(-1,1))

        start_time = time.time()
        mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
            y.reshape(-1,1),
            return_std=True,
        )

        plt.scatter(
            x,
            signals[j],
            color="black",
            label="Training Data Points",
        )

        # Plot the predictions of the gaussian process regressor
        plt.plot(
            y,
            mean_predictions_gpr,
            label="Gaussian process regressor",
            linewidth=2,
            linestyle="dotted",
        )

        plt.fill_between(
            y.ravel(),
            mean_predictions_gpr - std_predictions_gpr,
            mean_predictions_gpr + std_predictions_gpr,
            color="tab:green",
            alpha=0.2,
        )

        plt.plot(y,original[j])

        plt.title(f'Fitting {j} wave with {i} kernel')

        plt.legend(['Training Data','Prediction','Uncertainity','Original Curve'],loc='upper right')

        plt.show()
