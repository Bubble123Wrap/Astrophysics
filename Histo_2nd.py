import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse

def plot_confidence_ellipse(mu, cov, alph, ax, clabel=None, label_bg='white', **kwargs):
    """Display a confidence ellipse of a bivariate normal distribution

    Arguments:
        mu {array-like of shape (2,)} -- mean of the distribution
        cov {array-like of shape(2,2)} -- covariance matrix
        alph {float btw 0 and 1} -- level of confidence
        ax {plt.Axes} -- axes on which to display the ellipse
        clabel {str} -- label to add to ellipse (default: {None})
        label_bg {str} -- background of clabel's textbox
        kwargs -- other arguments given to class Ellipse
    """
    c = -2 * np.log(1 - alph)  # quantile at alpha of the chi_squarred distr. with df = 2
    Lambda, Q = la.eig(cov)  # eigenvalues and eigenvectors (col. by col.)

    ## Compute the attributes of the ellipse
    width, height = 2 * np.sqrt(c * Lambda)
    # compute the value of the angle theta (in degree)
    theta = 180 * np.arctan(Q[1, 0] / Q[0, 0]) / np.pi if cov[1,0] else 0

    ## Create the ellipse
    if 'fc' not in kwargs.keys():
        kwargs['fc'] = 'None'
    level_line = Ellipse(mu, width, height, angle=theta, **kwargs)

    ## Display a label 'clabel' on the ellipse
    if clabel:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'  # color of the text
        pos = Q[:, 1] * np.sqrt(c * Lambda[1]) + mu  # position along the heigth

        ax.text(*pos, clabel, color=col,
                rotation=theta, ha='center', va='center', rotation_mode='anchor',  # rotation
                bbox=dict(boxstyle='round', ec='None', fc=label_bg, alpha=1))  # white box

    return ax.add_patch(level_line)

#Declaring Mean and Covarience
mean = np.array([0,0])
cov = np.array([[1,0],[0,1]])

#Creating Multinomial Gaussian Variable
x1,y1 = np.random.multivariate_normal(mean, cov, 10).T
x2,y2 = np.random.multivariate_normal(mean, cov, 50).T
x3,y3 = np.random.multivariate_normal(mean, cov, 100).T

# Original plot
fig = plt.figure()

# Creating Scatter Plots
#500 obs
ax31 = fig.add_axes([0.05, 0.1, 0.2, 0.3])
ax31.scatter(x1,y1)
plot_confidence_ellipse(mean, cov, 0.68, ax31 ,ec='y',label='sd(ideal)')
ax31.plot(*mean, 'kx')
plot_confidence_ellipse(mean, cov, 0.95, ax31 ,ec='r',label='2sd(ideal)')
ax31.plot(*mean, 'kx')
ax31.set_xlabel('x')
ax31.set_ylabel('y')
#1000 obs
ax33 = fig.add_axes([0.37, 0.1, 0.2, 0.3])
ax33.scatter(x2,y2)
plot_confidence_ellipse(mean, cov, 0.68, ax33 ,ec='y',label='sd(ideal)')
ax33.plot(*mean, 'kx')
plot_confidence_ellipse(mean, cov, 0.95, ax33 ,ec='r',label='2sd(ideal)')
ax33.plot(*mean, 'kx')
ax33.set_xlabel('x')
#10000 obs
ax35 = fig.add_axes([0.69, 0.1, 0.2, 0.3])
ax35.scatter(x3,y3)
plot_confidence_ellipse(mean, cov, 0.68, ax35 ,ec='y',label='sd(ideal)')
ax35.plot(*mean, 'kx')
plot_confidence_ellipse(mean, cov, 0.95, ax35 ,ec='r',label='2sd(ideal)')
ax35.plot(*mean, 'kx')
ax35.set_xlabel('x')

#Reconstructing Gaussian
#500obs
data1 = list(zip(x1,y1))
mean1 = np.mean(data1,axis=0)
cov1 = np.cov(data1,rowvar=False)
plot_confidence_ellipse(mean1, cov1, 0.68, ax31 ,ec='y',linestyle='dashed',label='sd(real)')
ax31.plot(*mean1, 'kx')
plot_confidence_ellipse(mean1, cov1, 0.95, ax31 ,ec='r',linestyle='dashed',label='2sd(real)')
ax31.plot(*mean1, 'kx')
ax31.legend(loc='upper right',bbox_to_anchor =(1.45, 1))
ax31.set_xlim(-5,5)
ax31.set_ylim(-5,5)
#10000obs
data2 = list(zip(x2,y2))
mean2 = np.mean(data2,axis=0)
cov2 = np.cov(data2,rowvar=False)
plot_confidence_ellipse(mean2, cov2, 0.68, ax33 ,ec='y',linestyle='dashed',label='sd(real)')
ax33.plot(*mean2, 'kx',linestyle='dashed')
plot_confidence_ellipse(mean2, cov2, 0.95, ax33 ,ec='r',linestyle='dashed',label='2sd(real)')
ax33.plot(*mean2, 'kx')
ax33.legend(loc='upper right',bbox_to_anchor =(1.45,1))
ax33.set_xlim(-5,5)
ax33.set_ylim(-5,5)
#10000obs
data3 = list(zip(x3,y3))
mean3 = np.mean(data3,axis=0)
cov3 = np.cov(data3,rowvar=False)
plot_confidence_ellipse(mean3, cov3, 0.68, ax35 ,ec='y',linestyle='dashed',label='sd(real)')
ax35.plot(*mean3, 'kx')
plot_confidence_ellipse(mean3, cov3, 0.95, ax35 ,ec='r',linestyle='dashed',label='2sd(real)')
ax35.plot(*mean3, 'kx')
ax35.legend(loc='upper right',bbox_to_anchor =(1.5, 1))
ax35.set_xlim(-5,5)
ax35.set_ylim(-5,5)

#Creating Histograms
#500obs
#Y-Axis
ax32 = fig.add_axes([0.05, 0.7, 0.2, 0.1])
ax32.hist(y1,40, histtype='stepfilled', orientation='vertical',range=[-5,5],density=True)
ax32.set_xlabel('y')
ax32.set_ylabel('Density')
ax32.set_title('10obs')
#X-Axis
ax21 = fig.add_axes([0.05, 0.5, 0.2, 0.1])
ax21.hist(x1,40, histtype='stepfilled', orientation='vertical',range=[-5,5],density=True)
ax21.set_xlabel('x')
ax21.set_ylabel('Density')
#10000obs
#Y-Axis
ax34 = fig.add_axes([0.37, 0.7, 0.2, 0.1])
ax34.hist(y2,40, histtype='stepfilled', orientation='vertical',range=[-5,5],density=True)
ax34.set_xlabel('y')
ax34.set_title('50obs')
#X-Axis
ax22 = fig.add_axes([0.37, 0.5, 0.2, 0.1])
ax22.hist(x2,40, histtype='stepfilled', orientation='vertical',range=[-5,5],density=True)
ax22.set_xlabel('x')
#10000obs
#Y-Axis
ax36 = fig.add_axes([0.69, 0.7, 0.2, 0.1])
ax36.hist(y3,40, histtype='stepfilled', orientation='vertical',range=[-5,5],density=True)
ax36.set_xlabel('y')
ax36.set_title('100obs')
#X-Axis
ax23 = fig.add_axes([0.69, 0.5, 0.2, 0.1])
ax23.hist(x3,40, histtype='stepfilled', orientation='vertical',range=[-5,5],density=True)
ax23.set_xlabel('x')

#Creating Gaussians

#500obs
#Y-Axis
my, sdy = norm.fit(y1)
ymin, ymax = -5, 5
y = np.linspace(ymin, ymax, 100)
py = norm.pdf(y, my, sdy)
ax32.plot(y, py)
#X-Axis
mx, sdx = norm.fit(x1)
xmin, xmax = -5, 5
x = np.linspace(xmin, xmax, 100)
px = norm.pdf(x, mx, sdx)
ax21.plot(x, px)
#10000obs
#Y-Axis
my, sdy = norm.fit(y2)
ymin, ymax = -5, 5
y = np.linspace(ymin, ymax, 100)
py = norm.pdf(y, my, sdy)
ax34.plot(y, py)
#X-Axis
mx2, sdx2 = norm.fit(x2)
xmin, xmax = -5, 5
x = np.linspace(xmin, xmax, 100)
px = norm.pdf(x, mx, sdx)
ax22.plot(x, px)
#1000obs
#Y-Axis
my, sdy = norm.fit(y3)
ymin, ymax = -5, 5
y = np.linspace(ymin, ymax, 100)
py = norm.pdf(y, my, sdy)
ax36.plot(y, py)
#X-Axis
mx3, sdx3 = norm.fit(x3)
xmin, xmax = -5, 5
x = np.linspace(xmin, xmax, 100)
px = norm.pdf(x, mx, sdx)
ax23.plot(x, px)

plt.show()