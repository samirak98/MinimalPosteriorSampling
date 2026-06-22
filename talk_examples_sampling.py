# =============================================================================
# .py
# =============================================================================
# Authors:
#   - Samira Kabri <samira.kabri@desy.de> (DESY)
#   - Tim Roith <tim.roith@tum.de> (TU Munich)
# =============================================================================


#%%
import numpy as np

from distributions import Gaussian_noise_likelihood, Gaussian
from diffusion_sampling import Reverse_Markov
from utils import linear_fwd_operator

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy.stats import multivariate_normal
from sklearn import mixture
from plotting import plot_ellipse

%matplotlib widget
clf = mixture.GaussianMixture(n_components=1, covariance_type="full")

np.random.seed(80)

#%%
# diffusion functions
def beta(t, beta_min=0.1, beta_max=20.):
    return beta_min + t * (beta_max - beta_min)

def alpha(t, beta_min=0.1, beta_max=20.):
    return np.exp(-0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t**2))

def sigma(t, beta_min=0.1, beta_max=20.):
    return np.sqrt(1 - alpha(t, beta_min, beta_max)**2)

# SDE class for Gaussian densities
class sde_Gauss:
    def __init__(self, mean, var):
        self.mean = mean
        self.var  = var

    def __call__(self, x, t):
        var_t = alpha(t) ** 2 * self.var + (sigma(t) ** 2*np.eye(x.shape[-1]))
        mean_t = self.mean * alpha(t)
        return multivariate_normal.pdf(x, mean=mean_t, cov=var_t)

    def score(self, x, t):
        var_t = alpha(t) ** 2 * self.var + (sigma(t) ** 2*np.eye(x.shape[-1]))
        mean_t = self.mean * alpha(t)
        return -np.linalg.solve(var_t, (x-mean_t).T).T
    
    def apply_information(self, x, t, z):
        var_t = alpha(t) ** 2 * self.var + (sigma(t) ** 2*np.eye(x.shape[-1]))
        return - np.linalg.solve(var_t.T, z.T).T

#%%
# initialize a Gaussian prior
m_prior = -1. * np.ones(2)
theta = np.pi/4
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
L = rot@(np.array([1.,0.1])*np.eye(2))#1/3*np.array([[1.,0],[2.,3.]])
sigma_prior = L@L.T
prior = Gaussian(mean=m_prior, variance=sigma_prior)

# plot the prior
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.grid(color='black', linestyle='solid', linewidth=0.2, alpha=1.)
num_pts = 150
X,Y = np.meshgrid(np.linspace(-4,4,num_pts), np.linspace(-3,3,num_pts))
pos = np.dstack((X, Y)).reshape(-1,2)

ax.contourf(X, Y, prior(pos).reshape(num_pts,num_pts), levels=40, cmap='jet')
ax.set_title('Prior density $p_X$')
plt.show()

# prior sampling
# to sample from the prior, we initialize an SDE with prior mean and variance
sde_prior= sde_Gauss(prior.mean, prior.variance)

# %%
# Posterior sampling

# initialize forward operator
A = 2*np.array([[1., 0.], [0., .5]])
fwd = linear_fwd_operator(A)

# initialize measurement noise variance
sigma_meas = 0.1

# generate measurement
x_true = prior.sample()
y_meas = fwd(x_true) + np.sqrt(sigma_meas)*np.random.normal(size=x_true.shape)

# initialize likelihood
likeli = Gaussian_noise_likelihood(y_meas, sigma=sigma_meas, fwd=fwd)

# compute posterior
# posterior (with Gaussian props)
sigmainv = np.linalg.inv(sigma_prior) + (1/sigma_meas*np.eye(2))@A@A.T
sigma_post = np.linalg.inv(sigmainv)
m_post = np.linalg.solve(sigmainv, np.linalg.solve(sigma_prior, m_prior)+A.T@(y_meas/sigma_meas))
post = Gaussian(m_post, sigma_post)
# %%
# plot prior, likelihood, posterior
fig, ax = plt.subplots(1,3, figsize=(8,3))
for i in range(3):
    ax[i].grid(color='black', linestyle='solid', linewidth=0.2, alpha=1.)
num_pts = 150
X,Y = np.meshgrid(np.linspace(-4,4,num_pts), np.linspace(-3,3,num_pts))
pos = np.dstack((X, Y)).reshape(-1,2)

ax[0].contourf(X, Y, prior(pos).reshape(num_pts,num_pts), levels=40, cmap='jet')
ax[1].contourf(X, Y, likeli(pos).reshape(num_pts,num_pts), levels=40, cmap='jet')
ax[2].contourf(X, Y, post(pos).reshape(num_pts,num_pts), levels=40, cmap='jet')

ax[0].set_title('Prior $x \mapsto p_X(x)$')
ax[1].set_title('Likelihood $x \mapsto p_{Y|X=x}(y)$')
ax[2].set_title('Posterior $x \mapsto p_{X|Y=y}(x)$')

# %%
# posterior sampling
# to sample from the posterior, we initialize an SDE with posterior mean and variance
sde_post= sde_Gauss(post.mean, post.variance)

# %%
# Posterior sampling without knowing the posterior
class sde_posterior_Bayes:
    def __init__(self, score_prior, score_likeli):
        self.score_prior = score_prior
        self.score_likeli = score_likeli

    def __call__(self, *args, **kwds):
        return 0
    def score(self, x, t):
        return self.score_prior(x, t) + self.score_likeli(x, t)

def funadd(f,g, a=1):
    return lambda x, t: f(x, t) + a*g(x, t)

def ALD_score(x, t):
    return 1/ sigma_meas * (y_meas - (x @ A.T)) @ A

def Song_score(x, t):
    z = np.random.normal(size=x.shape)
    y_noised = alpha(t) * y_meas + sigma(t) * (z @ A.T)
    return 1/sigma_meas*(y_noised - x @ A.T) @ A

def DPS_score(x,t):
    x0 = (x + sigma(t)**2*sde_prior.score(x,t))/alpha(t)
    grad_x0 = (y_meas-x0@A.T)@A/sigma_meas
    information = sde_prior.apply_information(x, t, grad_x0)
    return (grad_x0 + sigma(t)**2*information)/alpha(t) 

sde_post_ALD = sde_posterior_Bayes(sde_prior.score, ALD_score)
sde_post_DPS = sde_posterior_Bayes(sde_prior.score, DPS_score)

# %%
sde = sde_prior # Possible: sde_prior, sde_post, sde_post_ALD, sde_post_DPS

N = 10000 # number of time steps
x = np.random.normal(size = (1000,2)) # number of samples

rev_markov = Reverse_Markov(sde, beta) # initialize reverse Markov chain
hist = rev_markov(T=1, N=N, xT=x) # sample
n_hist = hist.shape[0]-1 

# plot sampling process
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.grid(color='black', linestyle='solid', linewidth=0.2, alpha=1.)


ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
slider = Slider(ax_slider, 'Time step', 0, n_hist, valinit=0, valstep=1)

def update(val):
    t = int(slider.val)
    ax.cla()
    ax.grid(color='black', linestyle='solid', linewidth=0.2, alpha=1.)
    ax.scatter(hist[t,:,0], hist[t, :, 1], alpha=0.1)
    fig.canvas.draw_idle()
    ax.axis('equal')
    try:
        clf.fit(hist[t])
        plot_ellipse(clf.covariances_[0], clf.means_[0], ax=ax, edgecolor='black', linewidth=2)
        plot_ellipse(sde.var, sde.mean, ax=ax, edgecolor='red', linewidth=2)
    except:
        clf.fit(hist[t])
        plot_ellipse(clf.covariances_[0], clf.means_[0], ax=ax, edgecolor='black', linewidth=2)
        plot_ellipse(sde_post.var, sde_post.mean, ax=ax, edgecolor='red', linewidth=2)

slider.on_changed(update)
update(0)
plt.show()
# %%
