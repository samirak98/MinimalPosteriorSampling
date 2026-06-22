# =============================================================================
# .py
# =============================================================================
# Authors:
#   - Tim Roith <tim.roith@tum.de> (TU Munich)
# =============================================================================


import numpy as np
from scipy.stats import norm, multivariate_normal


class Gaussian:
    def __init__(self, mean, variance):
        self.mean = mean
        if len(variance.shape) == 1:
            self.variance = np.diag(variance)
        elif len(variance.shape) == 2:
            if variance.shape[1] == 1: 
                self.variance = np.diag(variance)
            elif variance.shape[0] == variance.shape[1]:
                self.variance = variance
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
    def __call__(self, x):
        return np.atleast_1d(multivariate_normal.pdf(x, mean=self.mean, cov=self.variance))
    
    def sample(self, N=1):
        return multivariate_normal.rvs(mean=self.mean, cov=self.variance, size=N)
    

class GaussianMixtureModel:
    def __init__(self, Gs, ws = None):
        self.nc = len(Gs)
        self.ws = np.ones(self.nc)/self.nc if ws is None else ws
        self.Gs = Gs

    def __call__(self, x):
        out = np.zeros(x.shape[:-1])
        for w, G in zip(self.ws, self.Gs):
            out += w * G(x)
        return out
    
    def score(self, x):
        out = 0
        for w, G in zip(self.ws, self.Gs):
            mean, cov = G.mean, G.variance
            out += w * G(x)[:, None] * np.linalg.solve(cov, (mean - x).T).T
        return out / self(x)[:, None]

class NoisyGaussianMixtureModel(GaussianMixtureModel):
    def __init__(self, GMM, sigma = 0.1):
        self.ws = GMM.ws
        self.nc = GMM.nc
        self.d  = GMM.Gs[0].variance.shape[0]
        self.sigma = sigma
        self.Gs = [Gaussian(G.mean, G.variance + sigma * np.eye(self.d)) for G in GMM.Gs]

    def x0_Tweedie(self, x):
        return x + self.sigma * self.score(x)
    

    def Tweedie_inverse(self,x):
        idx   = np.argmax(np.array([G(x) for G in self.Gs]),axis=0)
        out   = np.zeros_like(x)

        for i, (w, G) in enumerate(zip(self.ws, self.Gs)):
            mean, var = G.mean, G.variance
            op = (self.sigma) * np.linalg.inv(var)
            mean_tilde = op @ mean
            inv = np.linalg.solve(np.eye(self.d) - op, (x - mean_tilde).T).T
            out[idx==i, :] = inv[idx==i, :]
            #out += G(x)[:,None] * inv
        return out

def Gaussian_posterior(prior, A, y_meas, sigma_meas=0.1):
    m_prior, sigma_prior = prior.mean, prior.variance

    sigmainv   = np.linalg.inv(sigma_prior) + (1/sigma_meas*np.eye(2))@A@A.T
    sigma_post = np.linalg.inv(sigmainv)
    m_post     = np.linalg.solve(sigmainv, np.linalg.solve(sigma_prior, m_prior)+A.T@(y_meas/sigma_meas))
    post       = Gaussian(m_post, sigma_post)
    re

class GaussianMixture1D:
    def __init__(self, means, variances, weights):
        self.means = means
        self.variances = variances
        self.weights = weights

    def __call__(self, x):
        total = np.zeros_like(x)
        for mu, var, w in zip(self.means, self.variances, self.weights):
            total += w * np.exp(-0.5 * ((x - mu) / var)**2) / (var * np.sqrt(2 * np.pi))
        return total

    def decoder(self, z):
        if len(self.means) == 1:
            return self.means[0] + self.variances[0] * z
        elif len(self.means) == 2:
            return self.decoder_calc_approx(z)
        else:
            raise NotImplementedError("Decoder not implemented for more than 2 mixture components.")


    def decoder_calc_approx(self, z, alpha = 0.5,):
        u = norm.cdf(z)
        u = np.asarray(u, dtype=float)
        assert np.all((u > 0) & (u < 1)), "u must be in (0,1)"
        m1, m2 = self.means[0], self.means[1]
        s1, s2 = self.variances[0]**2, self.variances[1]**2

        mask1 = u <= alpha
        mask2 = ~mask1
        y0 = np.empty_like(u)

        if np.any(mask1):
            u1 = np.clip(u[mask1] / alpha, 1e-15, 1 - 1e-15)
            y0[mask1] = m1 + np.sqrt(s1) * norm.ppf(u1)

        # Second component
        if np.any(mask2):
            u2 = np.clip((u[mask2] - alpha) / (1 - alpha), 1e-15, 1 - 1e-15)
            y0[mask2] = m2 + np.sqrt(s2) * norm.ppf(u2)

        return y0
    

class Gaussian_noise_likelihood:
    def __init__(self, y, sigma=1., fwd = None):
        self.y     = y
        self.sigma = sigma
        self.fwd   = (lambda x: x) if fwd is None else fwd

    def __call__(self, x):
        return multivariate_normal.pdf(self.fwd(x), mean=self.y, cov=self.sigma)
    
class Bayes_posterior:
    def __init__(self,prior, likelihood, marginal=1.):
        self.prior      = prior
        self.likelihood = likelihood
        self.marginal   = marginal

    def __call__(self, x):
        return self.likelihood(x) * self.prior(x) / self.marginal
    

#%% sampling
def sample_decoder(num_samples, D):
    z_samples = np.random.normal(0, 1, num_samples)
    x_samples = D(z_samples)
    return x_samples