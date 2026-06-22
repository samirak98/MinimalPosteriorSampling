# =============================================================================
# .py
# =============================================================================
# Authors:
#   - Samira Kabri <samira.kabri@desy.de> (DESY)
#   - Tim Roith <tim.roith@tum.de> (TU Munich)
# =============================================================================


import numpy as np
from scipy.stats import multivariate_normal

class Reverse_Markov:
    def __init__(self, sde, beta):
        self.sde = sde
        self.beta = beta
    
    def __call__(self, T, N, xT, save_int=10):
        x_history = []
        x_history.append(xT)
        
        x_running = xT
        
        for i in range(N):
            x_running = self.reverse_step(x = x_running, t = (N-i)*T/N, N = N)
            if i % save_int == 0:
                x_history.append(x_running)
        
        return np.array(x_history)
    
    def reverse_step(self, x, t, N):
        noise = np.random.normal(size = x.shape)
        b = self.beta(t)/N
        s = self.sde.score(x,t)
        
        return 1/(np.sqrt(1-b))*(x+b*s)+np.sqrt(b)*noise

 
class beta_linear:
    def __init__(self, beta_min=0.1, beta_max=20.):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

class alpha_linear:
    def __init__(self, beta):
        self.beta_min = beta.beta_min
        self.beta_max = beta.beta_max
    def __call__(self, t):
        return np.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))

def sigma_schedule(alpha):
    return lambda t: np.sqrt(1 - alpha(t)**2)


def get_scheduler_linear(beta):
    beta  = beta_linear() if beta is None else beta
    alpha = alpha_linear(beta)
    sigma = sigma_schedule(alpha)
    return alpha, beta, sigma


class sde:
    def __init__(self, alpha = None, beta = None, sigma = None):
        if alpha is None or sigma is None:
            self.alpha, self.beta, self.sigma = get_scheduler_linear(beta)

    def score(self, x, t):
        return np.zeros_like(x)

    def x0_tweedie(self, x, t):
       return (x + self.sigma(t)**2 * self.score(x,t)) / self.alpha(t)
    

class sde_Gauss(sde):
    def __init__(self, mean, var, alpha = None, beta = None, sigma = None):
        super().__init__(alpha = alpha, beta = beta, sigma = sigma)
        self.mean  = mean
        self.var   = var

    def __call__(self, x, t):
        var_t = self.alpha(t) ** 2 * self.var + (self.sigma(t) ** 2*np.eye(x.shape[-1]))
        mean_t = self.mean * self.alpha(t)
        return multivariate_normal.pdf(x, mean=mean_t, cov=var_t)

    def score(self, x, t):
        var_t = self.alpha(t) ** 2 * self.var + (self.sigma(t) ** 2*np.eye(x.shape[-1]))
        mean_t = self.mean * self.alpha(t)
        return -np.linalg.solve(var_t, (x-mean_t).T).T
    
    def apply_information(self, x, t, z):
        var_t = self.alpha(t) ** 2 * self.var + (self.sigma(t) ** 2*np.eye(x.shape[-1]))
        return - np.linalg.solve(var_t.T, z.T).T
    
class sde_likeli_Bayes(sde):
    def __init__(self, sde_prior, sde_post, marginal = 1.):
        self.sde_prior = sde_prior
        self.sde_post  = sde_post
        self.marginal  = marginal
        for n in ['alpha', 'beta', 'sigma']:
            setattr(self, n, getattr(sde_prior, n)) 

    def __call__(self, x, t):
        return self.sde_post(x, t) * self.marginal / self.sde_prior(x, t)
    
    def score(self, x, t):
        return self.sde_post.score(x, t) - self.sde_prior.score(x, t)
    

class sde_posterior_Bayes:
    def __init__(self, sde_prior, sde_likeli, marginal = 1.):
        super().__init__()
        self.sde_prior = sde_prior
        self.sde_likeli = sde_likeli
        self.marginal = marginal

    def __call__(self, x, t):
        return self.sde_likeli(x, t) * self.sde_prior(x, t) / self.marginal
    
    def score(self, x, t):
        return self.sde_prior.score(x, t) + self.sde_likeli.score(x, t)
    

# ---------------------------------------------------
# Likelihood approximations
class sde_likeli_approx(sde):
    def __init__(self, A, y_meas, beta = None, sigma_meas = 0.1):
        super().__init__(beta=beta)
        self.A          = A
        self.y_meas     = y_meas
        self.sigma_meas = sigma_meas

    def __call__(self, x, t):
        return np.zeros(x.shape[:-1])

class sde_likeli_ALD(sde_likeli_approx):
    def __call__(self, x, t):
        var = self.sigma_meas * np.eye(x.shape[-1])
        return multivariate_normal.pdf(x @ self.A.T, mean=self.y_meas, cov=var)

    def score(self, x, t):
        return 1/ self.sigma_meas * (self.y_meas - (x @ self.A.T)) @ self.A

class sde_likeli_Song(sde_likeli_approx):
    def score(self, x, t):
        z = np.random.normal(size=x.shape)
        y_noised = self.alpha(t) * self.y_meas + self.sigma(t) * (z @ self.A.T)
        return 1/self.sigma_meas *( y_noised - x @ self.A.T) @ self.A

class sde_likeli_DPS(sde_likeli_approx):
    def __init__(self, A, y_meas, sde_prior, beta = None, sigma_meas = 0.1):
        super().__init__(A, y_meas, beta = beta, sigma_meas = sigma_meas)
        self.sde_prior = sde_prior

    def __call__(self, x, t):
        x0 = self.sde_prior.x0_tweedie(x,t)
        var = self.sigma_meas * np.eye(x.shape[-1])
        return multivariate_normal.pdf(x0 @ self.A.T, mean=self.y_meas, cov=var)

    def score(self, x,t):
        x0 = (x + self.sigma(t)**2 * self.sde_prior.score(x,t)) / self.alpha(t)
        grad_x0 = (self.y_meas - x0 @ self.A.T) @ self.A / self.sigma_meas
        information = self.sde_prior.apply_information(x, t, grad_x0)
        return (grad_x0 + self.sigma(t)**2*information) / self.alpha(t) 