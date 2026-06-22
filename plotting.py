# =============================================================================
# .py
# =============================================================================
# Authors:
#   - Tim Roith <tim.roith@tum.de> (TU Munich)
# =============================================================================


import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def plot_ellipse(cov, m, ax=None, facecolor='none', **kwargs):
    if ax is None:
        ax = plt.gca()
        
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * np.sqrt(vals)
    
    ellipse = Ellipse(xy=m, width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)
    
    ax.add_patch(ellipse)

    ax.plot(m[0], m[1], marker='*', color=kwargs.get('edgecolor', 'k'))
    return ellipse