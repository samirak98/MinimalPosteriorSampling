# =============================================================================
# .py
# =============================================================================
# Authors:
#   - Tim Roith <tim.roith@tum.de> (TU Munich)
# =============================================================================

class linear_fwd_operator:
    def __init__(self, A):
        self.A = A
    def __call__(self, x):
        return x @ self.A.T
    
def funadd(f,g, a=1):
    return lambda x, t: f(x, t) + a*g(x, t)