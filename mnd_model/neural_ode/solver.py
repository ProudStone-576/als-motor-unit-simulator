from torchdiffeq import odeint

def step_degeneration(ode_func, z0, t_span):
    """
    z0: initial degeneration state
    t_span: tensor([t0, t1])
    """
    zt = odeint(ode_func, z0, t_span)
    return zt[-1]
