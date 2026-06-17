import equinox as eqx 

from .schedules import Schedule, GeomSchedule

class SLIPSParams(eqx.Module):
    """
    Parameters for the SLIPS algorithm.
    Object to be passed to the geom_slips function, which contains all
    the parameters for the algorithm.
    """
    # Dynamic params
    sigma: float
    
    # Dynamic named params
    schedule: Schedule = GeomSchedule()
    target_accept: float = 0.75
    step_min: float = 1e-2
    step_max: float = 1e2
    learning_rate: float = 2.0

    # Static named params
    n_mcmc_steps:   int   = eqx.field(default=32, static=True)
    n_chains:       int   = eqx.field(default=4, static=True)          
    n_init_steps:   int   = eqx.field(default=32, static=True)
    burn_in_ratio:  float = eqx.field(default=0.5, static=True)
    return_history: bool  = eqx.field(default=False, static=True)
