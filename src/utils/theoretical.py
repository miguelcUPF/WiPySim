def bianchi_collision_probability(n: int, tau: float) -> float:
    """
    Computes the Bianchi collision probability in a saturated IEEE 802.11 network.

    Parameters
    ----------
    n: int
        Number of stations.
    tau: float
        Transmission probability of a station.

    Returns
    -------
    float
        Bianchi collision probability.
    """
    if n <= 1:
        return 0.0
    return 1 - (1 - tau) ** (n - 1)


def bianchi_transmission_probability(n: int, m: int, cw_min: int) -> float:
    """
    Calculates the transmission probability (tau) according to Bianchi's model.

    Parameters
    ----------
    n: int
        Number of stations.
    m: int
        Maximum backoff stage.
    cw_min: int
        Minimum contention window size.

    Returns
    -------
    float
        Transmission probability (tau).
    """

    w = cw_min
    if m == 0:
        return 2 / (w + 1)

    tau = 2 / (w + 1)  # Initial guess for tau
    p = bianchi_collision_probability(n, tau)  # initial collision prob.

    for _ in range(1000):  # Maximum iterations
        tau_old = tau
        numerator = 2 * (1 - 2 * p)
        denominator = (1 - 2 * p) * (w + 1) + p * w * (1 - (2 * p) ** m)
        tau = numerator / denominator if denominator != 0 else 0
        p = bianchi_collision_probability(n, tau)
        if abs(tau - tau_old) < 1e-9:  # Convergence check
            break
    return tau

def compute_collision_probability(n: int, m: int, cw_min: int) -> float:
    tau = bianchi_transmission_probability(n, m, cw_min)
    return bianchi_collision_probability(n, tau)