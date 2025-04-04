# bianchi's paper: https://ieeexplore.ieee.org/abstract/document/840210?casa_token=P57rDYu5jzMAAAAA:1FH23s-9kRrLfYbE1O4i1VSM5OYrQ-ycrJxsb4lp76iHpQAsd9VEMzLup93LVAt0ySJeufl6
# source: https://tetcos.com/pdf/WiFi-NetSim-results-vs-Bianchi-predictions.pdf
def compute_collision_probability(n: int, m: int, cw_min: int) -> float:
    """
    Calculates the collision probability (p) according to Bianchi's model.

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
        Collision probability (p)
    """

    w = cw_min
    alpha = 0.95  # Fixed point relaxation parameter
    thresh = 1e-9  # Convergence threshold

    tau = 2 / (w + 1)  # Initial guess for tau
    p = 1 - (1 - tau) ** (n - 1)  # initial collision prob.

    for _ in range(1000):  # Maximum iterations
        p_old = p
        numerator = 2 * (1 - 2 * p)
        denominator = (1 - 2 * p) * (w + 1) + p * w * (1 - (2 * p) ** m)

        tau = numerator / denominator if denominator != 0 else 0

        p_int = 1 - (1 - tau) ** (n - 1)  # Intermediate collision probability

        p = (1 - alpha) * p_int + alpha * p

        if abs(p - p_old) < thresh:  # Convergence check
            break
    return p
