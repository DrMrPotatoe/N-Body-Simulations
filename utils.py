import numpy as np

def generate_initial_state(nBodies: int, mu: float):
    """
    Generate random 2D Keplerian state vectors.
    """

    # --- orbital elements ---
    a_min, a_max = 0.1, np.log10(nBodies)             # semi-major axis range
    e_max = 0.95                         # strictly < 1
    a = a_min + (a_max - a_min) * np.random.rand(nBodies, 1)
    e = e_max * np.random.rand(nBodies, 1)

    nu = 2.0 * np.pi * np.random.rand(nBodies, 1)   # true anomaly
    omega = 2.0 * np.pi * np.random.rand(nBodies, 1)  # argument of periapsis

    # --- geometry ---
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu))

    # position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # rotate by argument of periapsis
    cosw = np.cos(omega)
    sinw = np.sin(omega)

    r_vec = np.hstack((
        cosw * x_orb - sinw * y_orb,
        sinw * x_orb + cosw * y_orb
    ))

    # --- velocities ---
    h = np.sqrt(mu * p)

    vr = (mu / h) * e * np.sin(nu)
    vt = (mu / h) * (1.0 + e * np.cos(nu))

    er = r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True)
    etheta = np.hstack((-er[:, 1:2], er[:, 0:1]))

    v_vec = vr * er + vt * etheta

    return r_vec.T, v_vec.T

def P_to_P_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    ''' 
    Point to Point Distance\n
    (x1, y1) coords of 1st point\n
    (x2, y2) coords of second point\n
    Returns float
    '''
    return np.hypot(x2-x1, y2-y1)

def P_in_Sc(px: float, py: float, cx: float, cy: float, ss: float) -> bool:
    '''
    Point inside Square \n

    (px, py) Point Coordinates \n
    (cx, cy) Coordinates of centre of square\n
    (ss) Size of the square
    '''
    w = cx - ss/2
    e = cx + ss/2
    s = cy - ss/2
    n = cy + ss/2

    return (px >= w and
            px < e and 
            py > s and 
            py <= n)

def S_intersects_S(x1: float, y1: float, s1: float, x2: float, y2: float, s2: float) -> bool:
    '''
    Intersection between 2 squares\n
    
    (x1, y1) Coordinates of centre of first square\n
    (s1) Size of the first square\n

    (x2, y2) Coordinates of centre of second square\n
    (s2) Size of the second square\n
    '''
    w1 = x1 - s1/2
    e1 = x1 + s1/2
    s1 = y1 - s1/2
    n1 = y1 + s1/2

    w2 = x2 - s2/2
    e2 = x2 + s2/2
    s2 = y2 - s2/2
    n2 = y2 + s2/2

    return not (w1 < e2 or
                w2 < e1 or
                s1 < n2 or
                s2 < n1)