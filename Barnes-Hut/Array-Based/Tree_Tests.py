import numpy as np

from Config import Config
from State import State
from QuadTree import build_tree, validate_tree
from Initial_States import initial_state_cluster_outliers
from Forces import compute_acceleration, compute_acceleration_brute_force

def test_clustered(state: State, cfg: Config, n_tests: int = 100, seed: int=  42,) -> None:
    ''' Test for big clusters of points in close proximity'''

    for _ in range(n_tests):

        initial_state_cluster_outliers(state= state, seed= seed)

        build_tree(state= state, cfg= cfg)
        validate_tree(state= state)

    print(f" {n_tests} Cluster Tests passed")


import numpy as np


def test_accelerations(state: State, cfg: Config) -> None:
    """
    Compare Barnes-Hut accelerations against brute-force accelerations.

    Prints absolute and relative errors for every alive particle together
    with overall statistics.
    """

    particles = state.particles

    # Barnes-Hut

    particles.ax.fill(0.0)
    particles.ay.fill(0.0)

    compute_acceleration(state, cfg)

    bh_ax = particles.ax.copy()
    bh_ay = particles.ay.copy()

    # Brute force

    particles.ax.fill(0.0)
    particles.ay.fill(0.0)

    compute_acceleration_brute_force(state, cfg)

    bf_ax = particles.ax.copy()
    bf_ay = particles.ay.copy()

    # Compare

    alive = particles.alive

    bh = np.column_stack((bh_ax[alive], bh_ay[alive]))
    bf = np.column_stack((bf_ax[alive], bf_ay[alive]))

    error = bh - bf

    abs_error = np.linalg.norm(error, axis=1)
    true_acc = np.linalg.norm(bf, axis=1)

    # Avoid divide-by-zero
    rel_error = abs_error / np.maximum(true_acc, 1e-30)

    # Global L2 error
    global_rel_l2 = (
        np.linalg.norm(error)
        / np.linalg.norm(bf)
    )

    print(f"\nAcceleration Error Report (θ = {cfg.theta})")
    print("-" * 60)

    print(f"Alive particles        : {len(abs_error)}")

    print("\nAbsolute Error")
    print(f"  Mean                : {np.mean(abs_error):.6e}")
    print(f"  Median              : {np.median(abs_error):.6e}")
    print(f"  Max                 : {np.max(abs_error):.6e}")

    print("\nAcceleration Magnitude")
    print(f"  Mean                : {np.mean(true_acc):.6e}")
    print(f"  Median              : {np.median(true_acc):.6e}")

    print("\nRelative Error")
    print(f"  Mean                : {np.mean(rel_error):.6e}")
    print(f"  Median              : {np.median(rel_error):.6e}")
    print(f"  Max                 : {np.max(rel_error):.6e}")

    print("\nRelative Error Percentiles")
    print(f"  50%                 : {np.percentile(rel_error, 50):.6e}")
    print(f"  90%                 : {np.percentile(rel_error, 90):.6e}")
    print(f"  95%                 : {np.percentile(rel_error, 95):.6e}")
    print(f"  99%                 : {np.percentile(rel_error, 99):.6e}")

    print("\nGlobal Error")
    print(f"  L2 Relative Error   : {global_rel_l2:.6e}")
    print(f"  Mean Abs / Mean Acc : {np.mean(abs_error)/np.mean(true_acc):.6e}")

    print("\nWorst Particles")
    worst = np.argsort(abs_error)[-10:][::-1]

    print(
        f"{'ID':>6}"
        f"{'|a|':>14}"
        f"{'Abs Err':>14}"
        f"{'Rel Err':>14}"
    )

    for idx in worst:
        print(
            f"{np.flatnonzero(alive)[idx]:6d}"
            f"{true_acc[idx]:14.6e}"
            f"{abs_error[idx]:14.6e}"
            f"{rel_error[idx]:14.6e}"
        )


    print("-" * 60)
    print("\nParticles below error thresholds")

    for tol in [1e-3, 1e-2, 5e-2, 1e-1]:
        frac = np.mean(rel_error < tol)
        print(f"  < {tol:6.3f} : {100*frac:6.2f}%")