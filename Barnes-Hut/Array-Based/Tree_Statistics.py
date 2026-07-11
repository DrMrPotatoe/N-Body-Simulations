import numpy as np

from State import State
from Config import Config

def print_tree_statistics(state: State, cfg: Config) -> None:
    """Print statistics about the current Barnes-Hut tree."""

    nodes = state.nodes
    particles = state.particles

    node_count = state.node_count

    if node_count == 0:
        print("Tree is empty.")
        return

    # Only consider allocated nodes
    leaf = nodes.leaf[:node_count]
    depth = nodes.depth[:node_count]
    subtree = nodes.subtree_particle_count[:node_count]
    local = nodes.local_particle_count[:node_count]

    n_leaf = int(leaf.sum())
    n_internal = node_count - n_leaf

    max_depth = int(depth.max())
    avg_depth = float(depth.mean())

    max_particles_leaf = int(local[leaf].max()) if n_leaf else 0
    avg_particles_leaf = float(local[leaf].mean()) if n_leaf else 0.0

    alive_particles = int(particles.alive.sum())

    print("=" * 60)
    print("Barnes-Hut Tree Statistics")
    print("=" * 60)

    print(f"Allocated nodes               : {node_count}")
    print(f"Leaf nodes                    : {n_leaf}")
    print(f"Internal nodes                : {n_internal}")

    print(f"Alive particles               : {alive_particles}")
    print(f"Particles in root             : {subtree[state.root]}")

    print(f"Maximum depth                 : {max_depth}")
    print(f"Average depth                 : {avg_depth:.2f}")

    print(f"Max particles/leaf            : {max_particles_leaf}")
    print(f"Avg particles/leaf            : {avg_particles_leaf:.2f}")

    print(f"Node utilisation              : {100 * node_count / len(nodes.x):6.2f}%")

    if hasattr(state, "node_visits"):
        print(f"Node visits                   : {state.node_visits:,}")
        print(f"    per particle              : {state.node_visits / alive_particles:.2f}")

    if hasattr(state, "p2p_interactions"):
        print(f"Particle-Particle interactions: {state.p2p_interactions:,}")
        print(f"    per particle              : {state.p2p_interactions / alive_particles:.2f}")

    if hasattr(state, "p2n_interactions"):
        print(f"Particle-Node interactions    : {state.p2n_interactions:,}")
        print(f"    per particle              : {state.p2n_interactions / alive_particles:.2f}")

    occupied_leaves = local[leaf]

    print(f"Empty leaves                  : {(occupied_leaves == 0).sum()}")
    print(f"Single-particle leaf          : {(occupied_leaves == 1).sum()}")
    print(f"Full leaves                   : {(occupied_leaves == cfg.node_capacity).sum()}")
    
    print("\nNodes per depth")
    for d in range(max_depth + 1):
        n = (depth == d).sum()
        print(f"  {d:2d}: {n}")

    print("=" * 60)