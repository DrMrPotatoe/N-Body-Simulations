# Array-based Barnes-Hut algorithm

A high performance implementation of the Barnes-Hut algorithm using arrays memory layout

The simulator stores all the particle and tree information inside contiuous Numpy arrays to minimize overhead. The implementation serves as a foundation for further optimization using tools such as GPU, parallel or Numba implementations.

---

## Features
 - Structure-Of-Arrays architecture
 - Contiguous NumPy storage
 - Iterative QuadTree Construction
 - Iterative Barnes-Hut tree traversal
 - Leapfrog integration
 - Particle collisions and merging
 - Escape Detection
 - Configurable Initial Conditiona
 - Automatic Video generation

---

## Project Structure

```
ArrayBased/
│
├── main.py 
│   └─ Main simulation loop
├── Config.py
│   └─ Simulation parameters
├── State.py
│   └─ Simulation state and data structures
│   
├── Forces.py
├── Integrator.py
├── Escaped_Particles.py
├── Simulate.py
├── Tree_Statistics.py
├── Tree_Tests.py
├── Video_Making.py
├── Visuals.py
├── utils.py
│
└── README.md
```

---

## Architecture

The simulator follows a Structure-of-Arrays design.

Rather than representing particles as Python objects, each physical property is stored in its own NumPy array. For example,

- positions
- velocities
- masses
- radii
- accumulated forces

are all stored independently.

Similarly, quadtree nodes are stored as arrays containing quantities such as

- node centre
- width
- total mass
- centre of mass
- child indices
- particle lists
- subtree statistics

Nodes are referenced by integer indices instead of object references.

---

## Quadtree

The Barnes–Hut tree is constructed iteratively.

Each node stores:

- spatial boundaries
- total mass
- centre of mass
- child indices
- linked list of contained particles

Force evaluation traverses the tree using an explicit stack and applies the Barnes–Hut opening-angle criterion to determine whether a node should be approximated or explored further.

---

## Simulation Pipeline

A typical simulation step consists of:

1. Build the quadtree.
2. Compute node masses and centres of mass.
3. Evaluate gravitational forces.
4. Resolve collisions (optional).
5. Advance particle positions and velocities.
6. Remove escaped particles (optional).
7. Render the current frame (optional).

---

## Configuration

Simulation parameters are defined in `Config.py`.

Typical parameters include:

- gravitational constant
- timestep
- opening angle
- softening length
- simulation duration
- collision settings
- initial conditions

---

## Running

Run the simulation using

```bash
python main.py
```

Simulation behaviour can be modified by changing the configuration though the Config object, or selecting different initial condition generators.

---

## Future Improvements

Possible extensions include:

- Numba compilation
- Parallel force computation
- SIMD-friendly optimisations
- GPU acceleration
- Adaptive timesteps(Probably not worth it)

## Ancknowledgements
This code is written completely by me, adapted from the class-based implementation in this same repo.

This readme.md was outlined by chatgpt and writen and edited by me
