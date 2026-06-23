# Barnes–Hut QuadTree N-Body Simulation

A 2D Barnes–Hut N-body gravitational simulation implemented in Python using a recursive quadtree structure.

This project was built as a prototype for experimenting with:

- Barnes–Hut force approximation
- Recursive quadtrees
- Collision detection and body merging
- Integration (Kick-Drift-Kick / Leapfrog)
- GIF and MP4 simulation rendering

---

## Features

- Recursive quadtree implementation
- Barnes–Hut approximation for efficient force computation
- Gravitational softening support
- Collision detection using spatial queries
- Inelastic body merging with momentum conservation
- Leapfrog integration
- Output of single frame quadtree visualization
- Animated GIF generation
- MP4 video export using FFmpeg

---

## Barnes–Hut Approximation

The Barnes–Hut algorithm reduces the computational complexity of force calculations from:

```text
O(N²)
```

to approximately:

```text
O(N log N)
```

by approximating distant clusters of particles as a single center-of-mass body.

The approximation criterion used is:

```text
s / d < θ
```

Where:

- `s` = node width
- `d` = distance to node COM
- `θ` = opening angle threshold

Smaller `θ` values improve accuracy but reduce performance.

---

## Example Output

The simulation generates orbital systems similar to:

- Star systems
- Particle accretion disks
- Galaxy-like structures
- Chaotic gravitational interactions

Output formats:

- `QuadTree.gif`
- `QuadTree.mp4`

---

## Dependencies

Install required packages:

```bash
pip install numpy matplotlib
```

For MP4 rendering, FFmpeg must also be installed.

---

## File Structure

```text
project/
│
├── main.py
├── QuadTree.py
├── Point.py
├── Rect.py
├── Circ.py
├── QuadTree_Interface.py
├── utils.py
├── README.md
└── Example_Outputs
    ├── QuadTree.svg
    ├── QuadTree.gif
    └── QuadTree.mp4
```

---

## Core Classes

### `Point`

Represents a body in the simulation.

Stores:

- Position
- Velocity
- Force
- Mass
- Radius

Includes:

- Distance calculations
- Force computation
- Collision checks
- Integration methods

---

### `Rect`

Axis-aligned square boundary used by quadtree nodes.

Supports:

- Point containment
- Rectangle intersection
- Quadrant detection

---

### `Circ`

Circular query region used for collision searches.

Supports:

- Circle-point intersection
- Circle-rectangle intersection
- Circle-circle intersection

---

### `QuadTree`

Recursive Barnes–Hut quadtree node.

Stores:

- Child nodes
- Center of mass
- Total node mass
- Node bounds
- Point capacity

Implements:

- Recursive insertion
- Force approximation
- Spatial querying
- Tree visualization

---

### `Quad_Tree_Interface`

Simulation driver and visualization interface.

Handles:

- System generation
- Tree rebuilding
- Force computation
- Collision handling
- Time stepping
- Rendering

---



## Running the Simulation

In main.py:

```python
testmethod = Quad_Tree_Interface(1000)

testmethod.capacity = 1
testmethod.T1 = 1000
testmethod.collide = True
testmethod.density = 1e4
testmethod.verbose = 0

testmethod.create_points_orbiting()

testmethod.video_simulate(fps=60)
```

Run with:

```bash
python main.py
```

---

## Simulation Parameters

### General

| Parameter | Description |
|---|---|
| `npoints` | Number of orbiting bodies |
| `dt` | Timestep |
| `T1` | Total simulation duration |
| `G` | Gravitational constant |
| `eps` | Softening parameter |
| `theta` | Barnes–Hut opening angle |

---

### Tree Parameters

| Parameter | Description |
|---|---|
| `capacity` | Max points per leaf node |
| `bounds` | Root node boundary |

---

### Collision Parameters

| Parameter | Description |
|---|---|
| `density` | Used to derive body radii |
| `collide` | Enable/disable collisions |
| `escape_radius` | Remove distant bodies |

---

## Integration Scheme

The simulation uses a Kick-Drift-Kick leapfrog integrator:

1. Half velocity update
2. Position update
3. Force recomputation
4. Final half velocity update

This method provides significantly better energy stability than standard Euler integration.

---

## Collision Handling

Collision detection is accelerated using quadtree spatial queries.

When two bodies collide:

- Mass is conserved
- Momentum is conserved
- Bodies merge into one object
- Radius is recalculated

---

## Rendering

### GIF Export

```python
testmethod.gif_simulate()
```

Uses:

```python
PillowWriter
```

---

### MP4 Export

```python
testmethod.video_simulate(filename, fps)
```

Uses:

```python
FFMpegWriter
```

---

## Current Limitations

- 2D only
- Single-threaded
- Recursive implementation
- No SIMD/vectorized force kernel
- No GPU acceleration
- Tree rebuilt every frame
- Collision handling is pairwise sequential

---

## Future Improvements

Potential next steps:

- Full vectorized force traversal
- Iterative tree traversal
- Morton/Z-order indexing
- Cython or C++ backend
- OpenGL rendering
- 3D octree implementation
- Fast multipole methods (FMM)

---

## References

Inspired by:

- The Coding Train  
  https://thecodingtrain.com/challenges/98-quadtree

- SciPython Quadtree Implementation  
  https://scipython.com/blog/quadtrees-2-implementation-in-python/

- BarnesHut-py  
  https://github.com/alessialin/BarnesHut-py

- DeadlockCode Barnes–Hut  
  https://github.com/DeadlockCode/barnes-hut

---

## License

Dont worry about it,use it as you wish. But if you do use it, credit is nice. 