from __future__ import annotations
import numpy as np
from pathlib import Path
from solvers import N_Body_Solver



body_count = 10000
time_step = 6
time_end = 3600
mass_diff = 1000
grav_constant = 0.01
solver = N_Body_Solver(body_count,time_end,time_step)
solver.prepare()

print("EOF")