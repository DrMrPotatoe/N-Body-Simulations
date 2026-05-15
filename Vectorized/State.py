'''
Vectorized Implementation of A quadtree
'''

from __future__ import annotations
import numpy as np
import math


#####################################
# Simulation Data ###################
#####################################

# Number of points
P_n= 100

# Number of Nodes
N_n = P_n * 8

# Node Capacity
N_capacity = 4


#####################################
# Particle Data #####################
#####################################

# Particle x and y position
P_x = np.full(P_n, np.nan)
P_y = np.full(P_n, np.nan)

# Particle x and y velocity
P_vx = np.full(P_n, np.nan)
P_vy = np.full(P_n, np.nan)

# Particle x and y force
P_fx = np.full(P_n, np.nan)
P_fy = np.full(P_n, np.nan)

# Particle mass and radius
P_mass = np.full(P_n, np.nan)
P_rad = np.full(P_n, np.nan)

# Next particle in node:
P_next = np.full(P_n, -1, dtype= np.int32)


#####################################
# Node Data #########################
#####################################

# TREE GEOMETRY
# Node x and y centre positions
N_x = np.full(N_n, np.nan)
N_y = np.full(N_n, np.nan)
# Node width 
N_width = np.full(N_n, np.nan)

# TREE MASS ACCUMULATION
# Node centre of mass x and y positions
N_com_x = np.zeros(N_n)
N_com_y = np.zeros(N_n)
# Node mass
N_mass = np.zeros(N_n)

# TREE TOPOLOGY
# Node children (+0->TL, +1->TR, +2->BL, +3->BR)
N_first_child = np.full(N_n, -1, dtype= np.int32)
# Node Parent
N_parent = np.full(N_n, -1, dtype= np.int32)

# Tree Occupation
# Node Points
N_first_particle = np.full(N_n, -1, dtype= np.int32)
# Number of points in the node
N_point_count = np.zeros(N_n, dtype= np.int32)
# Node is leaf
N_is_leaf = np.ones(N_n, dtype= np.bool_)




print('EOF')