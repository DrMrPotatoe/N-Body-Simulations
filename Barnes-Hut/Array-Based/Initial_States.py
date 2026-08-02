import numpy as np
import math

from Config import Config
from State import State
from utils import split_integer

def initial_state_random(state: State, cfg: Config, extent= 1, seed= 42,) -> None:
    ''' Geneates a random distribuition of points'''
    
    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)

    state.particles.x[:] = rng.uniform(low= -extent,
                                       high= extent,
                                       size= n)
    state.particles.y[:] = rng.uniform(low= -extent, 
                                       high= extent,
                                       size= n)
    state.particles.mass[:]= rng.normal(loc= cfg.seconday_mass,
                                        size= n)
        
def initial_state_uniform(state: State, cfg: Config, sigma= 0.25, seed= 42):
    ''' Generates a cluster of points centered around 0.0'''

    rng = state.rng_generator

    n = len(state.particles.x)

    state.particles.x[:] = rng.normal(loc= 0.0, 
                                      scale= sigma,
                                      size= n)
    state.particles.y[:] = rng.normal(loc= 0.0, 
                                      scale= sigma, 
                                      size= n)
    state.particles.mass[:]= rng.lognormal(mean= 0,
                                           sigma= 0.5,
                                           size= n)
    
def initial_state_cluster_outliers(state= State, cluster_frac= 0.8, sigma= 0.05, extent= 1, seed= 42):
    ''' Generates a tight cluster around 0, 0 with random points all around'''
    
    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)
    n_cluster = int(cluster_frac * n)
    n_random = n - n_cluster

    cluster_x = rng.normal(loc= 0.0, scale= sigma, size= n_cluster)
    cluster_y = rng.normal(loc= 0.0, scale= sigma, size= n_cluster)

    random_x = rng.uniform(low= -extent, high= extent, size= n_random)
    random_y = rng.uniform(low= -extent, high= extent, size= n_random)

    state.particles.x[:] = np.concatenate([cluster_x, random_x])
    state.particles.y[:] = np.concatenate([cluster_y, random_y])
    state.particles.mass[:]= rng.uniform(size= n)

def initial_state_galaxies(state: State, separation= 5, galaxies=3, sigma= 0.3, seed= 42):
    '''Generates clusters of points separation distance away'''
    from utils import split_integer
    rng = np.random.default_rng(seed= seed)

    n = len(state.particles.x)

    n_cluster_x = []
    n_cluster_y = []
    n_per_cluster = split_integer(n, galaxies)
    offset = rng.uniform(high= np.pi)

    for cluster in range(galaxies):
        n_cluster = n_per_cluster[cluster]

        cluster_x = separation * np.cos(cluster / galaxies * 2*np.pi + offset) 
        cluster_y = separation * np.sin(cluster / galaxies * 2*np.pi + offset)

        n_cluster_xi = rng.normal(loc= cluster_x, scale= sigma, size= n_cluster)
        n_cluster_yi = rng.normal(loc= cluster_y, scale= sigma, size= n_cluster)

        n_cluster_x.append(n_cluster_xi)
        n_cluster_y.append(n_cluster_yi)

    state.particles.x[:] = np.concatenate(n_cluster_x)
    state.particles.y[:] = np.concatenate(n_cluster_y)
    state.particles.mass[:]= rng.uniform(size= n)


def generate_initial_state(state: State, cfg: Config) -> None:
    ''' Generates the initial state of the system (position, velocity, mass)'''

    particles = state.particles
    rng = state.rng_generator

    major_bodies = cfg.main_bodies
    minor_bodies = cfg.n_particles - major_bodies
    random_bodies = int(minor_bodies * cfg.outlier_ratio) 
    minor_bodies -= random_bodies
    n_clusters = cfg.n_clusters

    cluster_sigma = cfg.cluster_sigma
    total_extent = cfg.generation_extent

    if n_clusters <= 0:
        # If random dist:
        extent = total_extent / 2
        n = cfg.n_particles 
        particles.x[:] = rng.uniform(low= -extent,
                                               high= extent,
                                               size= n)
        particles.y[:] = rng.uniform(low= -extent, 
                                               high= extent,
                                               size= n)
        particles.mass[:]= rng.lognormal(mean= cfg.minor_mass,
                                               sigma= cfg.mass_sigma,
                                               size= n)

        if cfg.initiate_orbiting:
            vmax = math.pow(extent, 1/4)
            particles.vx[:] = rng.uniform(low= -vmax,
                                          high= vmax,
                                          size=n)
            particles.vy[:] = rng.uniform(low= -vmax,
                                          high= vmax,
                                          size=n)

        return
    
    elif n_clusters == 1:
        cluster_x, cluster_y = [0], [0]
        cluster_vx, cluster_vy = [0], [0]

    else: 
        offset = rng.uniform(high= np.pi)
        cluster_separation = cfg.cluster_separation

        cluster_x = []
        cluster_y = []

        cluster_vx = []
        cluster_vy = []

        for i in range(n_clusters):

            cluster_angle = i / n_clusters * 2*np.pi + offset

            cluster_i_x = cluster_separation * np.cos(cluster_angle) 
            cluster_i_y = cluster_separation * np.sin(cluster_angle)

            cluster_x.append(cluster_i_x)
            cluster_y.append(cluster_i_y)

            if cfg.initiate_orbiting:
                orbit_speed = math.sqrt(cfg.G * cfg.main_mass * cfg.main_bodies / cluster_separation)
                cluster_vx.append(-orbit_speed * np.sin(cluster_angle) * (1 - cfg.speed_variation))
                cluster_vy.append(orbit_speed * np.cos(cluster_angle) * (1 - cfg.speed_variation))
            else:
                cluster_vx.append(0)
                cluster_vy.append(0)



    major_bodies_per_cluster = split_integer(total= major_bodies, n= n_clusters)
    minor_bodies_per_cluster = split_integer(total= minor_bodies, n= n_clusters)

    particle = 0

    mu = cfg.G * cfg.main_mass

    for cluster in range(n_clusters):
        for major in range(major_bodies_per_cluster[cluster]):
            if major == 0:

                particles.x[particle] = cluster_x[cluster]
                particles.y[particle] = cluster_y[cluster]

                particles.vx[particle] = cluster_vx[cluster]
                particles.vy[particle] = cluster_vy[cluster]

                particles.mass[particle] = cfg.main_mass

                particle += 1

            else:
                particle_r = max(abs(rng.normal(loc= 0, scale= cluster_sigma)), cfg.eps)
                particle_theta = rng.uniform(low= 0, high=2*np.pi)

                particles.x[particle] = cluster_x[cluster] + particle_r * np.cos(particle_theta)
                particles.y[particle] = cluster_y[cluster] + particle_r * np.sin(particle_theta)

                if cfg.initiate_orbiting:
                    orbit_speed = math.sqrt(mu / particle_r)
                else:
                    orbit_speed = 0

                particles.vx[particle] = cluster_vx[cluster] + (-orbit_speed * np.sin(particle_theta) * (1+rng.normal(loc= 0, scale= cfg.speed_variation)))
                particles.vy[particle] = cluster_vy[cluster] + (orbit_speed * np.cos(particle_theta) * (1+rng.normal(loc= 0, scale= cfg.speed_variation)))

                particles.mass[particle] = cfg.main_mass / cfg.main_bodies

                particle += 1

        for _ in range(minor_bodies_per_cluster[cluster]):

            particle_r = max(abs(rng.normal(loc= 0, scale= cluster_sigma)), cfg.eps)
            particle_theta = rng.uniform(low= 0, high=2*np.pi)

            particles.x[particle] = cluster_x[cluster] + particle_r * np.cos(particle_theta)
            particles.y[particle] = cluster_y[cluster] + particle_r * np.sin(particle_theta)

            if cfg.initiate_orbiting:
                orbit_speed = math.sqrt(mu / particle_r)
            else:
                orbit_speed = 0

            particles.vx[particle] = cluster_vx[cluster] + (-orbit_speed * np.sin(particle_theta) * (1+rng.normal(loc= 0, scale= cfg.speed_variation)))
            particles.vy[particle] = cluster_vy[cluster] + (orbit_speed * np.cos(particle_theta) * (1+rng.normal(loc= 0, scale= cfg.speed_variation)))

            particles.mass[particle] = rng.lognormal(mean= cfg.minor_mass, sigma= cfg.mass_sigma)

            particle += 1


    for _ in range(random_bodies):

        extent = total_extent / 2
        n = cfg.n_particles 
        particles.x[particle] = rng.uniform(low= -extent, high= extent)
        particles.y[particle] = rng.uniform(low= -extent, high= extent)
        particles.mass[particle]= rng.lognormal(mean= cfg.minor_mass, sigma= cfg.mass_sigma)

        if cfg.initiate_orbiting:
            vmax = math.pow(extent, 1/4)
            particles.vx[particle] = rng.uniform(low= -vmax, high= vmax)
            particles.vy[particle] = rng.uniform(low= -vmax, high= vmax)

        particle += 1




