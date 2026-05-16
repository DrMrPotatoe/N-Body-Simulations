from __future__ import annotations
import math
import numpy as np


class Point:
    '''
    A point at (x, y) with mass m with ID id
    '''
    def __init__(self, x: float, y: float, mass: float, ID: int = None, density: float = 0.1):
       
       self.x = x
       self.y = y

       self.vx = 0
       self.vy = 0

       self.fx = 0
       self.fy = 0

       self.m = mass

       self.id = ID

    def __str__(self):
        '''Formated string putput'''
        return f'P {self.id:5.0f}: ({self.x:10.6f}, {self.y:10.6f}); m={self.m:10.6f}; F=({self.fx:10.6f}, {self.fy:10.6f})'
    
    def mu(self, G= 1):
        '''Returns mu of the body'''
        return self.m * G
    
    def radius(self, density: float = 0.1):
        self.r = np.sqrt(self.m / (density * np.pi))

    def set_force(self, fx, fy):
        ''' Defines force'''
        self.fx = fx
        self.fy = fy

    def distance2(self, other: Point) -> float:
        '''Distance^2 to another point'''
        other_x, other_y = other.x, other.y
        return (other_x - self.x) * (other_x - self.x) + (other_y - self.y) * (other_y - self.y)

    def distance_to(self, other: Point) -> float:
        '''Distance to another point'''
        other_x, other_y = other.x, other.y
        return math.hypot((other_x - self.x), (other_y - self.y))

    def distance2_xy(self, x: float, y: float) -> float:
        ''' Distance^2 to xy coods'''
        return ((self.x - x) * (self.x - x)) + ((self.y - y) * (self.y - y))
    
    def distance_to_xy(self, x: float, y: float) -> float:
        ''' Distance to xy coords'''
        return math.hypot((self.x - x), (self.y - y))

    def collides(self, other:Point) -> bool:
        ''' Whether this point and another point intersect (for collisions)'''
        c_rad = self.r + other.r # Collision Radius
        return (self.distance2(other=other)) < (c_rad*c_rad)
    
    def force_between(self, point:Point, G= 1) -> tuple[float, float]:
        ''' Calculates the force between it and another point'''
        d = self.distance_to(point)
        d2 = d * d
        df = - self.mu(G) * point.m / d2
        ux = (self.x - point.x)/d
        uy = (self.y - point.y)/d
        fx = df * ux
        fy = df * uy
        return fx, fy

    def force_between_xy(self, x: float, y: float, m: float, G: float = 1., eps: float = 1e-3) -> tuple[float, float]:
        ''' Calculates the force between it and another point'''
        d2 = self.distance2_xy(x, y) + eps * eps
        d = math.sqrt(d2)
        df = self.mu(G) * m / d2
        fx = df * (x - self.x)/d
        fy = df * (y - self.y)/d
        return fx, fy

    def add_force(self, fx=0, fy=0):
        ''' Adds the forces to the already present one on the point'''
        self.fx += fx
        self.fy += fy

    def reset_force(self, fx=0, fy=0):
        ''' Resets the force back to (fx, fy)'''
        self.fx, self.fy = fx, fy

    def update_position_euler(self, dt):
        ''' Uses Euler integration to update the position of the point'''
        ax, ay = self.fx / self.m, self.fy / self.m

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx += ax * dt
        self.vy += ay * dt

    def half_kick(self, dt: float):
        ''' Give a half kick to the point'''
        self.vx += 0.5 * self.fx / self.m * dt
        self.vy += 0.5 * self.fy / self.m * dt

    def drift(self, dt: float):
        ''' Update Position using the velocity'''
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, ax, size=10, style='o'):
        ''' Draws the point on the plot'''
        ax.scatter(self.x, self.y, s=size)
    