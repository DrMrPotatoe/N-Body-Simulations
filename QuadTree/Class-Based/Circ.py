from __future__ import annotations
import numpy as np
import matplotlib.patches as patches
from Point import Point
from Rect import Rect

class Circ:
    '''
    Circle at (x, y) with radius r
    '''
    def __init__(self, x:float, y:float, r:float):
        self.x = x
        self.y = y
        self.r = r
        self.r2 = r*r

    def contains(self, point: Point) -> bool:
        '''
        circle contains Point
        '''
        point_x, point_y = point.x, point.y
        d = (self.x - point_x) * (self.x - point_x) + (point_y - self.y) * (point_y - self.y)
        return d < self.r2
    
    def rect_intersect(self, other: Rect) -> bool:
        '''
        Circle intersects with rectangle
        '''
        cx, cy = self.x, self.y
        dx = cx - np.clip(cx, other.W, other.E)
        dy = cy - np.clip(cy, other.S, other.N)
        return (dx*dx + dy*dy) <= self.r2
    
    def circ_intersect(self, other:Circ) -> bool:
        '''
        this circle intersect with another
        '''
        dist2 = (other.x - self.x) * (other.x - self.x) + (other.y - self.y) * (other.y - self.y)
        return (dist2 < ((self.r + other.r) * (self.r + other.r)))
    
    def draw(self, ax, c='k', lw=1):
        '''Draws a circle'''
        ax.add_patch(patches.Circle([self.x, self.y], self.r, fill= False, c=c, linewidth= lw))

