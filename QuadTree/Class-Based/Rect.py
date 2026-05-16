from __future__ import annotations
import math
import matplotlib.pyplot as plt
from Point import Point

class Rect:
    '''
    A rectangle centered at (x, y) with width w
    '''
    def __init__(self, cx: float, cy: float, w: float):
        self.cx = cx
        self.cy = cy
        self.w = w

        self.N = cy + w/2
        self.S = cy - w/2
        self.W = cx - w/2
        self.E = cx + w/2

    def __str__(self):
        return f'({self.N:.4f}, {self.W:.4f}), ({self.S:.4f}, {self.E:.4f})'
        
    def contains(self, point:Point) -> bool:
        '''
        Whether a point is inside this
        '''
        point_x, point_y = point.x, point.y
        return (point_x >= self.W and
                point_x < self.E and 
                point_y > self.S and 
                point_y <= self.N)
    
    def intersects(self, other:Rect) -> bool:
        '''
        Whether a rectangole intersects this
        '''
        return not (self.E < other.W or
                    other.E < self.W or
                    self.N < other.S or
                    other.N < self.S)
    
    def distance2(self, other:Point):
        '''Distance^2 to a point from the centre'''
        other_x, other_y = other.x, other.y
        return (other_x - self.cx) * (other_x - self.cx) + (other_y - self.cy) * (other_y - self.cy)

    def distance_to(self, other: Point) -> float:
        '''Distance to a point from the centre'''
        other_x, other_y = other.x, other.y
        return math.hypot(other_x - self.cx, other_y - self.cy) 

    def Quadrant(self, point: Point) -> str:
        '''Find which quadant the point belongs into'''
        point_x, point_y = point.x, point.y
        cx, cy = self.cx, self.cy
        if self.contains(point) == False:
            return 'None'
        elif (point_x >= cx) and (point_y < cy): return 'SE'
        elif (point_x >= cx) and (point_y >= cy): return 'NE'
        elif (point_x < cx) and (point_y < cy): return 'SW'
        elif (point_x < cx) and (point_y >= cy): return 'NW'
        else: return 'None'

    def draw(self, ax, c='k', lw=1):
        ''' Draws the breakdown of the quadtree'''
        x1, x2 = self.W, self.E
        y1, y2 = self.N, self.S
        plt.plot()
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c, linewidth=lw)
