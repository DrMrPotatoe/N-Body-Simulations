from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Point:
    '''
    A point at (x, y) with mass m with ID id
    '''
    def __init__(self, x: float, y: float, mass: float, ID: int = None, density: float = 0.1):
       self.x = x
       self.y = y
       self.m = mass
       self.id = ID
       self.r = ((mass*density)/(np.pi))**(1/2)

    def __str__(self):
        return f'P{self.id}: ({self.x:.4f}, {self.y:.4f}); m={self.m:.4f}'
    
    def distance2(self, other: Point) -> float:
        '''Distance^2 to another point'''
        other_x, other_y = other.x, other.y
        return np.square(self.x - other_x) + np.square(other_y - self.y)

    def distance_to(self, other: Point) -> float:
        '''Distance to another point'''
        other_x, other_y = other.x, other.y
        return np.hypot(other_x - self.x, other_y - self.y)
    
    def collides(self, other:Point) -> bool:
        '''
        Whether this point and another point intersect (for collisions)
        '''
        c_rad = self.r + other.r # Collision Radius
        return (self.distance2(other=other)) < (c_rad*c_rad)
    
    def draw(self, ax, size=10, style='o'):
        '''Draws the point on the plot'''
        ax.scatter(self.x, self.y, s=size)
    

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
                    self.S < other.N or
                    other.S < self.N)
    
    def distance2(self, other:Point):
        '''Distance^2 to a point from the centre'''
        other_x, other_y = other.x, other.y
        return np.square(other_x - self.cx) + np.square(other_y - self.cy)

    def distance_to(self, other: Point) -> float:
        '''Distance to a point from the centre'''
        other_x, other_y = other.x, other.y
        return np.hypot(other_x - self.x, other_y - self.y)

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
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, linewidth=lw)


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
        d = np.square(self.x - point_x) + np.square(point_y - self.y)
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
        dist2 = np.square(other.x - self.x) + np.square(other.y - self.y)
        return (dist2 < np.square(self.r + other.r))
    
    def draw(self, ax, c='k', lw=1):
        '''Draws a circle'''
        ax.add_patch(patches.Circle([self.x, self.y], self.r, fill= False, c=c, linewidth= lw))


class QuadTree:
    '''
    Recursive Quad-tree implementation
    '''
    def __init__(self, bounds: Rect, capacity = 1, depth = 0):
        '''Initialize the quadtee.\n
        bounds is a Rect object showing the bounds\n
        capacity is the number of points each node holds
        depth is the current depth of the node
        '''
        self.bounds = bounds
        self.capacity = capacity
        self.depth = depth

        self.mass = 0
        self.mx = 0
        self.my = 0

        self.divided = False
        self.points = []

        self.verbose = 0
        
    def divide(self):
        '''
        Divide the node into its 4 children
        '''
        cx, cy = self.bounds.cx, self.bounds.cy
        w = self.bounds.w/2
        self.NW = QuadTree(Rect(cx= cx-w/2, cy= cy+w/2, w= w/2), capacity= self.capacity, depth= self.depth + 1)
        self.NE = QuadTree(Rect(cx= cx+w/2, cy= cy+w/2, w= w/2), capacity= self.capacity, depth= self.depth + 1)
        self.SW = QuadTree(Rect(cx= cx-w/2, cy= cy-w/2, w= w/2), capacity= self.capacity, depth= self.depth + 1)
        self.SE = QuadTree(Rect(cx= cx+w/2, cy= cy-w/2, w= w/2), capacity= self.capacity, depth= self.depth + 1)
        self.divided = True
        for p in self.points:
            self.insert(p)    
        self.points = []
        if self.verbose > 1:
            print(f'Node Divided (Depth {self.depth})')

    def insert_to_quadrant(self, point:Point):
        ''' Insert a point into the appropriate quadrant'''
        quadrant = self.bounds.Quadrant(point)
        if self.verbose > 0:
            print(f'Point {point.id} to {quadrant} at depth {self.depth}')
        if   quadrant == 'NW': 
            return self.NW.insert(point)
        elif quadrant == 'NE': 
            return self.NE.insert(point)
        elif quadrant == 'SW': 
            return self.SW.insert(point)
        elif quadrant == 'SE': 
            return self.SE.insert(point)
        else:                  
            return False

    def insert(self, point: Point):
        assert self.bounds.contains(point), 'ERROR POINT OOB??'
        
        self.mx = ((self.mx * self.mass + point.x * point.m) / (self.mass + point.m))
        self.my = ((self.my * self.mass + point.y * point.m) / (self.mass + point.m))
        self.mass += point.m

        if self.divided == True:
            return self.insert_to_quadrant(point)

        if len(self.points) < self.capacity:
            self.points.append(point)
            if self.verbose > 0:
                print(f'Point {point.id} appended at depth {self.depth}')
            return True
        else:
            self.divide()
            if self.verbose > 0:
                print(f'Point {point.id} triggered divide at depth {self.depth}')
            return self.insert_to_quadrant(point)

    def draw(self, ax):
        ''' Draws the tree'''
        self.bounds.draw(ax)
        if self.divided:
            self.NW.draw(ax)
            self.NE.draw(ax)
            self.SW.draw(ax)
            self.SE.draw(ax)



class Test:
    '''Test the Quad-Tree implementation'''
    def __init__(self):
        self.capacity = 1
        self.points = []
        self.tree = None
    
    def create_points(self, npoints: int):
        '''Geneate a random assortment of points'''
        x = np.random.random_sample((npoints, 1)) * 2
        y = np.random.random_sample((npoints, 1)) * 2
        m = np.random.normal(1, 0.2, (npoints, 1))

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        w = np.max([(x_max - x_min), (y_max - y_min)])

        cx = 0.5 * (x_max - x_min)
        cy = 0.5 * (y_max - y_min)

        self.bounds = Rect(cx, cy, w)

        for i in range(npoints):
            point = Point(x = x[i], y = y[i], mass = m[i], ID = i)
            self.points.append(point)

        print([testmethod.bounds.contains(p) for p in testmethod.points])

    def build_tree(self):
        '''Builds the tree'''
        self.tree = QuadTree(bounds= self.bounds,
                             capacity= self.capacity)
        
        self.tree.verbose = 2
        for point in self.points:
            self.tree.insert(point)
            print(f'Point {point.id} inserted')

    def draw(self):
        '''Draws the whole tree'''
        fig, ax = plt.subplots()

        self.tree.draw(ax)

        for p in self.points:
            p.draw(ax)
        plt.savefig('quadtree_test.png')

testmethod = Test()
testmethod.create_points(10)
testmethod.build_tree()
testmethod.draw()

        
