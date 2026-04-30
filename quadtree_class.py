from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("QtAgg")
import matplotlib.patches as patches
from utils import generate_initial_state


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
       self.r = ((mass*density)/(np.pi))**(1/2)

    def __str__(self):
        '''Formated string putput'''
        return f'P{self.id}: ({self.x:.4f}, {self.y:.4f}); m={self.m:.4f}'
    
    def velocity(self, vx, vy):
        ''' Defines Velocity'''
        self.vx = vx
        self.vy = vy

    def acceleration(self, fx, fy):
        ''' Defines Acceleration'''
        self.fx = fx
        self.fy = fy

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

    def update_position_euler(self, fx, fy, dt):
        ''' Uses Euler integration to update the position of the point'''
        self.fx, self.fy = fx, fy

        self.x += self.x + self.vx * dt
        self.y += self.y + self.vy * dt

        self.vx += self.vx + fx * dt
        self.vy += self.vy + fy * dt


    
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
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c, linewidth=lw)


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
    def __init__(self, bounds: Rect, capacity = 1, depth = 0, verbose = 0):
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

        self.verbose = verbose
        
    def divide(self):
        '''
        Divide the node into its 4 children
        '''
        cx, cy = self.bounds.cx, self.bounds.cy
        w = self.bounds.w
        self.NW = QuadTree(Rect(cx= cx-w/4, cy= cy+w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose)
        self.NE = QuadTree(Rect(cx= cx+w/4, cy= cy+w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose)
        self.SW = QuadTree(Rect(cx= cx-w/4, cy= cy-w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose)
        self.SE = QuadTree(Rect(cx= cx+w/4, cy= cy-w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose)
        self.divided = True
        for p in self.points:
            self.insert_to_quadrant(p)    
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

    def print_tree(self):
        '''Return a string representation of the tree'''
        prefix = (self.depth * 2 + 1) *  " " 
        if self.divided:
            print(f"{prefix}[Node d={self.depth}] "
                f"mass={self.mass:.3f} "
                f"COM=({self.mx:.3f},{self.my:.3f})")

            self.NW.print_tree()
            self.NE.print_tree()
            self.SW.print_tree()
            self.SE.print_tree()

        else:
            pts = ", ".join(f"{p.id}" for p in self.points)
            print(f"{prefix}[Leaf d={self.depth}] "
                f"n={len(self.points)} "
                f"pts ID=[{pts}] "
                f"mass={self.mass:.3f} "
                f"COM=({self.mx:.3f},{self.my:.3f})")
            
    def draw(self, ax, c='k', lw=1):
        ''' Draws the tree'''
        self.bounds.draw(ax, c=c, lw=lw)
        if self.divided:
            self.NW.draw(ax, c=c, lw=lw)
            self.NE.draw(ax, c=c, lw=lw)
            self.SW.draw(ax, c=c, lw=lw)
            self.SE.draw(ax, c=c, lw=lw)


class Test:
    '''Test the Quad-Tree implementation'''
    def __init__(self):
        self.capacity = 4
        self.points = []
        self.tree = None
    
    def create_points(self, npoints: int):
        '''Geneate a random assortment of points'''

        x = np.round(np.random.normal(1, 0.2, (1, npoints)).T * 2 -1, 6)
        y = np.round(np.random.normal(1, 0.2, (1, npoints)).T * 2 -1, 6)

        #x = np.round(np.random.random_sample((1, npoints)).T * 2 -1, 6)
        #y = np.round(np.random.random_sample((1, npoints)).T * 2 -1, 6)
        
        m = np.random.normal(1, 0.2, (npoints, 1))

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        w = max(x_max - x_min, y_max - y_min)

        cx = 0.5 * (x_max + x_min)
        cy = 0.5 * (y_max + y_min)

        self.bounds = Rect(cx, cy, w*1.01)

        for i in range(npoints):
 
            point = Point(x = x[i].item(), y = y[i].item(), mass = m[i].item(), ID = i)
            self.points.append(point)
            #print(f'Point {i}: ({x[i].item()}, {y[i].item()})')

        #print([testmethod.bounds.contains(p) for p in testmethod.points])

    def create_points_orbiting(self, npoints: int, main_mass= 3e4):

        m = np.random.rand(1, npoints+1).T
        m[0,0] = main_mass
        mpos, mvel = generate_initial_state(npoints, mu= main_mass * 0.01)
        x,y = np.vstack([np.asarray([0,0]).reshape([1,2]),mpos.T]).T
        vx, vy = np.vstack([np.asarray([0,0]).reshape([1,2]),mvel.T]).T

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        w = max(x_max - x_min, y_max - y_min)

        cx = 0.5 * (x_max + x_min)
        cy = 0.5 * (y_max + y_min)

        self.bounds = Rect(cx, cy, w*1.01)

        for i in range(npoints+1):
 
            point = Point(x = x[i].item(), y = y[i].item(), mass = m[i].item(), ID = i)
            point.velocity(vx = vx[i], vy= vy[i])
            self.points.append(point)

    def build_tree(self, debug):
        '''Builds the tree'''
        self.tree = QuadTree(bounds= self.bounds,
                             capacity= self.capacity)
        
        self.tree.verbose = debug
        #print(20 * '=')
        for point in self.points:
            #print(f'Insterting Point {point.id}...')
            self.tree.insert(point)
            #print(f'Point {point.id} inserted')
            #print(20 * '=')

    def draw(self):
        '''Draws the whole tree'''
        fig, ax = plt.subplots(figsize=(9,9))

        self.tree.draw(ax, c='k', lw = 0.5)

        for p in self.points:
            p.draw(ax, size=1)

        ax.set_aspect('equal')
        ax.set_xlim(self.bounds.W, self.bounds.E)
        ax.set_ylim(self.bounds.S, self.bounds.N)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig('quadtree_test.svg', bbox_inches= 'tight')

testmethod = Test()
#testmethod.create_points(1000)
testmethod.create_points_orbiting(100)
testmethod.build_tree(debug= 0)
testmethod.draw()
testmethod.tree.print_tree()

print('EOF')

        
