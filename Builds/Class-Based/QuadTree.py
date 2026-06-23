from __future__ import annotations
from Point import Point
from Rect import Rect
from Circ import Circ


class QuadTree:
    '''
    Recursive Quad-tree implementation
    '''
    def __init__(self, bounds: Rect, capacity = 1, depth = 0, verbose= 1):
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
        Divide the node into its 4 children and re-distribute the points
        '''
        #Create children nodes
        cx, cy = self.bounds.cx, self.bounds.cy # centre of node
        w = self.bounds.w  # Width of the node
        self.NW = QuadTree(Rect(cx= cx-w/4, cy= cy+w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose) # Top Left Node
        self.NE = QuadTree(Rect(cx= cx+w/4, cy= cy+w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose) # Top Right Node
        self.SW = QuadTree(Rect(cx= cx-w/4, cy= cy-w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose) # Bottom Left Node
        self.SE = QuadTree(Rect(cx= cx+w/4, cy= cy-w/4, w= w/2), capacity= self.capacity, depth= self.depth + 1, verbose= self.verbose) # Bottom Right Node
        self.divided = True

        # Insert points into new nodes
        for p in self.points:
            self.insert_to_quadrant(p)    
        self.points = []

        if self.verbose > 1:
            print(f'Node Divided (Depth {self.depth})')

    def insert_to_quadrant(self, point:Point):
        ''' Insert a point into the appropriate quadrant'''
        # Find which quadrant to insert to
        quadrant = self.bounds.Quadrant(point)
        # Insert to that quadrant
        if self.verbose > 1:
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
        ''' Insterts a point onto the tree'''
        # Error handling, should NEVER happen if everything is set up right.
        assert self.bounds.contains(point), 'ERROR POINT OOB??'
        
        # Re-compute centre of mass
        self.mx = ((self.mx * self.mass + point.x * point.m) / (self.mass + point.m))
        self.my = ((self.my * self.mass + point.y * point.m) / (self.mass + point.m))
        self.mass += point.m

        # Insert to quadrant if divided already
        if self.divided == True:
            return self.insert_to_quadrant(point)

        # Insert point if not divided
        if len(self.points) < self.capacity: # Check if too many points in node
            self.points.append(point)
            if self.verbose > 1:
                print(f'Point {point.id} appended at depth {self.depth}')
            return True
        else: # If too many points, trigger divide and insert points
            self.divide()
            if self.verbose > 1:
                print(f'Point {point.id} triggered divide at depth {self.depth}')
            return self.insert_to_quadrant(point)

    def force_on(self, point: Point, theta = 0.5, G = 1., eps= 1e-3) -> tuple[float, float]:
        ''' Computes the force on the point'''

        force_x, force_y = 0, 0

        if self.mass ==0: # skip node if mass is 0 
            return force_x, force_y
        
        if not self.divided: # If the point is not divided

            for p in self.points: #iterate though the points in the node

                if p is point: # No force computation with itself
                    continue
                
                # Calculate force
                fx, fy = point.force_between_xy(x= p.x, y= p.y, m= p.m, G= G, eps= eps) 
                force_x += fx
                force_y += fy
            return force_x, force_y
        
        else:
            dx = self.mx - point.x
            dy = self.my - point.y
            d2 = (dx*dx + dy*dy + eps*eps)
            s2 = self.bounds.w * self.bounds.w
             # s/d < theta
            if s2 < theta * theta * d2: # Barnes hut threshold
                return point.force_between_xy(x= self.mx, y= self.my, m= self.mass, G= G, eps= eps)
            
            else: # if barnes hut threshold not met:
                fx, fy = self.NW.force_on(point= point, theta= theta, G= G, eps= eps)
                force_x += fx
                force_y += fy
                fx, fy = self.NE.force_on(point= point, theta= theta, G= G, eps= eps)
                force_x += fx
                force_y += fy
                fx, fy = self.SW.force_on(point= point, theta= theta, G= G, eps= eps)
                force_x += fx
                force_y += fy
                fx, fy = self.SE.force_on(point= point, theta= theta, G= G, eps= eps)
                force_x += fx
                force_y += fy       
        return force_x, force_y

    def query(self, area: Circ, found: list|None= None ) -> list:
        ''' Recursive search for points inside circle'''

        if found is None: # Initialise empty array
            found = []

        if self.mass == 0: # Empty node
            return found

        if not area.rect_intersect(self.bounds): # Circle area does not intersect bounds of node
            return found
        
        if self.divided: # Recursion
            self.NW.query(area= area, found= found)
            self.NE.query(area= area, found= found)
            self.SW.query(area= area, found= found)
            self.SE.query(area= area, found= found)
            
        else: # If not divided, go through every point in node and append if in range
            for p in self.points:
                if area.contains(p):
                    found.append(p)

        return found    

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

