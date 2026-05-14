from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("QtAgg")
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
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
        return np.square(self.x - other_x) + np.square(other_y - self.y)

    def distance_to(self, other: Point, eps: float= 1e-3) -> float:
        '''Distance to another point'''
        other_x, other_y = other.x, other.y
        return np.sqrt(np.square(other_x - self.x) + np.square(other_y - self.y) + eps*eps)

    def distance2_xy(self, x: float, y: float, eps: float= 1e-3) -> float:
        ''' Distance^2 to xy coods'''
        return np.square(self.x - x) + np.square(self.y - y) + np.square(eps)
    
    def distance_to_xy(self, x: float, y: float, eps: float= 1e-3) -> float:
        ''' Distance to xy coords'''
        d2 = np.square(self.x - x) + np.square(self.y - y) + np.square(eps)
        return np.sqrt(d2)

    def collides(self, other:Point) -> bool:
        ''' Whether this point and another point intersect (for collisions)'''
        c_rad = self.r + other.r # Collision Radius
        return (self.distance2(other=other)) < (c_rad*c_rad)
    
    def force_between(self, point:Point, G= 1) -> np.ndarray[float, float]:
        ''' Calculates the force between it and another point'''
        d = self.distance_to(point)
        d2 = d * d
        df = - self.mu(G) * point.m / d2
        ux = (self.x - point.x)/d
        uy = (self.y - point.y)/d
        fx = df * ux
        fy = df * uy
        return np.array([fx, fy])

    def force_between_xy(self, x: float, y: float, m: float, G: float = 1., eps: float = 1e-3) -> np.ndarray[float, float]:
        ''' Calculates the force between it and another point'''
        d2 = self.distance2_xy(x, y, eps)
        d = np.sqrt(d2)
        df = self.mu(G) * m / d2
        fx = df * (x - self.x)/d
        fy = df * (y - self.y)/d
        return np.array([fx, fy])

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

    def kick(self, dt: float):
        ''' Give a kick to the point'''
        self.vx += 0.5 * self.fx / self.m * dt
        self.vy += 0.5 * self.fy / self.m * dt

    def drift(self, dt: float):
        ''' Update Position using the velocity'''
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, ax, size=10, style='o'):
        ''' Draws the point on the plot'''
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
                    self.N < other.S or
                    other.N < self.S)
    
    def distance2(self, other:Point):
        '''Distance^2 to a point from the centre'''
        other_x, other_y = other.x, other.y
        return np.square(other_x - self.cx) + np.square(other_y - self.cy)

    def distance_to(self, other: Point) -> float:
        '''Distance to a point from the centre'''
        other_x, other_y = other.x, other.y
        return np.hypot(other_x - self.cx, other_y - self.cy)

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
        assert self.bounds.contains(point), 'ERROR POINT OOB??'
        
        self.mx = ((self.mx * self.mass + point.x * point.m) / (self.mass + point.m))
        self.my = ((self.my * self.mass + point.y * point.m) / (self.mass + point.m))
        self.mass += point.m

        if self.divided == True:
            return self.insert_to_quadrant(point)

        if len(self.points) < self.capacity:
            self.points.append(point)
            if self.verbose > 1:
                print(f'Point {point.id} appended at depth {self.depth}')
            return True
        else:
            self.divide()
            if self.verbose > 1:
                print(f'Point {point.id} triggered divide at depth {self.depth}')
            return self.insert_to_quadrant(point)

    def force_on(self, point: Point, theta = 0.5, G = 1., eps= 1e-3) -> np.ndarray[float, float]:
        ''' Computes the force on the point'''

        force = np.zeros(2)

        if self.mass ==0:
            return force
        
        if not self.divided:
            for p in self.points:
                if p is point:
                    continue

                force += point.force_between_xy(x= p.x, y= p.y, m= p.m, G= G, eps= eps)
            
            return force
        
        else:
            dx = self.mx - point.x
            dy = self.my - point.y
            d = np.sqrt(dx*dx + dy*dy + eps*eps)
            s = self.bounds.w

            if (s/d) < theta:
                return point.force_between_xy(x= self.mx, y= self.my, m= self.mass, G= G, eps= eps)
            
            else:
                force += self.NW.force_on(point= point, theta= theta, G= G, eps= eps)
                force += self.NE.force_on(point= point, theta= theta, G= G, eps= eps)
                force += self.SW.force_on(point= point, theta= theta, G= G, eps= eps)
                force += self.SE.force_on(point= point, theta= theta, G= G, eps= eps)
        return force

    def query(self, area: Circ, found: list|None= None ) -> list:
        ''' Recursive search for points inside circle'''
        if found is None:
            found = []

        if not area.rect_intersect(self.bounds):
            return found
        
        if self.divided:
            self.NW.query(area= area, found= found)
            self.NE.query(area= area, found= found)
            self.SW.query(area= area, found= found)
            self.SE.query(area= area, found= found)
            
        else:
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


class Quad_Tree_Interface_V1:
    '''Test the Quad-Tree implementation'''
    def __init__(self, npoints: int):
        self.capacity = 1
        self.points = []
        self.tree = None
        self.verbose = 1
        self.frame = 0
        self.gif = True
        self.collide = True

        self.npoints = npoints
        self.theta = 0.5
        self.G = 1e-6
        self.eps = 1e-3
        self.main_mass = 3e4
        self.escape_radius = 4 * np.log10(npoints)
        self.density = 1e6

        self.dt = 0.1
        self.T1 = 10
    
    def create_points(self):
        '''Geneate a random assortment of points'''

        x = np.round(np.random.normal(1, 0.2, (1, self.npoints)).T * 2 -1, 6)
        y = np.round(np.random.normal(1, 0.2, (1, self.npoints)).T * 2 -1, 6)

        #x = np.round(np.random.random_sample((1, npoints)).T * 2 -1, 6)
        #y = np.round(np.random.random_sample((1, npoints)).T * 2 -1, 6)
        
        m = np.random.normal(1, 0.2, (self.npoints, 1))

        x_max, x_min = x.max(), x.min()
        y_max, y_min = y.max(), y.min()
        w = max(x_max - x_min, y_max - y_min)

        cx = 0.5 * (x_max + x_min)
        cy = 0.5 * (y_max + y_min)

        self.bounds = Rect(cx, cy, w*1.01)

        for i in range(self.npoints):
 
            point = Point(x = x[i].item(), y = y[i].item(), mass = m[i].item(), ID = i)
            point.radius(self.density)
            self.points.append(point)
            #print(f'Point {i}: ({x[i].item()}, {y[i].item()})')

        #print([testmethod.bounds.contains(p) for p in testmethod.points])

    def create_points_orbiting(self):

        from utils import generate_initial_state

        m = np.random.rand(1, self.npoints+1).T
        m[0,0] = self.main_mass
        mpos, mvel = generate_initial_state(self.npoints, mu= self.main_mass * self.G)
        x,y = np.vstack([np.asarray([0,0]).reshape([1,2]),mpos.T]).T
        vx, vy = np.vstack([np.asarray([0,0]).reshape([1,2]),mvel.T]).T

        for i in range(self.npoints+1):

            point = Point(x = x[i].item(), y = y[i].item(), mass = m[i].item(), ID = i)
            point.vx, point.vy = vx[i].item(), vy[i].item()
            point.radius(self.density)
            if i == self.main_mass:
                point.r = point.r * 0.001

            self.points.append(point)

    def define_bounds(self):

        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        xmax, xmin = np.max(x), np.min(x)
        ymax, ymin = np.max(y), np.min(y)
        w = max(xmax - xmin, ymax - ymin)

        cx = 0.5 * (xmax + xmin)
        cy = 0.5 * (ymax + ymin)

        self.bounds = Rect(cx, cy, w*1.01)

    def remove_escaped_points(self):
        ''' Removes points that have excaped'''
        escaped2 = self.escape_radius * self.escape_radius

        points_before = len(self.points)
        self.points = [
            p for p in self.points if 
            p.distance2(self.points[0]) < escaped2
        ]
        if self.verbose > 1:
            print(f'Removed points: {points_before - len(self.points)}')

    def build_tree(self):
        '''Builds the tree'''

        self.define_bounds()

        self.tree = QuadTree(bounds= self.bounds,
                             capacity= self.capacity,
                             verbose= self.verbose)
        
        if self.verbose > 1: print(20 * '=')
        for point in self.points:
            self.tree.insert(point)
            if self.verbose > 1:
                print(f'Point {point.id} inserted')
                print(20 * '=')

    def compute_force(self):
        ''' Computes the force of every point'''

        for p in self.points:
            fx, fy = self.tree.force_on(point= p, theta= self.theta, G= self.G, eps= self.eps)
            p.set_force(fx= fx, fy= fy)
            if self.verbose > 2:
                print(p) 

        if self.verbose > 1:
            fx_tot = sum(p.fx for p in self.points)
            fy_tot = sum(p.fy for p in self.points)
            print(f"Net force = ({fx_tot:.6e}, {fy_tot:.6e})")

    def find_collisions(self) -> list:
        ''' Find all collisions in the set of points'''

        collisions = []
        checked = set()

        max_r = np.max([p.r for p in self.points])

        for p in self.points:
            search_area = Circ(p.x, p.y, p.r + max_r)
            
            nearby_points = self.tree.query(search_area)

            for other in nearby_points:
                if other is p:
                    continue
                
                # avoid duplicate pairs 
                pair = tuple(sorted((p.id, other.id)))
                if pair in checked:
                    continue

                checked.add(pair)
                if p.collides(other):
                    collisions.append((p, other))
        
        return collisions
    
    def merge_points(self, p1: Point, p2: Point):
        ''' Merges 2 points into the first'''

        m1, m2 = p1.m, p2.m

        m = m1 + m2

        p1.x = ((p1.x * m1) + (p2.x * m2)) / m
        p1.y = ((p1.y * m1) + (p2.y * m2)) / m

        p1.vx = ((p1.vx * m1) + (p2.vx * m2)) / m
        p1.vy = ((p1.vy * m1) + (p2.vy * m2)) / m

        p1.id = p1.id
        p1.m = m
        p1.radius(self.density)
        if m > self.main_mass -1:
            p1.r = p1.r * 0.001
            

    def collision_handler(self):
        ''' Handles collision math and removes collided points'''

        collisions = self.find_collisions()

        collision_count = 0
        if not collisions:
            return
        
        removed = set()

        for p1, p2 in collisions:
            if p1 in removed or p2 in removed:
                continue

            self.merge_points(p1, p2)

            removed.add(p2)
            collision_count += 1
        self.points = [p for p in self.points if p not in removed]
        print(f'Collided {collision_count:>3.0f} points in step {self.frame}')

    def step(self):
        '''Computes a step'''

        # Update Velocity
        for p in self.points:
            p.kick(self.dt)
        
        # Update Pos    
        for p in self.points:
            p.drift(self.dt)

        # Remove escaped particles
        self.remove_escaped_points()

        # Re-compute tree
        self.build_tree()

        if self.collide:
            # do collisions
            self.collision_handler()

            # Re-build the tree after collisions
            self.build_tree()

        # Re-compute forces
        self.compute_force()

        # Update Velocity
        for p in self.points:
            p.kick(self.dt)
        
        self.frame += 1

    def simulate(self):
        ''' Simulates a bunch of steps'''

        steps = int(self.T1 / self.dt)
        for t in range(0, steps):
            
            self.build_tree()

            self.compute_force()

            self.step()

            if self.verbose > 0:
                print(f'T= {t} Done')

    def init_plot(self):
        ''' Initialises the gif plot area'''
        self.fig, self.ax = plt.subplots(figsize=(9,9))

        self.ax.set_aspect('equal')
        self.ax.set_facecolor('k')

        self.scatter = self.ax.scatter([], [], s=[], c= 'white')

        self.ax.set_axis_off()
        self.fig.subplots_adjust(left= 0, bottom= 0, right= 1, top= 1)

        x0 = self.points[0].x
        y0 = self.points[0].y
        w = self.bounds.w
        #self.ax.set_xlim(self.bounds.W * 1.5, self.bounds.E * 1.5)
        #self.ax.set_ylim(self.bounds.S * 1.5, self.bounds.N * 1.5)
        self.ax.set_xlim((x0 - w), (x0 + w))
        self.ax.set_ylim((y0 - w), (y0 + w))
        #print('test')

    def animate(self, frame):
        ''' Animates the gif'''
        self.step()
        print(f'Step {self.frame:.0f} Animated')

        x = [p.x for p in self.points]
        y = [p.y for p in self.points]

        m = [float(0)] + [p.m for p in self.points[1:]]
        sz = 2 + 5 * np.log10([mi+1 for mi in m])

        self.scatter.set_offsets(np.c_[x, y])
        self.scatter.set_sizes(sz)

        #self.ax.set_xlim(min(x), max(x))
        #self.ax.set_ylim(min(y), max(y))

        return (self.scatter,)
    
    def simulate_gif(self):
        ''' Simulates the system for the GIF'''
        self.build_tree()
        self.compute_force()

        self.init_plot()

        steps = int(self.T1 / self.dt)

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=steps,
            interval=30,
            blit=True
        )

        anim.save(
            "QuadTree.gif",
            writer=PillowWriter(fps=30),
            savefig_kwargs= {
                "facecolor": "black",
                "pad_inches": 0
            }
        )

    def draw(self, save= True):
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
        if save == True:
            plt.savefig('quadtree_test.svg', bbox_inches= 'tight')
        else:
            return fig

testmethod = Quad_Tree_Interface_V1(1000)
testmethod.capacity = 2
testmethod.T1 = 10
testmethod.collide = True
# testmethod.create_points()
testmethod.create_points_orbiting()
testmethod.build_tree()
testmethod.draw()
# testmethod.compute_force(verbose= True)
# testmethod.tree.print_tree()
testmethod.simulate_gif()

print('EOF')

        
