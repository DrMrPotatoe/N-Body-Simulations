from QuadTree_Interface import QuadTree_Interface


testmethod = QuadTree_Interface(10)
testmethod.capacity = 1
testmethod.T1 = 10
testmethod.collide = True
testmethod.density = 1e4
testmethod.verbose = 0
# testmethod.create_points()
testmethod.create_points_orbiting()
# testmethod.build_tree()
# testmethod.draw()
# testmethod.compute_force(verbose= True)
# testmethod.tree.print_tree()
# testmethod.gif_simulate()
testmethod.video_simulate(fps= 60)

print('EOF')