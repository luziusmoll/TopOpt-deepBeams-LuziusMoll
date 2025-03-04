import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt
import numpy as np
from mesh import Mesh
from system import System
from utils import construct_polygons_from_neighbors, plot_polygons_and_nodes, plot_boundary_nodes


def create_mesh_cantilever0():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, -1.0], ID=0) # point 0
    g.point([4.0, -1.0], ID=1) # point 1
    g.point([4.0, 1.0], ID=2) # point 2
    g.point([0.0, 1.0], ID=3) # point 3


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3


    hole = False

    if hole:
        g.point([1.0, 0.5], ID=4)
        g.point([2.0, 0.5], ID=5)
        g.point([2.0, -0.5], ID=6)
        g.point([1.0, -0.5], ID=7)
        g.bspline([4,5,6,7,4], ID=4)
        g.surface([0, 1, 2, 3], [[4]])
    else:
        g.surface([0, 1, 2, 3])

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.03

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    
    # shapely geometry 
    # plot boundary nodes
    # plot_boundary_nodes(coords, bdofs)
    # Construct polygons
    polygons = construct_polygons_from_neighbors(coords, bdofs)
    # Plot the results
    plot_polygons_and_nodes(coords, polygons)


    # Define the systems parameters:
    name = 'cantilever0'
    volfrac=0.4
    penalty = 3
    x_min = 1e-3 
    r_min = 0.15  #0.25

    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([4,-1],[0,-1])
    
    s.apply_dirichlet_bc()
    
    s.name = f"{name}_N{len(element_list)}_r{r_min}_p{penalty}"
    s.shapely_geometry = polygons
    
    return s


def create_mesh_cantilever1():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, -1.0], ID=0) # point 0
    g.point([4.0, -1.0], ID=1) # point 1
    g.point([4.0, 1.0], ID=2) # point 2
    g.point([0.0, 1.0], ID=3) # point 3


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3


    hole = False

    if hole:
        g.point([1.0, 0.5], ID=4)
        g.point([2.0, 0.5], ID=5)
        g.point([2.0, -0.5], ID=6)
        g.point([1.0, -0.5], ID=7)
        g.bspline([4,5,6,7,4], ID=4)
        g.surface([0, 1, 2, 3], [[4]])
    else:
        g.surface([0, 1, 2, 3])

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.05

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    
    # shapely geometry 
    # plot boundary nodes
    # plot_boundary_nodes(coords, bdofs)
    # Construct polygons
    polygons = construct_polygons_from_neighbors(coords, bdofs)
    # Plot the results
    plot_polygons_and_nodes(coords, polygons)


    # Define the systems parameters:
    name = 'cantilever1'
    volfrac=0.4
    penalty = 3
    x_min = 1e-3 
    r_min = 0.1  #0.25

    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([4,0],[0,-1])
    
    s.apply_dirichlet_bc()
    
    s.name = f"{name}_N{len(element_list)}_r{r_min}_p{penalty}"
    s.shapely_geometry = polygons
    
    return s


def create_mesh_corbel():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0) # point 0
    g.point([50.0, 0.0], ID=1) # point 1
    g.point([50.0, 100.0], ID=2) # point 2
    g.point([110.0, 100.0], ID=3) # point 3
    g.point([110.0, 170.0], ID=4)
    g.point([50.0, 170.0], ID=5)
    g.point([50.0, 270.0], ID=6)
    g.point([0.0, 270.0], ID=7)


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 4], ID=3) # line 3
    g.spline([4, 5], ID=4)
    g.spline([5, 6], ID=5)
    g.spline([6, 7], ID=6)
    g.spline([7, 0], ID=7)


    g.surface([0, 1, 2, 3, 4, 5, 6, 7])



    #cfv.drawGeometry(g)
    #cfv.showAndWait()

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 1.5

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 


    # shapely geometry 
    # plot boundary nodes
    # plot_boundary_nodes(coords, bdofs)
    # Construct polygons
    polygons = construct_polygons_from_neighbors(coords, bdofs)
    # Plot the results
    plot_polygons_and_nodes(coords, polygons)


    # Define the systems parameters:
    name = 'corbel'
    volfrac=0.4
    penalty = 3
    x_min = 1e-3
    r_min = 3
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)

    # BC
    s.fix_line(np.array([0.0,0.0]), np.array([50.0,0.0]))
    s.fix_line(np.array([0.0,270.0]), np.array([50.0,270.0]))
    s.load_point([95,170],[0,-1])
    
    s.apply_dirichlet_bc()
    
    s.shapely_geometry = polygons
    
    s.name = f"{name}"
    return s


def create_mesh_wall_with_openings():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0) # point 0
    g.point([122.5, 0.0], ID=1) # point 1
    g.point([122.5, 75.0], ID=2) # point 2
    g.point([0.0, 75.0], ID=3) # point 3
    
    # opwning 1
    g.point([12.5, 30.0], ID=4)
    g.point([27.5, 30.0], ID=5)
    g.point([27.5, 45.0], ID=6)
    g.point([12.5, 45.0], ID=7)
    
    # opening 2
    g.point([95, 30.0], ID=8)
    g.point([110, 30.0], ID=9)
    g.point([110, 45.0], ID=10)
    g.point([95, 45.0], ID=11)


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3
    
    g.spline([4, 5], ID=4)
    g.spline([5, 6], ID=5)
    g.spline([6, 7], ID=6)
    g.spline([7, 4], ID=7)
    
    g.spline([8, 9], ID=8)
    g.spline([9, 10], ID=9)
    g.spline([10, 11], ID=10)
    g.spline([11, 8], ID=11)


    g.surface([0, 1, 2, 3], [[4,5,6,7],[8,9,10,11]])



    #cfv.drawGeometry(g)
    #cfv.showAndWait()

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 1

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 

    
    # shapely geometry 
    # plot boundary nodes
    # plot_boundary_nodes(coords, bdofs)
    # Construct polygons
    polygons = construct_polygons_from_neighbors(coords, bdofs)
    # Plot the results
    plot_polygons_and_nodes(coords, polygons)


    # Print the resulting polygons
    for i, polygon in enumerate(polygons):
        print(f"Polygon {i+1} vertices:", list(polygon.exterior.coords))

    
    # Define the systems parameters:
    name = 'wall_with_openings'
    volfrac=0.4
    penalty = 3
    x_min = 1e-3
    r_min = 3
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)

    # BC
    s.fix_node_by_coord([5,0])
    s.fix_node_by_coord([117.5,0], fix = [False,True])
    s.load_point([37.5,75],[0,-1])
    s.load_point([85,75],[0,-1])
    
    s.apply_dirichlet_bc()
    
    
    # additional attrbutes for system
    s.name = f"{name}_N{len(element_list)}_r{r_min}_p{penalty}"
    s.shapely_geometry = polygons 
    
    
    return s



#%% non maintained examples


def create_mesh_cantilever_short():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, -1.0], ID=0) # point 0
    g.point([2.0, -1.0], ID=1) # point 1
    g.point([2.0, 1.0], ID=2) # point 2
    g.point([0.0, 1.0], ID=3) # point 3


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3


    hole = False

    if hole:
        g.point([1.0, 0.5], ID=4)
        g.point([2.0, 0.5], ID=5)
        g.point([2.0, -0.5], ID=6)
        g.point([1.0, -0.5], ID=7)
        g.bspline([4,5,6,7,4], ID=4)
        g.surface([0, 1, 2, 3], [[4]])
    else:
        g.surface([0, 1, 2, 3])

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.05

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # shapely geometry 
    # plot boundary nodes
    # plot_boundary_nodes(coords, bdofs)
    # Construct polygons
    polygons = construct_polygons_from_neighbors(coords, bdofs)
    # Plot the results
    plot_polygons_and_nodes(coords, polygons)


    # Define the systems parameters:
    name = 'cantilever_short'
    volfrac=0.4
    penalty = 3
    x_min = 1e-3 
    r_min = 0.2  #0.25

    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([2,0],[0,-1])
    
    s.apply_dirichlet_bc()
    
    # additional attrbutes for system
    s.name = f"{name}_N{len(element_list)}_r{r_min}_p{penalty}"
    s.shapely_geometry = polygons 
    
    
    
    return s


def create_mesh_cantilever1_hole():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, -1.0], ID=0) # point 0
    g.point([4.0, -1.0], ID=1) # point 1
    g.point([4.0, 1.0], ID=2) # point 2
    g.point([0.0, 1.0], ID=3) # point 3


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3


    hole = True

    if hole:
        g.point([1.0, 0.5], ID=4)
        g.point([2.0, 0.5], ID=5)
        g.point([2.0, -0.5], ID=6)
        g.point([1.0, -0.5], ID=7)
        g.bspline([4,5,6,7,4], ID=4)
        g.surface([0, 1, 2, 3], [[4]])
    else:
        g.surface([0, 1, 2, 3])

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.05

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    
    # shapely geometry 
    # plot boundary nodes
    # plot_boundary_nodes(coords, bdofs)
    # Construct polygons
    polygons = construct_polygons_from_neighbors(coords, bdofs)
    # Plot the results
    plot_polygons_and_nodes(coords, polygons)


    # Define the systems parameters:
    name = 'cantilever1_hole'
    volfrac=0.4
    penalty = 3
    x_min = 1e-3 
    r_min = 0.15  #0.25

    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([4,0],[0,-1])
    
    s.apply_dirichlet_bc()
    
    s.name = f"{name}_N{len(element_list)}_r{r_min}_p{penalty}"
    s.shapely_geometry = polygons
    
    return s
def create_mesh_wall_without_openings():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0)  # point 0
    g.point([100, 0.0], ID=1)  # point 1
    g.point([100, 100.0], ID=2)  # point 2
    g.point([0.0, 100.0], ID=3)  # point 3

    g.spline([0, 1], ID=0)  # line 0
    g.spline([1, 2], ID=1)  # line 1
    g.spline([2, 3], ID=2)  # line 2
    g.spline([3, 0], ID=3)  # line 3

    # Create the surface for meshing
    g.surface([0, 1, 2, 3])

    # Debugging: Draw the geometry to ensure correctness
    # cfv.drawGeometry(g)
    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 3  
    mesh.dofsPerNode = 2
    mesh.elSizeFactor = 6

    # Generate the mesh
    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the TopOpt parameters:
    volfrac=1
    penalty = 3
    x_min = 1e-3
    r_min = 2
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac)

    # BC
    s.fix_node_by_coord([0,0])
    s.fix_node_by_coord([100,0], fix = [False,True])
    # s.load_point([37.5,75],[0,-1])
    # s.load_point([85,75],[0,-1])
    s.load_line(np.array([0,100]), np.array([100,100]),forces=np.array([0.0,-1]))
    
    s.apply_dirichlet_bc()
    
    return s



def create_mesh_tower():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0)  # point 0
    g.point([30, 0.0], ID=1)  # point 1
    g.point([30, 100.0], ID=2)  # point 2
    g.point([0.0, 100.0], ID=3)  # point 3

    g.spline([0, 1], ID=0)  # line 0
    g.spline([1, 2], ID=1)  # line 1
    g.spline([2, 3], ID=2)  # line 2
    g.spline([3, 0], ID=3)  # line 3

    # Create the surface for meshing
    g.surface([0, 1, 2, 3])

    # Debugging: Draw the geometry to ensure correctness
    # cfv.drawGeometry(g)
    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 3  
    mesh.dofsPerNode = 2
    mesh.elSizeFactor = 1

    # Generate the mesh
    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the TopOpt parameters:
    volfrac=0.3
    penalty = 3
    x_min = 1e-3
    r_min = 2
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac)

    # BC
    s.fix_line(np.array([0.0,0.0]), np.array([30.0,0.0]))
    s.load_point([15,100],[1,0])
    
    s.apply_dirichlet_bc()
    
    return s


def create_mesh_bridge():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0)  # point 0
    g.point([100, 0.0], ID=1)  # point 1
    g.point([100, 20.0], ID=2)  # point 2
    g.point([0.0, 20.0], ID=3)  # point 3

    g.spline([0, 1], ID=0)  # line 0
    g.spline([1, 2], ID=1)  # line 1
    g.spline([2, 3], ID=2)  # line 2
    g.spline([3, 0], ID=3)  # line 3

    # Create the surface for meshing
    g.surface([0, 1, 2, 3])

    # Debugging: Draw the geometry to ensure correctness
    # cfv.drawGeometry(g)
    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 3  
    mesh.dofsPerNode = 2
    mesh.elSizeFactor = 1

    # Generate the mesh
    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the TopOpt parameters:
    volfrac=0.4
    penalty = 3
    x_min = 1e-3
    r_min = 2
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac)

    # BC
    s.fix_node_by_coord([0,0])
    s.fix_node_by_coord([100,0])
    s.load_line(np.array([0,0]), np.array([100,0]),forces=np.array([0.0,-0.01]))
    
    s.apply_dirichlet_bc()
    
    return s

def create_mesh_bridge_1():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0)  # point 0
    g.point([100, 0.0], ID=1)  # point 1
    g.point([100, 20.0], ID=2)  # point 2
    g.point([0.0, 20.0], ID=3)  # point 3

    g.spline([0, 1], ID=0)  # line 0
    g.spline([1, 2], ID=1)  # line 1
    g.spline([2, 3], ID=2)  # line 2
    g.spline([3, 0], ID=3)  # line 3

    # Create the surface for meshing
    g.surface([0, 1, 2, 3])

    # Debugging: Draw the geometry to ensure correctness
    # cfv.drawGeometry(g)
    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 3  
    mesh.dofsPerNode = 2
    mesh.elSizeFactor = 1

    # Generate the mesh
    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the TopOpt parameters:
    volfrac=0.3
    penalty = 3
    x_min = 1e-3
    r_min = 2
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac)

    # BC
    s.fix_node_by_coord([0,0])
    s.fix_node_by_coord([100,0])
    #s.load_line(np.array([0,0]), np.array([100,0]),forces=np.array([0.0,-0.01]))
    s.load_point([50,20],[0,-1])
    
    s.apply_dirichlet_bc()
    
    return s

def create_mesh_bridge_2():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, 0.0], ID=0)  # point 0
    g.point([200, 0.0], ID=1)  # point 1
    g.point([200, 40.0], ID=2)  # point 2
    g.point([0.0, 40.0], ID=3)  # point 3

    g.spline([0, 1], ID=0)  # line 0
    g.spline([1, 2], ID=1)  # line 1
    g.spline([2, 3], ID=2)  # line 2
    g.spline([3, 0], ID=3)  # line 3

    # Create the surface for meshing
    g.surface([0, 1, 2, 3])

    # Debugging: Draw the geometry to ensure correctness
    # cfv.drawGeometry(g)
    # cfv.showAndWait()

    mesh = cfm.GmshMesh(g)
    mesh.elType = 3  
    mesh.dofsPerNode = 2
    mesh.elSizeFactor = 1

    # Generate the mesh
    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the TopOpt parameters:
    volfrac=0.3
    penalty = 3
    x_min = 1e-3
    r_min = 2
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min=r_min, volfrac=volfrac)

    # BC
    s.fix_node_by_coord([0,0])
    s.fix_node_by_coord([200,0],[False, True])
    # s.load_line(np.array([0,0]), np.array([100,0]),forces=np.array([0.0,-0.01]))
    s.load_point([100,40],[0,-1])
    
    s.apply_dirichlet_bc()
    
    return s
#%%


def create_mesh_2(hole_1=True, hole_2=True):
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    # membrane
    g.point([0.0, 0.0], ID=0)
    g.point([6.0, 0.0], ID=1) 
    g.point([6.0, 3.0], ID=2) 
    g.point([0.0, 3.0], ID=3) 


    g.spline([0, 1], ID=0)
    g.spline([1, 2], ID=1) 
    g.spline([2, 3], ID=2)
    g.spline([3, 0], ID=3)

    # hole 1
    if hole_1 == True:
        g.point([2.0, 1.0], ID=4) 
        g.point([4.0, 1.0], ID=5) 
        g.point([4.0, 2.0], ID=6) 
        g.point([2.0, 2.0], ID=7) 
        g.point([5.0, 1.5], ID=8) 
    
        g.spline([4, 5], ID=4) 
        g.spline([5, 8], ID=5) 
        g.spline([8, 6], ID=6) 
        g.spline([6, 7], ID=7)
    
        g.point([2, 1.5], ID=9)
        g.circle([7, 9, 4], ID=9)


    # hole 2
    if hole_2 == True:
        g.point([5.5, 2.0], ID=10)
        g.point([5.8, 2.0], ID=11)
        g.point([5.2, 2.0], ID=12)
    
        g.circle([12, 10, 11], ID=10)
        g.circle([11, 10, 12], ID=11)
        
        
    if hole_1 == False and hole_2 == False:
        g.surface([0, 1, 2, 3])
    if hole_1 == True and hole_2 == False:
        g.surface([0, 1, 2, 3], [[4,5,6,7,9]])
    if hole_1 == False and hole_2 == True:
        g.surface([0, 1, 2, 3], [[10,11]])
    if hole_1 == True and hole_2  == True:
        g.surface([0, 1, 2, 3], [[4,5,6,7,9],[10,11]])



    #cfv.drawGeometry(g)
    #cfv.showAndWait()

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.1 #0.2
    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the systems parameters:
    volfrac=0.4
    penalty = 3
    x_min = 1e-3
    r_min = 0.25 
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min,volfrac)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([4,-1],[0,-1])
    
    s.apply_dirichlet_bc()
    
    return s





def create_mesh_3():
    # Set up the geometry using calfem:
    g = cfg.Geometry()

    g.point([0.0, -1.0], ID=0) # point 0
    g.point([6.0, -1.0], ID=1) # point 1
    g.point([6.0, 1.0], ID=2) # point 2
    g.point([0.0, 1.0], ID=3) # point 3


    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3


    hole = True

    if hole:
        g.point([3.0, 0.7], ID=4)
        g.point([4.0, 0.7], ID=5)
        g.point([4.0, -0.7], ID=6)
        g.point([3.0, -0.7], ID=7)
        g.bspline([4,5], ID=4)
        g.bspline([5,6], ID=5)
        g.bspline([6,7], ID=6)
        g.bspline([7,4], ID=7)
        
        g.surface([0, 1, 2, 3], [[4,5,6,7]])
    else:
        g.surface([0, 1, 2, 3])



    #cfv.drawGeometry(g)
    #cfv.showAndWait()

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.03

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    node_list, element_list = Mesh.create(coords, dofs, edof)
    print('number of elements:', len(element_list)) 
    
    # Define the systems parameters:
    volfrac=0.4
    penalty = 3
    x_min = 1e-3
    r_min = 0.25 
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min,volfrac)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([4,-1],[0,-1])
    
    s.apply_dirichlet_bc()
    
    return s


def create_regular_mesh(nelx=80, nely=40):
    # Initialize the element dof matrix
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    
    # Initialize the coordinates array
    coords = np.zeros(((nely + 1) * (nelx + 1), 2))
    
    # Fill the coordinates array
    for i in range(nelx + 1):
        for j in range(nely + 1):
            node = i * (nely + 1) + j
            coords[node, :] = [i, j]

    
    # Initialize the degrees of freedom array
    dofs = np.arange(2 * (nelx + 1) * (nely + 1)).reshape((nelx + 1) * (nely + 1), 2)
    
    
    # Fill the element dof matrix
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([
                2 * n1 + 2, 2 * n1 + 3,
                2 * n2 + 2, 2 * n2 + 3,
                2 * n2, 2 * n2 + 1,
                2 * n1, 2 * n1 + 1
            ])
    
    # Initialize the element coordinates arrays
    ex = np.zeros((nelx * nely, 4))
    ey = np.zeros((nelx * nely, 4))
    
    # Fill the element coordinates arrays
    for el in range(nelx * nely):
        ex[el, :] = coords[edofMat[el, [0, 2, 4, 6]] // 2, 0]
        ey[el, :] = coords[edofMat[el, [0, 2, 4, 6]] // 2, 1]

    # ex is a 4 x n_ele matrix with x coords for every element, but they are not used further
    # coords is a n_nodes x 2 matrix with all coordinates
    # coords is a n_nodes x 2 matrix with all dofs: [[1,2],[3,4],...]
    # edof is a n_ele x 8 matrix containing the dofs mapping to the elements
    node_list, element_list = Mesh.create(coords, dofs, edofMat, regular_mesh=True)

    # Define the systems parameters:
    volfrac=0.4
    penalty = 3
    x_min = 1e-3
    r_min = 0.25 
    
    for e in element_list:
        e.E = 30000
        e.nu = 0.15

    # volume fraction for all elements is set to volfrac
    x = np.ones(len(element_list),dtype=float)*volfrac

    # Set up FE problem
    s = System(node_list, element_list, x, r_min,volfrac)

    # BC
    s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    s.load_point([4,-1],[0,-1])
    
    s.apply_dirichlet_bc()
    
    return s
