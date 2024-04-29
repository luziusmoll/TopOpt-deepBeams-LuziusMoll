import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt

def create_mesh():

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
        g.point([2.0, 2.5], ID=4)
        g.point([2.5, 3.0], ID=5)
        g.point([3.0, 2.5], ID=6)
        g.point([2.5, 2.0], ID=7)
        g.bspline([4,5,6,7,4], ID=4)
        g.surface([0, 1, 2, 3], [[4]])
    else:
        g.surface([0, 1, 2, 3])



    #cfv.drawGeometry(g)
    #cfv.showAndWait()

    mesh = cfm.GmshMesh(g)

    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.08

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    ex, ey = cfc.coordxtr(edof, coords, dofs)

    if 2<0:
        cfv.figure()
        cfv.drawMesh(
            coords=coords,
            edof=edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True)
        cfv.showAndWait()
    
    return [ex, ey], coords, dofs, edof


def create_mesh_2(hole_1=True, hole_2=True):

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

    ex, ey = cfc.coordxtr(edof, coords, dofs)

    if 2<0:
        cfv.figure()
        cfv.drawMesh(
            coords=coords,
            edof=edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True)
        cfv.showAndWait()
    
    return [ex, ey], coords, dofs, edof

if 2<0:
    [ex, ey], coords, dofs, edof = create_mesh_2()

    print(coords)
    print(dofs)
    print(edof)

    for x,y in zip(ex,ey):
        coord = [[x[0],y[0]],[x[1],y[1]],[x[2],y[2]],[x[3],y[3]],[x[0],y[0]]]
        xs, ys = zip(*coord) #create lists of x and y values
        plt.plot(xs,ys)

    plt.grid()
    plt.show()

