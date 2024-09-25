#from mesh_test import create_mesh as create_mesh
from mesh_test import create_regular_mesh 
from mesh_test import create_regular_mesh, create_mesh_cantilever, create_mesh_corbel, create_mesh_wall_with_openings, create_mesh_wall_without_openings
from node import Node
from element import Element


    
class Mesh:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create(mesh_name):   
        
        if mesh_name == 'cantilever':
            [ex, ey], coords, dofs, edof = create_mesh_cantilever()
        elif mesh_name == 'regular_mesh':
            [ex, ey], coords, dofs, edof = create_regular_mesh()
        elif mesh_name == 'corbel':
            [ex, ey], coords, dofs, edof = create_mesh_corbel()
        elif mesh_name == 'wall_with_openings':
            [ex, ey], coords, dofs, edof = create_mesh_wall_with_openings()
        elif mesh_name == 'wall_without_openings':
            [ex, ey], coords, dofs, edof = create_mesh_wall_without_openings()
        else:
            print('Mesh not defined')
        
            
        nr_nodes = len(dofs)
        nr_elements = len(edof)

        node_list = []
        for i in range(nr_nodes):
            node_dofs = [d-1 for d in dofs[i]]
            node_list.append(Node(coords[i],id=i,dofs=node_dofs))


        element_list = []
        for i in range(nr_elements):
            mesh_element_dofs = [d-1 for d in edof[i]]
            
            node_element_list = []
            for n in node_list:
                if n.dofs[0] == mesh_element_dofs[0]:
                    node_element_list.append(n)
                elif n.dofs[0] == mesh_element_dofs[2]:
                    node_element_list.append(n)
                elif n.dofs[0] == mesh_element_dofs[4]:
                    node_element_list.append(n)
                elif n.dofs[0] == mesh_element_dofs[6]:
                    node_element_list.append(n)

                if len(node_element_list) == 4:
                    break

            sorted_node_element_list = []
            
            for i in range(8):
                if i%2!=0: continue
                for n in node_element_list:
                    if n.dofs[0]==mesh_element_dofs[i]:
                        sorted_node_element_list.append(n)
                        break
                
            element_list.append(Element(sorted_node_element_list, mesh_name)) 

        return node_list, element_list 