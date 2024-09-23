#from mesh_test import create_mesh as create_mesh
from mesh_test import create_regular_mesh 
from mesh_test import create_mesh_wall_without_openings as create_mesh #create_mesh, create_mesh_wall_with_openings, create_mesh_corbel
from node import Node
from element import Element


    
class Mesh:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create(regular_mesh):   
        
        if regular_mesh == False:
            [ex, ey], coords, dofs, edof = create_mesh()
        if regular_mesh == True:
            [ex, ey], coords, dofs, edof = create_regular_mesh()
            
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
                
            element_list.append(Element(sorted_node_element_list, regular_mesh)) 

        return node_list, element_list 