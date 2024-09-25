
def choose_system():
    # Display available options to the user
    print('Choose a mesh type:')
    print('1. cantilever')
    print('2. regular_mesh')
    print('3. corbel')
    print('4. wall_with_openings')
    print('5. wall_without_openings')

    # Get user input
    try:
        choice = int(input('Enter the number corresponding to the mesh type: '))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    # Initialize mesh_name based on user input
    if choice == 1:
        mesh_name = 'cantilever'
    elif choice == 2:
        mesh_name = 'regular_mesh'
    elif choice == 3:
        mesh_name = 'corbel'
    elif choice == 4:
        mesh_name = 'wall_with_openings'
    elif choice == 5:
        mesh_name = 'wall_without_openings'
    else:
        print("Invalid choice. Please select a valid option.")
        return


    if mesh_name == 'cantilever':
        print("Running topology optimization for cantilever...")
    elif mesh_name == 'regular_mesh':
        print("Running topology optimization for regular mesh...")
    elif mesh_name == 'corbel':
        print("Running topology optimization for corbel...")
    elif mesh_name == 'wall_with_openings':
        print("Running topology optimization for wall with openings...")
    elif mesh_name == 'wall_without_openings':
        print("Running topology optimization for wall without openings...")
        
    return mesh_name
