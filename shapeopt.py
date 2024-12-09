import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import os

def domain_penalty1(system, design_boundary, holes, node_weight=1.0, beam_weight=1.0, penalty_scale=1.0, ax=None, plot=None):
    """
    Calculate total domain penalty for a system, considering node and beam penalties, with visualization.

    Parameters:
    - system: The structural system containing nodes and elements (beams).
    - design_boundary: Polygon defining the design boundary.
    - holes: List of Polygon objects defining holes.
    - node_weight: Weight for node penalties.
    - beam_weight: Weight for beam penalties.
    - penalty_scale: Scaling factor for penalties.
    - ax: Matplotlib axis object for visualization. If None, no plot is generated.

    Returns:
    - Total domain penalty (weighted sum of node and beam penalties).
    """
    total_node_penalty = 0
    total_beam_penalty = 0

    if ax is not None:
        # Plot the design boundary
        x, y = design_boundary.exterior.xy
        ax.plot(x, y, 'blue', label='Design Boundary')

        # Plot holes
        if holes is not None:
            for hole in holes:
                x, y = hole.exterior.xy
                ax.plot(x, y, 'black', linestyle='--', label='Hole')

    # Node penalties
    for node in system.nodes:
        point = Point(node.coords)
        penalty = 0

        # Check if the node is outside the design boundary
        if not design_boundary.covers(point):
            nearest_point = design_boundary.exterior.interpolate(design_boundary.exterior.project(point))
            distance = design_boundary.exterior.distance(point)
            penalty += penalty_scale * distance

            # Visualization
            if ax is not None:
                ax.plot(
                    [node.coords[0], nearest_point.x],
                    [node.coords[1], nearest_point.y],
                    'orange', label='Node Penalty (Boundary)' if 'Node Penalty (Boundary)' not in ax.get_legend_handles_labels()[1] else ""
                )
                ax.scatter(node.coords[0], node.coords[1], color='orange', label='Invalid Node' if 'Invalid Node' not in ax.get_legend_handles_labels()[1] else "")

        # Check if the node is inside a hole
        if holes is not None:
            for hole in holes:
                if hole.contains(point):
                    nearest_point = hole.exterior.interpolate(hole.exterior.project(point))
                    distance = hole.exterior.distance(point)
                    penalty += penalty_scale * distance
    
                    # Visualization
                    if ax is not None:
                        ax.plot(
                            [node.coords[0], nearest_point.x],
                            [node.coords[1], nearest_point.y],
                            'orange', label='Node Penalty (Hole)' if 'Node Penalty (Hole)' not in ax.get_legend_handles_labels()[1] else ""
                        )
                        ax.scatter(node.coords[0], node.coords[1], color='orange', label='Invalid Node' if 'Invalid Node' not in ax.get_legend_handles_labels()[1] else "")

        total_node_penalty += penalty

    # Beam penalties
    for beam in system.elements:
        beam_segment = LineString([beam.nodes[0].coords, beam.nodes[1].coords])
        penalty = 0
        valid_segments = [beam_segment]  # Start with the full beam as valid

        if holes is not None:
            for hole in holes:
                if beam_segment.intersects(hole):
                    intersection = beam_segment.intersection(hole.exterior)
    
                    points = [Point(beam.nodes[0].coords)]
                    if intersection.geom_type == 'Point':
                        points.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        points.extend(intersection.geoms)  # Extract points from MultiPoint
                    points.append(Point(beam.nodes[1].coords))
    
                    points = sorted(points, key=lambda p: beam_segment.project(p))
                    subsegments = [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]
    
                    valid_segments = []
                    for segment in subsegments:
                        if segment.within(hole):
                            length_invalid = segment.length
                            penalty += penalty_scale * length_invalid
                            if ax is not None:
                                ax.plot(*segment.xy, color='red', linewidth=2, label='Invalid Beam (Hole)' if 'Invalid Beam (Hole)' not in ax.get_legend_handles_labels()[1] else "")
                        else:
                            valid_segments.append(segment)

        for segment in valid_segments:
            if not design_boundary.covers(segment):
                intersection = segment.intersection(design_boundary.exterior)

                points = [Point(segment.coords[0])]
                if intersection.geom_type == 'Point':
                    points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    # points.extend(list(intersection))
                    points.extend(intersection.geoms) 
                points.append(Point(segment.coords[-1]))

                points = sorted(points, key=lambda p: segment.project(p))
                subsegments = [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]

                valid_segments = []
                for subsegment in subsegments:
                    if not design_boundary.covers(subsegment):
                        length_invalid = subsegment.length
                        penalty += penalty_scale * length_invalid
                        if ax is not None:
                            ax.plot(*subsegment.xy, color='red', linewidth=2, label='Invalid Beam (Boundary)' if 'Invalid Beam (Boundary)' not in ax.get_legend_handles_labels()[1] else "")
                    else:
                        valid_segments.append(subsegment)

        for segment in valid_segments:
            if ax is not None:
                ax.plot(*segment.xy, color='green', linewidth=2, label='Valid Beam' if 'Valid Beam' not in ax.get_legend_handles_labels()[1] else "")

        total_beam_penalty += penalty

    # Weighted total penalty
    total_penalty = node_weight * total_node_penalty + beam_weight * total_beam_penalty

    if ax is not None and total_penalty>0 or plot:
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("Domain Penalty Visualization")
        ax.legend()
        ax.grid(True)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()
    else:
         plt.close()

    return total_penalty


def domain_penalty2(system, design_boundary, holes, node_weight=1.0, beam_weight=1.0, penalty_scale=1.0, n_samples=20, ax=None):
    """
    Calculate total domain penalty for a system, considering node and beam penalties, with visualization.

    Parameters:
    - system: The structural system containing nodes and elements (beams).
    - design_boundary: Polygon defining the design boundary.
    - holes: List of Polygon objects defining holes.
    - node_weight: Weight for node penalties.
    - beam_weight: Weight for beam penalties.
    - penalty_scale: Scaling factor for penalties.
    - n_samples: Number of sample points along the beam for evaluation.
    - ax: Matplotlib axis object for visualization. If None, no plot is generated.

    Returns:
    - Total domain penalty (weighted sum of node and beam penalties).
    """
    from shapely.geometry import Point, LineString
    import numpy as np

    total_node_penalty = 0
    total_beam_penalty = 0

    if ax is not None:
        # Plot the design boundary
        x, y = design_boundary.exterior.xy
        ax.plot(x, y, 'blue', label='Design Boundary')

        # Plot holes
        for hole in holes:
            x, y = hole.exterior.xy
            ax.plot(x, y, 'black', linestyle='--', label='Hole')

    # Node penalties
    for node in system.nodes:
        point = Point(node.coords)
        penalty = 0

        # Check if the node is outside the design boundary
        if not design_boundary.covers(point):
            nearest_point = design_boundary.exterior.interpolate(design_boundary.exterior.project(point))
            distance = design_boundary.exterior.distance(point)
            penalty += penalty_scale * distance

            # Visualization
            if ax is not None:
                ax.plot(
                    [node.coords[0], nearest_point.x],
                    [node.coords[1], nearest_point.y],
                    'orange', label='Node Penalty (Boundary)' if 'Node Penalty (Boundary)' not in ax.get_legend_handles_labels()[1] else ""
                )
                ax.scatter(node.coords[0], node.coords[1], color='orange', label='Invalid Node' if 'Invalid Node' not in ax.get_legend_handles_labels()[1] else "")

        # Check if the node is inside a hole
        for hole in holes:
            if hole.contains(point):
                nearest_point = hole.exterior.interpolate(hole.exterior.project(point))
                distance = hole.exterior.distance(point)
                penalty += penalty_scale * distance

                # Visualization
                if ax is not None:
                    ax.plot(
                        [node.coords[0], nearest_point.x],
                        [node.coords[1], nearest_point.y],
                        'orange', label='Node Penalty (Hole)' if 'Node Penalty (Hole)' not in ax.get_legend_handles_labels()[1] else ""
                    )
                    ax.scatter(node.coords[0], node.coords[1], color='orange', label='Invalid Node' if 'Invalid Node' not in ax.get_legend_handles_labels()[1] else "")

        total_node_penalty += penalty

    # Beam penalties
    for beam in system.elements:
        beam_segment = LineString([beam.nodes[0].coords, beam.nodes[1].coords])
        valid_segments = [beam_segment]
        invalid_segments = []

        # Split segments based on intersections with holes
        for hole in holes:
            if beam_segment.intersects(hole):
                intersection = beam_segment.intersection(hole.exterior)

                points = [Point(beam.nodes[0].coords)]
                if intersection.geom_type == 'Point':
                    points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    points.extend(intersection.geoms)
                points.append(Point(beam.nodes[1].coords))

                points = sorted(points, key=lambda p: beam_segment.project(p))
                subsegments = [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]

                valid_segments = []
                for segment in subsegments:
                    if segment.within(hole):
                        invalid_segments.append(segment)
                    else:
                        valid_segments.append(segment)

        # Split segments based on intersections with the design boundary
        for segment in valid_segments.copy():
            if not design_boundary.covers(segment):
                intersection = segment.intersection(design_boundary.exterior)

                points = [Point(segment.coords[0])]
                if intersection.geom_type == 'Point':
                    points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    points.extend(intersection.geoms)
                points.append(Point(segment.coords[-1]))

                points = sorted(points, key=lambda p: segment.project(p))
                subsegments = [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]

                for subsegment in subsegments:
                    if not design_boundary.covers(subsegment):
                        invalid_segments.append(subsegment)
                    else:
                        valid_segments.append(subsegment)

        # Plot valid segments
        if ax is not None:
            for segment in valid_segments:
                ax.plot(*segment.xy, color='green', linewidth=2, label='Valid Beam' if 'Valid Beam' not in ax.get_legend_handles_labels()[1] else "")

        # Plot invalid segments
        if ax is not None:
            for segment in invalid_segments:
                ax.plot(*segment.xy, color='red', linewidth=2, label='Invalid Beam' if 'Invalid Beam' not in ax.get_legend_handles_labels()[1] else "")

        # Highlight min max distance for invalid segments
        for segment in invalid_segments:
            sampled_points = [segment.interpolate(i / (n_samples - 1), normalized=True) for i in range(1, n_samples - 1)]

            # Beam axis vector
            start, end = segment.coords[0], segment.coords[1]
            beam_axis = np.array([end[0] - start[0], end[1] - start[1]])
            beam_axis = beam_axis / np.linalg.norm(beam_axis)

            # Perpendicular vectors
            perp_positive = np.array([-beam_axis[1], beam_axis[0]])
            perp_negative = np.array([beam_axis[1], -beam_axis[0]])

            max_dist_positive = -float('inf')
            max_dist_negative = -float('inf')
            max_conn_positive = None
            max_conn_negative = None

            for point in sampled_points:
                point_coords = np.array([point.x, point.y])

                nearest_positive, nearest_negative = None, None
                dist_positive, dist_negative = float('inf'), float('inf')

                for direction, perp_vector in [("positive", perp_positive), ("negative", perp_negative)]:
                    for step in np.linspace(0, 1, 50):
                        test_point_coords = point_coords + step * perp_vector
                        test_point = Point(test_point_coords)

                        in_design_space = design_boundary.contains(test_point)
                        in_hole = any(hole.contains(test_point) for hole in holes)

                        if in_design_space and not in_hole:
                            distance = np.linalg.norm(test_point_coords - point_coords)
                            if direction == "positive" and distance < dist_positive:
                                dist_positive = distance
                                nearest_positive = test_point
                            elif direction == "negative" and distance < dist_negative:
                                dist_negative = distance
                                nearest_negative = test_point
                            break

                if dist_positive > max_dist_positive:
                    max_dist_positive = dist_positive
                    max_conn_positive = (point, nearest_positive)
                if dist_negative > max_dist_negative:
                    max_dist_negative = dist_negative
                    max_conn_negative = (point, nearest_negative)

            min_max_conn = None
            if max_conn_positive and max_conn_negative:
                if max_dist_positive <= max_dist_negative:
                    min_max_conn = max_conn_positive
                else:
                    min_max_conn = max_conn_negative
            elif max_conn_positive:
                min_max_conn = max_conn_positive
            elif max_conn_negative:
                min_max_conn = max_conn_negative

            if min_max_conn and ax is not None:
                invalid_point, valid_point = min_max_conn
                distance = np.linalg.norm([valid_point.x - invalid_point.x, valid_point.y - invalid_point.y])
                total_beam_penalty += distance
                ax.plot(
                    [invalid_point.x, valid_point.x],
                    [invalid_point.y, valid_point.y],
                    'orange', linewidth=2, label='Min Max Distance' if 'Min Max Distance' not in ax.get_legend_handles_labels()[1] else ""
                )

    total_penalty = node_weight * total_node_penalty + beam_weight * total_beam_penalty

    if ax is not None and total_penalty > 0:
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("System Visualization with Domain Penalty")
        ax.legend()
        ax.grid(True)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()
    else:
        plt.close()

    return total_penalty


def shape_optimization(max_iter, system_shape_opt, design_boundary, holes, l_B=1, penalty_nodes=1, penalty_ele=1, domain_p_type=1, output_file=None, l_min=None):

    # Get bounding box from design boundary
    min_x, min_y, max_x, max_y = design_boundary.bounds
    
    # Compute dimensions
    dimension = [max_x - min_x, max_y - min_y]  # [width, height]
    
    # step size
    eta = min(dimension)/200
    
    if l_min is None:
        l_min = eta*10

    # Optimization parameters
    #max_iter = 200  # Maximum number of iterations
    tolerance = 1e-10  # Convergence tolerance for objective
    move_limit = eta*100  # Maximum allowable change in design update
  
    iteration = 0
    obj_prev = float('inf')  # Initialize previous objective to a large value
    objective_hist = [] 
    strain_energy_N_hist = []
    strain_energy_B_hist = []
    domain_penalty_hist = []

    while iteration < max_iter:
        
        # step 0: solve FE
        system_shape_opt.solve_FE()
        
        # Step 1: Compute objective and sensitivity
        fig, ax = plt.subplots(figsize=(10, 8))
        if domain_p_type == 1:
            total_penalty = domain_penalty1(system_shape_opt, design_boundary, holes, node_weight=penalty_nodes, beam_weight=penalty_ele, penalty_scale=1.0, ax=ax)
        elif domain_p_type == 2:
            total_penalty = domain_penalty2(system_shape_opt, design_boundary, holes, node_weight=penalty_nodes, beam_weight=penalty_ele, penalty_scale=1.0, ax=ax)
        else:
            print('invalid choice of domain penalty. Only option 1 or 2 are available. Therefore no penalty is used')
            total_penalty = 0

        
        u_N, u_B = system_shape_opt.strain_energy_beam_truss()
        strain_E =  u_N + u_B
        objective = u_N + l_B*u_B + total_penalty
        
        
        objective_hist.append(objective)
        strain_energy_N_hist.append(u_N)
        strain_energy_B_hist.append(u_B)
        domain_penalty_hist.append(total_penalty)

        d_c = np.zeros(len(system_shape_opt.nodes) * 2)  # Sensitivity array for x and y coordinates

        for i, node in enumerate(system_shape_opt.nodes):
            # Skip fixed nodes (coordinates are fixed)
            if any(node.fixed):
                d_c[i * 2] = 0
                d_c[i * 2 + 1] = 0
                # print(f"Node {i} fixed")
                continue

            # Skip nodes with non-zero external forces (coordinates are fixed)
            if np.linalg.norm(node.forces) > 0:  # Check if forces are non-zero
                d_c[i * 2] = 0
                d_c[i * 2 + 1] = 0
                # print(f"Node {i} has external forces")
                continue

            # Compute sensitivity for internal nodes
            for coord_index in range(2):  # x and y coordinates
                original_value = node.coords[coord_index]
                node.coords[coord_index] += eta  # Perturb the coordinate
                system_shape_opt.solve_FE()  # Recalculate system with perturbed geometry
                if domain_p_type == 1:
                    total_penalty = domain_penalty1(system_shape_opt, design_boundary, holes, node_weight=penalty_nodes, beam_weight=penalty_ele, penalty_scale=1.0)
                elif domain_p_type == 2:
                    total_penalty = domain_penalty2(system_shape_opt, design_boundary, holes, node_weight=penalty_nodes, beam_weight=penalty_ele, penalty_scale=1.0)
                else:
                    total_penalty = 0
                
                u_N, u_B = system_shape_opt.strain_energy_beam_truss()
                obj_var = u_N + l_B*u_B + total_penalty
                d_c[i * 2 + coord_index] = (obj_var - objective) / eta
                node.coords[coord_index] = original_value  # Reset to original

        # Step 2: Update nodal coordinates in the negative d_c direction
        # Update nodal coordinates with capped step size
        for i, node in enumerate(system_shape_opt.nodes):
            if any(node.fixed) or np.linalg.norm(node.forces) > 0:
                continue  # Skip fixed or loaded nodes
        
            # Compute step size for x and y directions
            step_x = max(min(eta * 10* d_c[i * 2], move_limit), -move_limit)  # Cap step size by move_limit_x
            step_y = max(min(eta * 10* d_c[i * 2 + 1], move_limit), -move_limit)  # Cap step size by move_limit_y
        
            # Update node coordinates with bounds checks
            node.coords[0] = max(min(node.coords[0] - step_x, max_x), min_x)
            node.coords[1] = max(min(node.coords[1] - step_y, max_y), min_y)


        # system_shape_opt.plot_deformed_stm_sf(100,10)

        # Step 3: Check convergence
        change = abs(obj_prev - objective)

        if change < tolerance:
            print("Convergence achieved!")
            break
        

        # Step 4: Check for merged nodes
        system_shape_opt.delete_short_elements(l_min)
       
        # Optional: Plot the deformed structure at each iteration
        if (iteration) % 10 ==0:
            print(f"Iteration {iteration + 1}")
            
            print(f"number of dofs {system_shape_opt.nr_dofs}")
            system_shape_opt.plot_deformed_stm_sf(100, scale=10, title=f'Iteration: {iteration}')
            print(f"Strain energy: {strain_E}, Change: {change}")
        
        
        # Update previous objective and iteration counter
        obj_prev = objective
        iteration += 1

    # Final output
    print("Optimization completed.")
    print(f"Final strain energy: {strain_E}")
    # calculate ratio of normal forces
    sts = system_shape_opt.sts()
    # formatted_sts = [f"{value[0]:.4f}" for value in sts]
    # print("STS per Element:", ", ".join(formatted_sts))
    print('sts:', np.mean(sts))

    # data
    iterations = np.arange(1, len(objective_hist) + 1)
    strain_energy_N =  np.array(strain_energy_N_hist)
    strain_energy_B =  np.array(strain_energy_B_hist)
    strain_energy = strain_energy_N + strain_energy_B
       
    
    # Plot 1: Strain Energies
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, strain_energy, label='Strain Energy', color='blue', linewidth=1.5)
    plt.plot(iterations, strain_energy_N, label='Axial Strain Energy', color='green', linestyle='--', linewidth=1.5)
    plt.plot(iterations, strain_energy_B, label='Bending Strain Energy', color='red', linestyle=':', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Strain Energy')
    plt.title('Strain Energy Histories')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("strain_energy_histories.pdf", format="pdf", dpi=300)
    plt.show()
    
    
    # Plot 2: Objective, strain energy, and Domain Penalty
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, objective_hist, label='Objective', color='purple', linewidth=1.5)
    plt.plot(iterations, strain_energy, label='Strain Energy', color='blue', linestyle='--', linewidth=1.5)
    plt.plot(iterations, domain_penalty_hist, label='Domain Penalty', color='orange', linestyle=':', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Objective, Strain Energy, and Domain Penalty')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("objective_compliance_penalty_histories.pdf", format="pdf", dpi=300)
    plt.show()

