""" Skrypt do generowania map """
import csv
import random
import os
import argparse
import sys
import numpy as np
from collections import deque

# Getting the absolute path to the maps directory
MAPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'maps'))

OBSTACLE_TYPES = ['Wall', 'Tree', 'AntiTankSpike']
TERRAIN_TYPES = ['Grass', 'Road', 'Swamp', 'PotholeRoad', 'Water']

def smooth_grid(grid, iterations=5):
    """
    Applies a simple averaging blur to the grid to create grouping.
    """
    for _ in range(iterations):
        # Rolling average with neighbors to smooth the noise
        up = np.roll(grid, -1, axis=0)
        down = np.roll(grid, 1, axis=0)
        left = np.roll(grid, -1, axis=1)
        right = np.roll(grid, 1, axis=1)
        grid = (grid + up + down + left + right) / 5.0
    return grid

def get_connected_components(map_data, passable_set):
    """
    Returns a list of sets, where each set contains (r, c) coordinates of a connected component.
    """
    rows = len(map_data)
    cols = len(map_data[0])
    visited = set()
    components = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and map_data[r][c] in passable_set:
                # Start BFS
                component = set()
                queue = deque([(r, c)])
                visited.add((r, c))
                component.add((r, c))
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    
                    # Check 4 neighbors
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if (nr, nc) not in visited and map_data[nr][nc] in passable_set:
                                visited.add((nr, nc))
                                component.add((nr, nc))
                                queue.append((nr, nc))
                components.append(component)
    return components

def connect_components(map_data, components, passable_set, fill_type):
    """
    Connects multiple components into one using the shortest path.
    """
    if len(components) <= 1:
        return

    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    main_comp = components[0]
    
    rows = len(map_data)
    cols = len(map_data[0])

    # Create a grid of "component IDs" for fast lookup
    # 0 for main component, 1..N for others. -1 for obstacle.
    # We will simply manage 'other_nodes' set for checking targets.
    
    other_nodes = set()
    for i in range(1, len(components)):
        for node in components[i]:
            other_nodes.add(node)
            
    # While there are still disconnected nodes
    while other_nodes:
        # BFS from main_comp to find closest 'other_node'
        queue = deque()
        visited = {} # coord -> parent
        
        # Add all main_comp nodes to queue (dist 0)
        for node in main_comp:
            queue.append(node)
            visited[node] = None
            
        found_target = None
        
        while queue:
            curr = queue.popleft()
            r, c = curr
            
            if curr in other_nodes:
                found_target = curr
                break
            
            # Expand
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited:
                        visited[(nr, nc)] = curr
                        queue.append((nr, nc))
        
        if found_target:
            # Reconstruct path and carve it
            curr = found_target
            path_nodes = []
            while curr is not None:
                path_nodes.append(curr)
                curr = visited[curr]
                
            # 'path_nodes' goes from Target -> Main
            for r, c in path_nodes:
                if map_data[r][c] not in passable_set:
                    map_data[r][c] = fill_type
                
                if (r, c) in other_nodes:
                    other_nodes.remove((r, c))
                
                main_comp.add((r, c))
                
            # Identify which component `found_target` belonged to and merge all its nodes to main_comp
            comp_idx_to_remove = -1
            for i in range(1, len(components)):
                if found_target in components[i]:
                    comp_idx_to_remove = i
                    break
            
            if comp_idx_to_remove != -1:
                # Merge this component into main
                for node in components[comp_idx_to_remove]:
                    if node in other_nodes:
                        other_nodes.remove(node)
                    main_comp.add(node)
                # Remove from list
                del components[comp_idx_to_remove]
                
        else:
            # Cannot reach any other node? Should not happen in grid.
            break

def ensure_neighbors(map_data, passable_set, fill_type):
    """
    Ensures every passable tile has at least one passable neighbor.
    """
    rows = len(map_data)
    cols = len(map_data[0])
    
    for r in range(rows):
        for c in range(cols):
            if map_data[r][c] in passable_set:
                # Check neighbors
                has_passable = False
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append((nr, nc))
                        if map_data[nr][nc] in passable_set:
                            has_passable = True
                            break
                
                if not has_passable and neighbors:
                    # Pick a random neighbor and make it passable
                    nr, nc = random.choice(neighbors)
                    map_data[nr][nc] = fill_type

def generate_map(width, height, filename, obstacle_ratio, terrain_ratio, obstacle_types, terrain_types):
    if not os.path.exists(MAPS_DIR):
        os.makedirs(MAPS_DIR)

    # 1. Generate Noise Maps using Numpy
    # Shape map: Obstacle vs Terrain
    shape_noise = np.random.rand(height, width)
    shape_noise = smooth_grid(shape_noise, iterations=8) # Increased iterations for better blobs
    
    # Biome map: Which Terrain?
    terrain_noise = np.random.rand(height, width)
    terrain_noise = smooth_grid(terrain_noise, iterations=4)
    
    # Obstacle Detail map: Which Obstacle?
    obstacle_noise = np.random.rand(height, width)
    obstacle_noise = smooth_grid(obstacle_noise, iterations=4)
    
    # 2. Determine Thresholds
    # Flatten to find percentiles
    shape_flat = np.sort(shape_noise.flatten())
    idx = int(obstacle_ratio * len(shape_flat))
    idx = min(idx, len(shape_flat) - 1)
    obs_threshold = shape_flat[idx]
    
    def create_mapper(noise_grid, types):
        flat = np.sort(noise_grid.flatten())
        boundaries = []
        n = len(types)
        for i in range(1, n):
            boundaries.append(flat[int(i / n * len(flat))])
        
        def mapper(val):
            for i, b in enumerate(boundaries):
                if val < b:
                    return types[i]
            return types[-1]
        return mapper

    terrain_mapper = create_mapper(terrain_noise, terrain_types)
    obstacle_mapper = create_mapper(obstacle_noise, obstacle_types)

    # 3. Construct Map
    map_data = []
    for r in range(height):
        row = []
        for c in range(width):
            if shape_noise[r, c] < obs_threshold:
                # Obstacle
                tile = obstacle_mapper(obstacle_noise[r, c])
            else:
                # Terrain
                tile = terrain_mapper(terrain_noise[r, c])
            row.append(tile)
        map_data.append(row)

    # 4. Post-processing: Connectivity & Neighbors
    passable_set = set(terrain_types)
    fill_type = terrain_types[0] if terrain_types else 'Grass'

    ensure_neighbors(map_data, passable_set, fill_type)

    components = get_connected_components(map_data, passable_set)
    if components:
        connect_components(map_data, components, passable_set, fill_type)
    else:
        # If no passable tiles (unlikely), fill center
        map_data[height//2][width//2] = fill_type

    # 5. Save
    filepath = os.path.join(MAPS_DIR, filename)
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map_data)

    print(f"Map generated and saved to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a map for the MSI project."
    )
    parser.add_argument("--width", type=int, default=50, help="The width of the map.")
    parser.add_argument("--height", type=int, default=50, help="The height of the map.")
    parser.add_argument("--filename", type=str, default="generated_map.csv", help="The name of the output file.")
    
    parser.add_argument("--obstacle-ratio", type=float, default=0.2, help="The total ratio of obstacle tiles.")
    parser.add_argument("--terrain-ratio", type=float, default=0.8, help="The total ratio of terrain tiles.")
    parser.add_argument("--obstacle-types", nargs='+', default=OBSTACLE_TYPES, help=f"List of obstacle types to use. Default: {' '.join(OBSTACLE_TYPES)}")
    parser.add_argument("--terrain-types", nargs='+', default=TERRAIN_TYPES, help=f"List of terrain types to use. Default: {' '.join(TERRAIN_TYPES)}")

    args = parser.parse_args()

    # Validation
    if args.obstacle_ratio < 0 or args.terrain_ratio < 0:
        print("Error: Obstacle and terrain ratios cannot be negative.", file=sys.stderr)
        sys.exit(1)
        
    if abs(args.obstacle_ratio + args.terrain_ratio - 1.0) > 1e-6:
        print(f"Error: The sum of ratios must be 1.0. Current: {args.obstacle_ratio + args.terrain_ratio}", file=sys.stderr)
        sys.exit(1)

    generate_map(
        args.width, 
        args.height, 
        args.filename, 
        args.obstacle_ratio, 
        args.terrain_ratio,
        args.obstacle_types,
        args.terrain_types
    )