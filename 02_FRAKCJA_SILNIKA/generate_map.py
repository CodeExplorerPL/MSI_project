""" Skrypt do generowania map """
import csv
import random
import os
import argparse
import sys

# Getting the absolute path to the maps directory
MAPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'maps'))

OBSTACLE_TYPES = ['Wall', 'Tree', 'AntiTankSpike']
TERRAIN_TYPES = ['Grass', 'Road', 'Swamp', 'PotholeRoad', 'Water']

def generate_map(width, height, filename, tile_ratios):
    """
    Generates a random map based on specified tile ratios and saves it as a CSV file.
    """
    if not os.path.exists(MAPS_DIR):
        os.makedirs(MAPS_DIR)

    num_tiles = width * height

    population = list(tile_ratios.keys())
    weights = list(tile_ratios.values())

    # Generate the tiles based on the ratios
    tiles = random.choices(population, weights=weights, k=num_tiles)

    # Arrange tiles into a grid
    map_data = []
    for i in range(height):
        row = tiles[i * width:(i + 1) * width]
        map_data.append(row)

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
    
    # New arguments for obstacle and terrain ratios
    parser.add_argument("--obstacle-ratio", type=float, default=0.2, help="The total ratio of obstacle tiles.")
    parser.add_argument("--terrain-ratio", type=float, default=0.8, help="The total ratio of terrain tiles.")
    parser.add_argument("--obstacle-types", nargs='+', default=OBSTACLE_TYPES, help=f"List of obstacle types to use. Default: {' '.join(OBSTACLE_TYPES)}")
    parser.add_argument("--terrain-types", nargs='+', default=TERRAIN_TYPES, help=f"List of terrain types to use. Default: {' '.join(TERRAIN_TYPES)}")

    args = parser.parse_args()

    obstacle_ratio = args.obstacle_ratio
    terrain_ratio = args.terrain_ratio
    selected_obstacle_types = args.obstacle_types
    selected_terrain_types = args.terrain_types

    # Validation for main ratios
    if obstacle_ratio < 0 or terrain_ratio < 0:
        print("Error: Obstacle and terrain ratios cannot be negative.", file=sys.stderr)
        sys.exit(1)
        
    if abs(obstacle_ratio + terrain_ratio - 1.0) > 1e-6:
        print(f"Error: The sum of obstacle and terrain ratios must be 1.0. Current sum is {obstacle_ratio + terrain_ratio}", file=sys.stderr)
        print("Please adjust the ratios. For example: --obstacle-ratio 0.3 --terrain-ratio 0.7", file=sys.stderr)
        sys.exit(1)

    tile_ratios = {}

    # Distribute obstacle ratio evenly among obstacle types
    if obstacle_ratio > 0 and selected_obstacle_types:
        part = obstacle_ratio / len(selected_obstacle_types)
        for o in selected_obstacle_types:
            tile_ratios[o] = part
    
    # Distribute terrain ratio evenly among terrain types
    if terrain_ratio > 0 and selected_terrain_types:
        part = terrain_ratio / len(selected_terrain_types)
        for t in selected_terrain_types:
            tile_ratios[t] = part
            
    if not tile_ratios:
        print("Error: No tile types defined or ratios provided. Cannot generate map.", file=sys.stderr)
        sys.exit(1)

    generate_map(args.width, args.height, args.filename, tile_ratios)

