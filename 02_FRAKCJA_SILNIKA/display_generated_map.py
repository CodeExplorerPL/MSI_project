import pygame
import csv
import os
import argparse
import sys

# Konfiguracja
TILE_SIZE = 32
SCROLL_SPEED = 10

# Ścieżki
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(BASE_DIR, 'backend', 'maps')
ASSETS_DIR = os.path.join(BASE_DIR, 'frontend', 'assets', 'tiles')

# Mapowanie nazw z CSV na pliki
TILE_MAPPING = {
    'Grass': 'grass.png',
    'Road': 'road.png',
    'Swamp': 'swamp.png',
    'PotholeRoad': 'potholeroad.png',
    'Water': 'water.png',
    'Wall': 'Wall.png',
    'Tree': 'tree.png',
    'AntiTankSpike': 'antitankspike.png'
}

def load_assets():
    """Ładuje i skaluje assety."""
    assets = {}
    if not os.path.exists(ASSETS_DIR):
        print(f"Error: Assets directory not found at {ASSETS_DIR}")
        sys.exit(1)
        
    for tile_name, filename in TILE_MAPPING.items():
        path = os.path.join(ASSETS_DIR, filename)
        try:
            img = pygame.image.load(path)
            img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
            assets[tile_name] = img
        except FileNotFoundError:
            print(f"Warning: Asset {filename} not found at {path}. Using placeholder.")
            s = pygame.Surface((TILE_SIZE, TILE_SIZE))
            s.fill((255, 0, 255)) # Magenta placeholder
            assets[tile_name] = s
        except Exception as e:
            # Pygame error handling if display is not initialized usually happens later,
            # but loading images requires pygame.display.set_mode() usually? 
            # Actually pygame.image.load() works after pygame.init().
            print(f"Error loading {filename}: {e}")
            # Don't exit, just use placeholder
            s = pygame.Surface((TILE_SIZE, TILE_SIZE))
            s.fill((255, 0, 0)) 
            assets[tile_name] = s
            
    return assets

def load_map(filename):
    """Wczytuje mapę z pliku CSV."""
    # Check if filename is a path or just a name in maps dir
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.join(MAPS_DIR, filename)
        
    if not os.path.exists(path):
        print(f"Error: Map file not found at {path}")
        return None
            
    map_data = []
    try:
        with open(path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row: # Skip empty rows
                    map_data.append(row)
    except Exception as e:
        print(f"Error reading map file: {e}")
        return None
        
    return map_data

def main():
    parser = argparse.ArgumentParser(description="Display a generated map using game assets.")
    parser.add_argument("map_name", nargs='?', help="Name of the map file in maps directory (e.g. map1.csv)")
    parser.add_argument("--tile-size", type=int, default=32, help="Size of tiles in pixels (default: 32)")
    args = parser.parse_args()

    global TILE_SIZE
    TILE_SIZE = args.tile_size

    # List maps if no argument provided
    if not args.map_name:
        print("Available maps:")
        if os.path.exists(MAPS_DIR):
            found = False
            for f in os.listdir(MAPS_DIR):
                if f.endswith('.csv'):
                    print(f" - {f}")
                    found = True
            if not found:
                print(" (No .csv maps found)")
        else:
            print(f" (Maps directory not found: {MAPS_DIR})")
        print("\nUsage: python display_generated_map.py <map_filename>")
        print("Optional: --tile-size <pixels>")
        return

    # Init Pygame
    try:
        pygame.init()
    except Exception as e:
        print(f"Failed to initialize pygame: {e}")
        print("Please ensure pygame is installed: pip install pygame")
        return

    # Load resources
    print("Loading assets...")
    assets = load_assets()
    
    print(f"Loading map: {args.map_name}...")
    map_data = load_map(args.map_name)

    if not map_data:
        pygame.quit()
        return

    rows = len(map_data)
    cols = max(len(r) for r in map_data) if rows > 0 else 0
    
    print(f"Map loaded. Size: {cols}x{rows} tiles.")
    
    map_width_px = cols * TILE_SIZE
    map_height_px = rows * TILE_SIZE

    # Screen setup (max 80% of screen resolution usually, hardcoded safe limit here)
    screen_width = min(1280, map_width_px)
    screen_height = min(800, map_height_px)
    
    # If map is smaller than min window, use map size
    screen_width = max(400, screen_width)
    screen_height = max(300, screen_height)
    
    # But strictly limit to map size if map is smaller than screen limits? 
    # Better to center it. Let's just clamp to map size if smaller than screen limits.
    display_w = min(1280, map_width_px)
    display_h = min(800, map_height_px)
    
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption(f"Map Viewer - {args.map_name} ({cols}x{rows}) - Arrows to Scroll")

    clock = pygame.time.Clock()
    camera_x = 0
    camera_y = 0

    running = True
    print("Viewer started. Use Arrow keys to move camera. ESC to exit.")
    
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 2. Input (Camera movement)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera_x = max(0, camera_x - SCROLL_SPEED)
        if keys[pygame.K_RIGHT]:
            camera_x = min(max(0, map_width_px - display_w), camera_x + SCROLL_SPEED)
        if keys[pygame.K_UP]:
            camera_y = max(0, camera_y - SCROLL_SPEED)
        if keys[pygame.K_DOWN]:
            camera_y = min(max(0, map_height_px - display_h), camera_y + SCROLL_SPEED)

        # 3. Drawing
        screen.fill((30, 30, 30)) # Dark background

        # Calculate visible tile range to optimize rendering
        start_col = max(0, camera_x // TILE_SIZE)
        end_col = min(cols, (camera_x + display_w) // TILE_SIZE + 1)
        start_row = max(0, camera_y // TILE_SIZE)
        end_row = min(rows, (camera_y + display_h) // TILE_SIZE + 1)

        for r in range(start_row, end_row):
            # Check if row index is valid (map might be jagged list)
            if r >= len(map_data): 
                continue
                
            row_data = map_data[r]
            for c in range(start_col, end_col):
                if c >= len(row_data):
                    continue
                    
                tile_name = row_data[c].strip()
                
                x = c * TILE_SIZE - camera_x
                y = r * TILE_SIZE - camera_y
                
                if tile_name in assets:
                    screen.blit(assets[tile_name], (x, y))
                else:
                    # Unknown tile - draw text or color
                    pygame.draw.rect(screen, (100, 100, 100), (x, y, TILE_SIZE, TILE_SIZE))
                    pygame.draw.rect(screen, (255, 0, 0), (x, y, TILE_SIZE, TILE_SIZE), 1)

        # Draw UI overlay
        # Optional: Coordinates or help text
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
