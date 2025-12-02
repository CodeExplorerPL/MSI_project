import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

script_dir = Path(__file__).resolve().parent
file_name = script_dir / 'maps' / 'input.csv'
data_list = []

try:
    with open(file_name, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                data_list.append(row)

    print(f"\nSukces: Wczytano dane z {file_name}")
    print(f"Liczba wierszy: {len(data_list)}")
except FileNotFoundError:
    print(f"BŁĄD: Plik nie został znaleziony pod ścieżką: {file_name}")
    exit()

color_definitions = {
    "Water":       'blue',
    "Grass":       'green',
    "Road":        'lightgray',
    "Wall":        'saddlebrown',
    "Tree":        'darkgreen',
    "PotholeRoad": 'dimgray',
    "Swamp":       'yellow',
    "AntiTankSpike": 'black'
}

unique_terrains = sorted(list(set(cell for row in data_list for cell in row)))
terrain_mapping = {name: i for i, name in enumerate(unique_terrains)}
colors = [color_definitions.get(name, 'magenta') for name in unique_terrains]
unknown_terrains = [name for name in unique_terrains if name not in color_definitions]
if unknown_terrains:
    print(f"Uwaga: Następujące tereny nie mają zdefiniowanego koloru i będą różowe (magenta): {unknown_terrains}")

map_array = np.array([
    [terrain_mapping[cell] for cell in row]
    for row in data_list
], dtype=int)

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(len(unique_terrains) + 1) - 0.5, len(unique_terrains))
plt.figure(figsize=(10, 10))
plt.imshow(map_array, cmap=cmap, norm=norm)
tick_positions = np.arange(len(unique_terrains))
cbar = plt.colorbar(ticks=tick_positions, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_ticklabels(unique_terrains)
cbar.ax.tick_params(labelsize=10) # Lepsza czytelność legendy
plt.grid(True, color='black', linewidth=0.5, alpha=0.5)
plt.title('Wizualizacja Mapy Terenów (Niestandardowe Kolory)')
plt.xlabel('Kolumna')
plt.ylabel('Wiersz')
plt.xticks(np.arange(map_array.shape[1]))
plt.yticks(np.arange(map_array.shape[0]))
plt.gca().set_aspect('equal', adjustable='box')
plots_dir = script_dir / 'plots'
plots_dir.mkdir(exist_ok=True)
output_path = plots_dir / 'mapa_terenu_custom.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"\nWykres został zapisany jako: {output_path}")