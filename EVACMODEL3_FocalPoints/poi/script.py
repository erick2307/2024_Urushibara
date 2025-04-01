# GTEM Tsunami Evacuation Simulation Framework

import os
import requests
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime
from scipy.spatial import KDTree
from networkx import Graph, shortest_path, from_edgelist, all_shortest_paths
import networkx as nx


# --- Step 1: Data Collection ---

class DataCollector:
    def __init__(self, city_name):
        self.city_name = city_name
        self.data_dir = f'data/{city_name}'
        os.makedirs(self.data_dir, exist_ok=True)

    def download_gis_data(self, url, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_name}")
        return file_path

    def load_gis_data(self, file_path):
        return gpd.read_file(file_path)


# --- Step 2: Data Preparation ---

def prepare_data(road_network, population_data, tsunami_map):
    road_network = road_network.to_crs('EPSG:4326')
    population_data = population_data.to_crs('EPSG:4326')
    tsunami_map = tsunami_map.to_crs('EPSG:4326')

    road_network = road_network[['geometry']]
    population_data = population_data[['geometry', 'population']]

    road_edges = []
    for _, line in road_network.iterrows():
        if isinstance(line['geometry'], LineString):
            coords = list(line['geometry'].coords)
            for i in range(len(coords) - 1):
                p1 = Point(coords[i])
                p2 = Point(coords[i + 1])
                weight = np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1]))
                if tsunami_map.contains(p1).any() or tsunami_map.contains(p2).any():
                    weight *= 10  
                road_edges.append((coords[i], coords[i + 1], {'weight': weight}))

    road_graph = from_edgelist(road_edges)

    return road_network, population_data, road_graph


# --- Step 3: Model Setup ---

def setup_simulation(road_network, population_data, road_graph):
    agents = []
    shelters = []

    for idx, road in road_network.iterrows():
        if idx % 50 == 0:
            shelters.append(road['geometry'].centroid)

    shelter_tree = KDTree([(s.x, s.y) for s in shelters])

    for idx, row in population_data.iterrows():
        for _ in range(int(row['population'])):
            position = row['geometry'].centroid
            _, nearest_shelter_idx = shelter_tree.query([position.x, position.y])
            agent = {
                'position': position,
                'status': 'evacuating',
                'speed': np.random.uniform(1.0, 1.5),
                'target': shelters[nearest_shelter_idx],
                'congestion_factor': 1.0  
            }
            agents.append(agent)

    return agents, shelters


# --- Step 4: Simulation Execution ---

def run_simulation(agents, road_graph, steps=200):
    evacuation_times = []
    agent_paths = []  

    for step in range(steps):
        congestion_map = {}

        for agent in agents:
            if agent['status'] == 'evacuating':
                current_pos = (agent['position'].x, agent['position'].y)
                target = (agent['target'].x, agent['target'].y)

                if current_pos not in congestion_map:
                    congestion_map[current_pos] = 0
                congestion_map[current_pos] += 1

                if nx.has_path(road_graph, current_pos, target):
                    all_paths = list(all_shortest_paths(road_graph, current_pos, target, weight='weight'))
                    if all_paths:
                        path = min(all_paths, key=lambda p: sum(road_graph.edges[p[i], p[i+1]]['weight'] for i in range(len(p)-1)))
                        if len(path) > 1:
                            next_pos = path[1]
                            congestion_value = congestion_map[current_pos]
                            speed_reduction = min(1.0, 0.5 + congestion_value * 0.05)
                            agent['speed'] *= (1.0 / speed_reduction)
                            agent['position'] = Point(next_pos)
                            agent_paths.append((current_pos, next_pos))
                        else:
                            agent['status'] = 'safe'
                            evacuation_times.append(step)

    return evacuation_times, agent_paths


# --- Step 5: Data Analysis ---

def analyze_results(evacuation_times, agent_paths, city_name):
    evacuation_times = np.array(evacuation_times)
    avg_time = np.mean(evacuation_times)
    median_time = np.median(evacuation_times)

    plt.figure(figsize=(10, 6))
    plt.hist(evacuation_times, bins=30, alpha=0.7)
    plt.title(f'{city_name} - Evacuation Time Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Number of Agents')
    plt.show()

    plt.figure(figsize=(12, 8))
    for path in agent_paths:
        plt.plot([path[0][0], path[1][0]], [path[0][1], path[1][1]], color='blue', alpha=0.5)
    plt.title(f'{city_name} - Evacuation Routes Visualization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    print(f"{city_name} - Average Evacuation Time: {avg_time} steps")
    print(f"{city_name} - Median Evacuation Time: {median_time} steps")


# --- Step 6: Execution Example ---
if __name__ == "__main__":
    cities = ['Sendai', 'Ishinomaki', 'Kamaishi']  
    results = []

    for city_name in cities:
        collector = DataCollector(city_name)

        road_data_path = collector.download_gis_data(f'https://example.com/{city_name}_road_network.shp', 'road_network.shp')
        population_data_path = collector.download_gis_data(f'https://example.com/{city_name}_population_data.shp', 'population_data.shp')
        tsunami_map_path = collector.download_gis_data(f'https://example.com/{city_name}_tsunami_map.shp', 'tsunami_map.shp')

        road_network = collector.load_gis_data(road_data_path)
        population_data = collector.load_gis_data(population_data_path)
        tsunami_map = collector.load_gis_data(tsunami_map_path)

        road_network, population_data, road_graph = prepare_data(road_network, population_data, tsunami_map)
        agents, shelters = setup_simulation(road_network, population_data, road_graph)

        evacuation_times, agent_paths = run_simulation(agents, road_graph)
        analyze_results(evacuation_times, agent_paths, city_name)
        results.append({'city': city_name, 'average_time': np.mean(evacuation_times), 'median_time': np.median(evacuation_times)})

    # Comparative Analysis Plot
    df = pd.DataFrame(results)
    df.plot(x='city', y=['average_time', 'median_time'], kind='bar', title='Comparative Analysis of Evacuation Times')
    plt.ylabel('Evacuation Time (Steps)')
    plt.show()
