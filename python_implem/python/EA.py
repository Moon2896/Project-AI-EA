### Imports

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
import random
import matplotlib.pyplot as plt
import folium
import csv

### class

class individual:

    def __init__(
        self,
        n_cities = None,
        randomize = True,
        fixed = None
     ):
        self.n = n_cities
        if fixed:
            self.path = fixed
        else:
            self.path = [i for i in range(self.n)] # [1,..., n]
            if randomize:
                np.random.shuffle(self.path) # random permutation of 
        
    def evaluate(self, D):
        # Create an array of city indices from the path
        city_indices = np.array(self.path)
        
        # Create an array of next city indices by shifting the city_indices array by one position
        next_city_indices = np.roll(city_indices, shift=-1)
        
        # Use advanced indexing to get the distances from D
        distances = D[city_indices, next_city_indices]
        
        # Sum the distances to get the total distance
        total_distance = np.sum(distances)
        
        self.total_distance = total_distance
        return total_distance

    
    def mutate(self):
        # Swap c_i and c_j
        i, j = random.choices(range(self.n), k=2)
        self.path[i], self.path[j] = self.path[j], self.path[i]
        return self

    def crossover(self, other):
        # ordered crossover
        start, end = sorted(random.choices(range(self.n), k=2))
        segment = self.path[start:end+1]
        
        newborn_path = [-1] * self.n
        newborn_path[start:end+1] = segment
        
        pointer = (end + 1) % self.n
        for city in other.path:
            if city not in segment:
                newborn_path[pointer] = city
                pointer = (pointer + 1) % self.n
        
        return individual(n_cities=self.n, fixed=newborn_path)
    
    def __str__(self) -> str:
        res = str(self.path)
        return res

### Viz

def visualize_paths_grid(individuals, cities_coordinates, distance_matrix, grid_shape=None):
    """
    Visualize multiple paths in a grid of plots.
    
    Parameters:
    - individuals: A list of instances of the 'individual' class.
    - cities_coordinates: A list of (x, y) coordinates for each city.
    - distance_matrix: A matrix of distances between cities.
    - grid_shape: Tuple specifying the shape of the grid (rows, columns). If None, a square grid is used.
    """
    n_individuals = len(individuals)
    
    # Determine grid shape if not provided
    if not grid_shape:
        grid_size = int(np.ceil(np.sqrt(n_individuals)))
        grid_shape = (grid_size, grid_size)
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 15))
    
    # Flatten axes for easy iteration
    axes = axes.ravel()
    
    for i, ind in enumerate(individuals):
        ax = axes[i]
        
        # Plot cities
        for city_coord in cities_coordinates:
            ax.scatter(city_coord[0], city_coord[1], color='blue', marker='o')
        
        # Plot individual's path
        x_coords = [cities_coordinates[city][0] for city in ind.path]
        y_coords = [cities_coordinates[city][1] for city in ind.path]
        
        # Add the starting city to the end to close the loop
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        ax.plot(x_coords, y_coords, color='red', linestyle='-', linewidth=1)
        ax.set_title(f'Path {i+1} - Distance: {ind.evaluate(distance_matrix):.2f}')
        ax.grid(True)
    
    # Remove any unused subplots
    for i in range(n_individuals, grid_shape[0] * grid_shape[1]):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_path_cities(individual, cities_df, distance_matrix):
    """
    Visualize the path taken by an individual.
    
    Parameters:
    - individual: An instance of the 'individual' class.
    - cities_df: A DataFrame with columns 'city', 'lat', and 'lng'.
    """
    # Extract the lat and lng coordinates for the cities in the path
    lat_coords = [cities_df.loc[city, 'lat'] for city in individual.path]
    lng_coords = [cities_df.loc[city, 'lng'] for city in individual.path]
    
    # Add the starting city to the end to close the loop
    lat_coords.append(lat_coords[0])
    lng_coords.append(lng_coords[0])
    
    # Plot the cities as points
    plt.scatter(lng_coords, lat_coords, color='blue', marker='o')
    
    # Label the cities with their names
    for i, city in enumerate(individual.path):
        plt.text(cities_df.loc[city, 'lng'], cities_df.loc[city, 'lat'], city, fontsize=12, ha='right')
    
    # Plot the path
    plt.plot(lng_coords, lat_coords, color='red', linestyle='-', linewidth=1)
    
    plt.title(f'Path taken by the salesman Distance: {individual.evaluate(distance_matrix):.2f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

### Epochs

def compute_fitness(individual, distance_matrix):
   return 1 / individual.evaluate(distance_matrix)

# import multiprocessing as mp

def epoch(individuals, distance_matrix, alpha=0.5, beta=0.5, gamma=0.5):
    
    n_individuals = len(individuals)
    
    # 1. Selection
    fitnesses = [compute_fitness(ind, distance_matrix) for ind in individuals]
    
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_parents = np.random.choice(individuals, size=n_individuals, p=probabilities)

    # 2. Crossover
    n_crossover = int(n_individuals * gamma)
    offspring = []
    for _ in range(n_crossover // 2):  # //2 because each iteration produces 2 offspring
        parent1, parent2 = random.choices(selected_parents, k=2)
        child1 = parent1.crossover(parent2)
        child2 = parent2.crossover(parent1)
        offspring.extend([child1, child2])
    
    # 3. Mutation
    n_mutation = int(n_individuals * alpha)
    for _ in range(n_mutation):
        individual_to_mutate = random.choice(offspring)
        n_genes_to_mutate = 1 #min(int(min(beta,0.5) * individual_to_mutate.n),1)
        for _ in range(n_genes_to_mutate):
            individual_to_mutate.mutate()
    
    # 4. Replacement
    best_parent = individuals[np.argmax(probabilities)]
    individuals = np.concatenate([[best_parent], offspring, selected_parents])
    individuals = individuals[:n_individuals]
    
    return individuals

def update_statistics(df, distance_matrix, iteration, individuals, alpha, beta, gamma):
    # Iteration step
    step = iteration

    # Best individual
    scores = [ind.evaluate(distance_matrix) for ind in individuals]
    best_individual = min(individuals, key=lambda x: x.evaluate(distance_matrix))
    
    # Number of same individuals
    unique_individuals = set(tuple(ind.path) for ind in individuals)
    num_same_individuals = len(individuals) - len(unique_individuals)
    
    # Number of shared patterns between individuals
    # Here, we'll count shared subpaths of length 3 as an example
    patterns = []
    for ind in individuals:
        for i in range(len(ind.path) - 2):
            patterns.append(tuple(ind.path[i:i+3]))
    num_shared_patterns = len(patterns) - len(set(patterns))
    
    # Score
    best_score = best_individual.total_distance
    
    # Append to DataFrame
    new_row = {
        'Iteration': step,
        'Number of Individuals': len(individuals),
        'Best Individual': best_individual.path,
        'All Scores': scores,
        'Number of Same Individuals': num_same_individuals,
        'Number of Shared Patterns': num_shared_patterns,
        'Score': best_score,
        'alpha':alpha,
        'beta':beta,
        'gamma':gamma,
        'Median':np.median(scores),
        'Q1':np.quantile(scores, 0.25),
        'Q3':np.quantile(scores, 0.75),
        'Max':np.max(scores)
    }
    df = df._append(new_row, ignore_index=True)

    append_to_csv(output_csv_path, new_row)    
    
    return df

def append_to_csv(filename, data):
    try:
        # Open the CSV file in append mode
        with open(filename, mode='a', newline='') as file:
            fieldnames = data.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')

            # If the file is empty, write the header
            if file.tell() == 0:
                writer.writeheader()

            # Write the data to the CSV file
            writer.writerow(data)

        #print("Data appended to", filename)
    except Exception as e:
        print("Error:", e)

def clear_csv_file(filename):
    try:
        # Open the CSV file in write mode to clear its contents
        with open(filename, mode='w', newline=''):
            pass  # The 'pass' statement does nothing, effectively clearing the file

        print("Cleared", filename)
    except Exception as e:
        print("Error:", e)

def compute_spherical_D(df):
    n = df.shape[0]
    r = 6371e3 # earth radiu
    D = np.zeros((n,n))
    for i in tqdm(range(n)):
        lat1 = df["lat"].iloc[i]
        phi1 = lat1 * np.pi / 180

        lng1 = df["lng"].iloc[i]
        tht1 = lng1 * np.pi / 180

        for j in range(i):
            lat2 = df["lat"].iloc[j]
            phi2 = lat2 * np.pi / 180
            dphi = phi2-phi1
            
            lng2 = df["lng"].iloc[j]
            tht2 = lng2 * np.pi / 180
            dtht = tht2-tht1

            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dtht/2)**2
            c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
            d = r*c

            D[i][j] = d
    return D + D.T

def train(distance_matrix, max_cities, n_individuals, initial_alpha, initial_beta, initial_gamma, max_iterations=1_000, early_stopping_rounds=250):
    n_cities = min(max_cities, distance_matrix.shape[0])
    
    # Initialize DataFrame
    columns = ['Iteration', 'Best Individual', 'Number of Same Individuals', 'Number of Shared Patterns', 'Median', 'Q1', 'Q3', 'Max']
    statistics_df = pd.DataFrame(columns=columns)

    individuals = [individual(n_cities=n_cities) for _ in range(n_individuals)]
    
    best_score = float('inf')
    no_improvement_counter = 0
    
    alpha, beta, gamma = initial_alpha, initial_beta, initial_gamma

    for i in tqdm(range(max_iterations)):
        individuals = epoch(individuals, distance_matrix, alpha, beta, gamma)
        statistics_df = update_statistics(statistics_df, distance_matrix, i, individuals, alpha, beta, gamma)
        
        current_score = statistics_df.iloc[-1]['Score']
        
        # Score tracking
        if current_score < best_score:
            best_score = current_score
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
        
        # Early stopping
        if no_improvement_counter >= early_stopping_rounds:
            print(f"Early stopping after {i} iterations.")
            break
        
        # Gradual change of hyperparameters
        # A slow decay rate as the rate of convergence of EA is sllllow 
        decay_rate = 0.99 
        alpha *= decay_rate
        # Adding perturbation in the number of mutations over the population
        # Mainly, if there is not enough mutation or there is no improvement
        if (alpha < 0.01 or no_improvement_counter > early_stopping_rounds //2):
            alpha = max(0.25,alpha) # a quarter of the population is mutated
            # print('Alpha got updated {}'.format(alpha))
            
        
        # beta regulates the number of genes that are mutated, 1: 100%, 0.01: 1%
        beta *= decay_rate
        number_of_same_individuals = statistics_df['Number of Same Individuals'].iloc[-1]
        if (number_of_same_individuals > n_individuals*0.05): # 5% of the individuals are the same
            # here we want high mutation rate in individuals when not a lot of diversity is reached
            beta = max(min(beta, 0.5),0.25) # here at max 10% or 10 (see aboe) gene from a single individual are mutated  
            # print('Beta got updated {}'.format(beta))


        # gamma is the crossover rate, meaning 2*gamma% of the offsprings are from crossover
        # we want to share the good genes if some are found to be very performant
        gamma *= decay_rate
        if statistics_df['Q3'].iloc[-1]<2*statistics_df['Q1'].iloc[-1]:
            # we want more crossover when "mid" individuals are stuck
            gamma = max(0.25, gamma)
            # print('Gamma got updated {}'.format(gamma))

    return statistics_df

def visualize_path_on_map(individual, cities_df):
    """
    Visualize the path taken by an individual on an interactive map.
    
    Parameters:
    - individual: An instance of the 'individual' class.
    - cities_df: A DataFrame with columns 'city', 'lat', and 'lng'.
    """
    # Get the starting city's coordinates to center the map
    start_lat = cities_df.loc[individual.path[0], 'lat']
    start_lng = cities_df.loc[individual.path[0], 'lng']
    
    # Create a folium map centered on the starting city
    m = folium.Map(location=[start_lat, start_lng], zoom_start=5)
    
    # Add markers for each city in the path
    for city in individual.path:
        lat = cities_df.loc[city, 'lat']
        lng = cities_df.loc[city, 'lng']
        folium.Marker([lat, lng], tooltip=city).add_to(m)
    
    # Add lines connecting the cities in the order they are visited
    path_coords = [(cities_df.loc[city, 'lat'], cities_df.loc[city, 'lng']) for city in individual.path]
    path_coords.append(path_coords[0])  # Close the loop
    folium.PolyLine(path_coords, color="red", weight=2.5, opacity=0.5).add_to(m)
    
    return m

global french_cities_path
french_cities_path = './python_implem/python/worldcities_10k.json'

global output_csv_path
output_csv_path = './python_implem/output.csv'

if __name__ == "__main__":
    print("Clearing output.csv")
    clear_csv_file(output_csv_path)

    # import cities csv
    print('Loading French cities')
    import os
    print("Current Working Directory:", os.getcwd())
    try:
        french_cities = pd.read_json(french_cities_path)
    except ValueError as e:
        print("Error reading CSV file:", e)
    
    # cities df
    C = french_cities[['city_ascii','lat','lng']]
    
    max_cities = 241

    # update D
    print(f'Computing the distance between first {max_cities} French cities:')
    D_spherical_france = compute_spherical_D(french_cities.iloc[:max_cities])
    # pd.DataFrame(D_spherical_france).to_csv('./dsf.csv')
    minimum, maximum = np.min(D_spherical_france), np.max(D_spherical_france)
    D_spherical_france = ( D_spherical_france - minimum) / ( maximum - minimum )
    D_spherical_france = np.array(D_spherical_france).astype(np.float64)

    print(f'Scaling parameters : \n minimum := {minimum} \n maximum := {maximum}.')

    initial_alpha = 1
    initial_gamma = 1
    initial_beta = 0

    start = time.time_ns()
    print('Training iteration {} {} {}...'.format(max_cities, initial_alpha, initial_gamma))

    results = train(
        distance_matrix = D_spherical_france,
        max_cities=max_cities,
        n_individuals=250,
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        initial_gamma=initial_gamma,
        max_iterations=10_000,
        early_stopping_rounds=10_000
        )

    # best_iteration = np.argmin(results['Score'])
    # best_individual_path = results['Best Individual'].iloc[best_iteration]
    # best_individual_path = [int(x) for x in best_individual_path[1:-1]]
    # best_individual = individual(n_cities=max_cities, fixed=best_individual_path)

    print('Saving results')
    best_score = min(results['Score'])
    results.to_csv(f'.python_implem/results/results_{max_cities}_{time.time_ns()-start}_{time.time_ns()}_{best_score}.csv',index=False)