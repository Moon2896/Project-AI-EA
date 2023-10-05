### Imports
### Made by Project made by Baptiste Lesné and Iona Dommel-Prioux

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
import argparse

### class

class individual:
    """
    The 'individual' class represents an individual in a Genetic Algorithm (GA) used for solving the Traveling Salesman Problem (TSP). This class defines the properties and behaviors of an individual, which includes generating random or specified paths, evaluating the fitness of a path, mutating the path, and performing crossover with another individual.

    Attributes:
        - n_cities (int): The number of cities in the TSP.
        - randomize (bool): A flag indicating whether to randomize the initial path.
        - fixed (list): An optional list specifying a fixed path if provided.

    Methods:
        - __init__(self, n_cities=None, randomize=True, fixed=None):
            Initializes an individual with a random or specified path.

        - evaluate(self, D):
            Evaluates the fitness of the current path based on a distance matrix 'D'. It calculates the total distance of the path and stores it.

        - mutate(self):
            Performs a mutation operation on the path by swapping two randomly selected cities.

        - crossover(self, other):
            Performs ordered crossover with another individual 'other' to generate a new individual with a combination of their paths.

        - __str__(self):
            Returns a string representation of the individual's path.

    Usage:
        # Create an individual with a random path of 10 cities
        ind = individual(n_cities=10)

        # Evaluate the fitness of the individual's path using a distance matrix 'D'
        fitness = ind.evaluate(D)

        # Mutate the individual's path
        ind.mutate()

        # Perform crossover with another individual 'other' to create a new individual
        new_individual = ind.crossover(other)

        # Get a string representation of the individual's path
        path_str = str(ind)
    """

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
                np.random.shuffle(self.path)

    def evaluate(self, D):
        # Get the cities and new cities [c_1, ..., c_n], [c_n, c1, ..., c_n-1]
        city_indices = np.array(self.path)
        next_city_indices = np.roll(city_indices, shift=-1)

        # Get the distances from D and add them up
        distances = D[city_indices, next_city_indices]
        total_distance = np.sum(distances)
        self.total_distance = total_distance

        return total_distance


    def mutate(self):
        i, j = random.choices(range(self.n), k=2)
        # Swap c_i and c_j
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
        # use for quick print
        res = str(self.path)
        return res

### Viz

def visualize_paths_grid(individuals, cities_coordinates, distance_matrix, grid_shape=None):
    """
    The 'visualize_paths_grid' function is used to visualize multiple paths of individuals in a grid of plots for the Traveling Salesman Problem (TSP). It provides a visual representation of the paths on a scatter plot with cities' coordinates and their connections.

    Parameters:
        - individuals (list): A list of instances of the 'individual' class representing different paths to visualize.
        - cities_coordinates (list): A list of (x, y) coordinates for each city in the TSP.
        - distance_matrix (numpy.ndarray): A matrix of distances between cities.
        - grid_shape (tuple, optional): Tuple specifying the shape of the grid (rows, columns) for the subplots. If None, a square grid is automatically determined based on the number of individuals.

    Usage:
        - Provide a list of 'individual' instances, each representing a different path.
        - Specify the coordinates of the cities as a list of (x, y) tuples.
        - Pass the distance matrix, which is used to evaluate the fitness of the individuals.
        - Optionally, specify the grid shape for arranging the subplots. If not provided, the function will create a square grid based on the number of individuals.

    Example:
        # Create a list of 'individual' instances
        individuals = [individual(n_cities=10), individual(n_cities=10), individual(n_cities=10)]

        # Specify the coordinates of cities
        cities_coordinates = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

        # Create a distance matrix for the cities

        # Visualize the paths of individuals in a grid
        visualize_paths_grid(individuals, cities_coordinates, distance_matrix)

    The function will create a grid of plots where each subplot shows the cities as blue points and the path of an individual in red. The title of each subplot includes the path number and its corresponding distance.
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

        # Plot the cities
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

    # Remove any unused subplots (n*m<len(individuals))
    for i in range(n_individuals, grid_shape[0] * grid_shape[1]):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_path_cities(individual, cities_df, distance_matrix):
    """
    The 'visualize_path_cities' function is used to visualize the path taken by an individual in the Traveling Salesman Problem (TSP). It displays a scatter plot of cities along with their names and the path taken by the individual to connect those cities.

    Parameters:
        - individual (individual): An instance of the 'individual' class representing the path to visualize.
        - cities_df (pandas.DataFrame): A DataFrame containing information about cities, including 'city', 'lat', and 'lng'.
        - distance_matrix (numpy.ndarray): A matrix of distances between cities.

    Usage:
        - Provide an 'individual' instance representing the path to visualize.
        - Specify a pandas DataFrame ('cities_df') with columns 'city', 'lat', and 'lng' containing information about the cities.
        - Pass the distance matrix, which is used to evaluate the fitness of the individual's path.

    Example:
        # Create an 'individual' instance representing a path
        ind = individual(n_cities=10)

        # Create a DataFrame with city information
        cities_df = pd.DataFrame({
            'city': range(10),
            'lat': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'lng': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        })

        # Visualize the path taken by the individual
        visualize_path_cities(ind, cities_df, distance_matrix)

    The function will create a scatter plot showing the cities as blue points, with their names labeled. The path taken by the individual is represented by a red line, and the plot includes the distance of the path in the title.
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
    """
    The 'compute_fitness' function calculates the fitness of an individual based on the inverse of the total distance it covers in a Traveling Salesman Problem (TSP).

    Parameters:
        - individual (individual): An instance of the 'individual' class representing the path to evaluate.
        - distance_matrix (numpy.ndarray): A matrix of distances between cities.

    Returns:
        - fitness (float): The fitness value of the individual, which is the inverse of the total distance.

    Usage:
        - Provide an 'individual' instance representing the path to evaluate.
        - Pass the distance matrix, which is used to evaluate the fitness.

    Example:
        # Calculate the fitness of an individual
        fitness = compute_fitness(ind, distance_matrix)
    """
    return 1 / individual.evaluate(distance_matrix)

# import multiprocessing as mp

def epoch(individuals, distance_matrix, alpha=0.5, beta=0.5, gamma=0.5):
    """
    The 'epoch' function performs one iteration of a genetic algorithm (GA) epoch, including selection, crossover, mutation, and replacement operations.

    Parameters:
        - individuals (list): A list of instances of the 'individual' class representing the population of individuals.
        - distance_matrix (numpy.ndarray): A matrix of distances between cities.
        - alpha (float): The mutation rate, representing the proportion of individuals to mutate.
        - beta (float): The mutation strength, representing the proportion of genes to mutate within an individual.
        - gamma (float): The crossover rate, representing the proportion of individuals to produce offspring.

    Returns:
        - individuals (list): The updated list of individuals after performing selection, crossover, mutation, and replacement.

    Usage:
        - Provide a list of 'individual' instances representing the population.
        - Pass the distance matrix, which is used for fitness evaluation.
        - Optionally, set alpha, beta, and gamma to control the mutation rate, mutation strength, and crossover rate, respectively.

    Example:
        # Perform one epoch of the genetic algorithm
        new_population = epoch(population, distance_matrix, alpha=0.5, beta=0.5, gamma=0.5)

    The function modifies the 'individuals' list according to the specified genetic algorithm operations.
    """
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
    """
    The 'update_statistics' function updates statistics and logs information about the progress of a genetic algorithm (GA) iteration for solving the Traveling Salesman Problem (TSP).

    Parameters:
        - df (pandas.DataFrame): A DataFrame for storing and tracking statistics across GA iterations.
        - distance_matrix (numpy.ndarray): A matrix of distances between cities.
        - iteration (int): The current iteration step.
        - individuals (list): A list of instances of the 'individual' class representing the population.
        - alpha (float): The mutation rate used in the GA.
        - beta (float): The mutation strength used in the GA.
        - gamma (float): The crossover rate used in the GA.

    Returns:
        - df (pandas.DataFrame): The updated DataFrame with statistics.

    Usage:
        - Provide a pandas DataFrame ('df') for storing statistics across iterations.
        - Pass the distance matrix, which is used for evaluating the individuals' fitness.
        - Specify the current iteration step.
        - Provide a list of 'individual' instances representing the population at the current iteration.
        - Set values for alpha, beta, and gamma, which are used in the statistics.

    Example:
        # Initialize a DataFrame for tracking statistics
        statistics_df = pd.DataFrame(columns=['Iteration', 'Number of Individuals', 'Best Individual', 'All Scores', 'Number of Same Individuals', 'Number of Shared Patterns', 'Score', 'alpha', 'beta', 'gamma', 'Median', 'Q1', 'Q3', 'Max'])

        # Update statistics after an iteration of the GA
        statistics_df = update_statistics(statistics_df, distance_matrix, iteration, population, alpha=0.5, beta=0.5, gamma=0.5)

    The function calculates and logs various statistics for a GA iteration, including the best individual's path, the number of same individuals, the number of shared patterns, and various statistics about the fitness scores of the population. It appends this information to the provided DataFrame and saves it to a CSV file.
    """
    # Iteration step
    step = iteration

    # Best individual
    scores = [ind.evaluate(distance_matrix) for ind in individuals]
    best_individual = min(individuals, key=lambda x: x.evaluate(distance_matrix))

    # Number of same individuals
    unique_individuals = set(tuple(ind.path) for ind in individuals)
    num_same_individuals = len(individuals) - len(unique_individuals)

    # Number of shared patterns between individuals
    # Here, we'll count shared subpaths of length 5
    subpathlength = 5
    patterns = []
    for ind in individuals:
        for i in range(len(ind.path) - subpathlength+1):
            patterns.append(tuple(ind.path[i:i+subpathlength]))
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
    """
    The 'append_to_csv' function appends data to a CSV (Comma-Separated Values) file. It is used to log information or records into an existing CSV file.

    Parameters:
        - filename (str): The name of the CSV file to which data will be appended.
        - data (dict): A dictionary containing the data to append to the CSV file. The keys of the dictionary represent the column names, and the values represent the data for each column.

    Usage:
        - Provide the 'filename' of the CSV file to which you want to append data.
        - Pass a 'data' dictionary containing the data to append, where keys are column names and values are the corresponding data for each column.

    Example:
        # Define data to append to a CSV file
        data_to_append = {
            ... some data ...
        }

        # Append the data to the CSV file
        append_to_csv('output.csv', data_to_append)

    The function opens the specified CSV file in append mode, writes the data, and handles the creation of the header row if the file is empty. It uses a dictionary to map column names to data values and writes them as rows in the CSV file.
    """
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

    except Exception as e:
        print("Error:", e)

def clear_csv_file(filename):
    """
    The 'clear_csv_file' function clears the contents of a CSV (Comma-Separated Values) file by opening it in write mode. It effectively empties the file, removing all its data.

    Parameters:
        - filename (str): The name of the CSV file to clear.

    Usage:
        - Provide the 'filename' of the CSV file that you want to clear.

    Example:
        # Clear the contents of a CSV file named 'example.csv'
        clear_csv_file('example.csv')

    The function opens the specified CSV file in write mode, which clears its contents by effectively overwriting it with an empty file. This can be useful when you want to reset or clear the data in an existing CSV file.
    """
    try:
        # Open the CSV file in write mode to clear its contents
        with open(filename, mode='w', newline=''):
            pass  # The 'pass' statement does nothing, effectively clearing the file

        print("Cleared", filename)
    except Exception as e:
        print("Error:", e)

def compute_spherical_D(df):
    """
    The 'compute_spherical_D' function calculates the spherical distance matrix between a set of geographic coordinates represented as latitude and longitude values. It uses the Haversine formula to compute distances on the surface of a sphere, typically representing Earth's surface.

    Parameters:
        - df (pandas.DataFrame): A DataFrame containing geographic coordinates with columns 'lat' (latitude) and 'lng' (longitude).

    Returns:
        - D (numpy.ndarray): A symmetric matrix representing the spherical distances between the coordinates in the DataFrame. The matrix is in meters.

    Usage:
        - Provide a DataFrame ('df') with latitude and longitude columns for the geographic coordinates.

    Example:
        # Create a DataFrame with latitude and longitude columns
        coordinates_df = pd.DataFrame({
            'lat': [51.5074, 48.8566, 40.7128],
            'lng': [-0.1278, 2.3522, -74.0060]
        })

        # Compute the spherical distance matrix
        distance_matrix = compute_spherical_D(coordinates_df)

    The function iterates through all pairs of coordinates in the DataFrame and calculates the spherical distances between them using the Haversine formula. The resulting distance matrix ('D') is symmetric and represents the distances in meters between each pair of coordinates.
    """
    n = df.shape[0]
    r = 6371e3 # earth's radius
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
    """
    The 'train' function is used to perform the training of a genetic algorithm (GA) for solving the Traveling Salesman Problem (TSP). It trains the GA using the provided distance matrix, various hyperparameters, and tracks statistics over multiple iterations.

    Parameters:
        - distance_matrix (numpy.ndarray): A matrix of distances between cities, used for evaluating the fitness of individuals.
        - max_cities (int): The maximum number of cities to consider in the TSP (to limit computation).
        - n_individuals (int): The number of individuals (paths) in the population.
        - initial_alpha (float): The initial mutation rate for individuals.
        - initial_beta (float): The initial mutation strength (percentage of genes to mutate within an individual).
        - initial_gamma (float): The initial crossover rate (percentage of offspring produced through crossover).
        - max_iterations (int, optional): The maximum number of iterations for the GA.
        - early_stopping_rounds (int, optional): The number of consecutive iterations with no improvement to trigger early stopping.

    Returns:
        - statistics_df (pandas.DataFrame): A DataFrame containing statistics tracked during the training process, such as best individual, number of same individuals, quartiles, and more.

    Usage:
        - Provide a distance matrix ('distance_matrix') for evaluating fitness.
        - Specify the maximum number of cities to consider ('max_cities').
        - Set the number of individuals in the population ('n_individuals').
        - Initialize mutation rate ('initial_alpha'), mutation strength ('initial_beta'), and crossover rate ('initial_gamma').
        - Optionally, set the maximum number of iterations and early stopping rounds.

    Example:
        # Train a GA for TSP
        stats = train(distance_matrix, max_cities=10, n_individuals=50, initial_alpha=0.5, initial_beta=0.5, initial_gamma=0.5)

    The function trains a GA by repeatedly applying the 'epoch' function, tracking statistics, and adjusting hyperparameters over iterations.
    Early stopping is triggered if no improvement is observed over a specified number of rounds. Hyperparameters alpha, beta, and gamma are adjusted based on the convergence and diversity of the population.
    """
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
    The 'visualize_path_on_map' function is used to visualize the path taken by an individual on an interactive map.
    It displays markers for cities and connects them in the order they are visited by the individual, creating a visual representation of the Traveling Salesman Problem (TSP) route.

    Parameters:
        - individual (individual): An instance of the 'individual' class representing the path to visualize.
        - cities_df (pandas.DataFrame): A DataFrame with columns 'city', 'lat' (latitude), and 'lng' (longitude) containing information about the cities.

    Returns:
        - m (folium.Map): An interactive folium map with markers for each city and lines connecting them in the order they are visited.

    Usage:
        - Provide an 'individual' instance representing the path to visualize.
        - Pass a pandas DataFrame ('cities_df') with city information, including latitude and longitude.

    Example:
        # Create an 'individual' instance representing a path
        ind = individual(n_cities=10)

        # Create a DataFrame with city information
        cities_df = pd.DataFrame({
            'city': range(10),
            'lat': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'lng': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        })

        # Visualize the path on an interactive map
        map = visualize_path_on_map(ind, cities_df)
        map.save('path_map.html')

    The function uses the Folium library to create an interactive map, adding markers for each city and lines connecting them to represent the TSP route taken by the individual.
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

# global french_cities_path
# french_cities_path = './python_implem/python/worldcities_10k.json'

# global output_csv_path
# output_csv_path = './python_implem/output.csv'



if __name__ == "__main__":
    """
    The main script for solving the Traveling Salesman Problem (TSP) using a genetic algorithm with French cities data.
    It performs various tasks such as data loading, distance computation, training, and result saving.

    Usage:
        - Make sure you have the required libraries installed and the necessary data files (e.g., 'french_cities.json') available.
        - Customize the script by adjusting parameters, such as 'max_cities,' 'initial_alpha,' 'initial_gamma,' 'initial_beta,' 'max_iterations,' and 'early_stopping_rounds.'
        - Run the script to execute the TSP solving process and save the results.

    Example:
        # Customize parameters
        max_cities = 241
        initial_alpha = 1
        initial_gamma = 1
        initial_beta = 0
        max_iterations = 10_000
        early_stopping_rounds = 10_000

        # Run the script
        python main_script.py

    The script first clears the output CSV file and then proceeds with loading the French cities data, computing distances, and training the genetic algorithm.
    The results, including statistics and the best individual's path, are saved to a CSV file.
    """

    parser = argparse.ArgumentParser(description="Solving the Traveling Salesman Problem using a genetic algorithm with French cities data.")

    # Add command-line arguments
    parser.add_argument("--input_cities_path", "-i", type=str, default='./datasets/worldcities_10k.json', help="Path to the Input cities data file.", )
    parser.add_argument("--output_path", "-o", type=str, default='./output.csv', help="Path to the output csv to use.")
    parser.add_argument("--result_folder", "-r", type=str, default='./results', help="Path to the result folder to use.")
    parser.add_argument("--max_cities", type=int, default=100, help="Maximum number of cities to consider.")
    parser.add_argument("--initial_alpha", type=float, default=1, help="Initial value for alpha.")
    parser.add_argument("--initial_gamma", type=float, default=1, help="Initial value for gamma.")
    parser.add_argument("--initial_beta", type=float, default=0, help="Initial value for beta.")
    parser.add_argument("--max_iterations", type=int, default=10_000, help="Maximum number of training iterations.")
    parser.add_argument("--early_stopping_rounds", type=int, default=250, help="Number of iterations for early stopping.")

    # Parse the arguments
    args = parser.parse_args()

    # Now, instead of hardcoding values, use the parsed arguments
    input_cities_path = args.input_cities_path
    output_csv_path = args.output_path
    result_folder = args.result_folder
    max_cities = args.max_cities
    initial_alpha = args.initial_alpha
    initial_gamma = args.initial_gamma
    initial_beta = args.initial_beta
    max_iterations = args.max_iterations
    early_stopping_rounds = args.early_stopping_rounds

    print("Clearing output.csv")
    clear_csv_file(output_csv_path)

    # import cities csv
    print('Loading French cities')
    import os
    print("Current Working Directory:", os.getcwd())
    try:
        input_cities = pd.read_json(input_cities_path)
    except ValueError as e:
        print("Error reading CSV file:", e)

    # cities df
    C = input_cities[['city_ascii','lat','lng']]


    # update D
    print(f'Computing the distance between first {max_cities} French cities:')
    D_spherical_matrix = compute_spherical_D(input_cities.iloc[:max_cities])
    # pd.DataFrame(D_spherical_matrix).to_csv('./dsf.csv')
    minimum, maximum = np.min(D_spherical_matrix), np.max(D_spherical_matrix)
    D_spherical_matrix = ( D_spherical_matrix - minimum) / ( maximum - minimum )
    D_spherical_matrix = np.array(D_spherical_matrix).astype(np.float64)

    print(f'Scaling parameters : \n minimum := {minimum} \n maximum := {maximum}.')

    initial_alpha = 1
    initial_gamma = 1
    initial_beta = 0

    start = time.time_ns()
    print('Training iteration {} {} {}...'.format(max_cities, initial_alpha, initial_gamma))

    results = train(
        distance_matrix = D_spherical_matrix,
        max_cities=max_cities,
        n_individuals=1_000,
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        initial_gamma=initial_gamma,
        max_iterations=10_000,
        early_stopping_rounds=250
        )

    # best_iteration = np.argmin(results['Score'])
    # best_individual_path = results['Best Individual'].iloc[best_iteration]
    # best_individual_path = [int(x) for x in best_individual_path[1:-1]]
    # best_individual = individual(n_cities=max_cities, fixed=best_individual_path)

    print('Saving results')
    best_score = min(results['Score'])
    results.to_csv(f'{result_folder}/results_{max_cities}_{time.time_ns()-start}_{time.time_ns()}_{best_score}.csv',index=False)
