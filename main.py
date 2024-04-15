# Ro’a Nafi , 1201959.
# Dunia Al’amal Hamada, 1201001

import random
import copy
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ----------------------------------------------------------------------------------------#
# ------------------------------------interfaces-------------------------------------------#
# ----------------------------------------------------------------------------------------#

def interface1(solution, algo_select, genetic_final_total_distance, csp_final_total_distance):
    plt.figure(figsize=(8, 4))
    colors = ['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta']

    # Plot packages
    for vehicle in solution:
        destinations = [pkg['destination'] for pkg in vehicle['packages']]
        if not destinations:  # skip if no packages for this vehicle
            continue
        x, y = zip(*destinations)
        plt.scatter(x, y, s=50, label=f"Vehicle {vehicle['id']}")
        x, y = [shop_location[0]] + list(x) + [shop_location[0]], [shop_location[1]] + list(y) + [shop_location[1]]
        plt.plot(x, y, color=colors[vehicle['id'] % len(colors)])

    plt.scatter(shop_location[0], shop_location[1], s=150, c='black', marker='X', label='Shop Location')

    if algo_select == 1:
        title = "Genetic Algorithm"
        total_distance = genetic_final_total_distance
    else:
        title = "SCP Algorithm"
        total_distance = csp_final_total_distance

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Print Genetic Total Distance below the x-coordinate axis
    plt.text(0.5, -0.15, f"                            Total Distance: {total_distance:.2f}",
             transform=plt.gca().transAxes, ha="left", fontsize=10)

    plt.tight_layout()  # Adjust subplot spacing
    plt.show()


def interface2(csp_solution, genetic_solution, shop_location, csp_distance, genetic_distance):
    plt.figure(figsize=(12, 6))  # Smaller figure size

    # Create a grid with 2 rows and 2 columns for plots and distances
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])

    # Plot CSP solution
    plt.subplot(gs[0, 0])  # Create the first subplot for CSP solution
    colors = ['red', 'blue', 'green', 'magenta', 'purple', 'yellow', 'cyan']
    for vehicle in csp_solution:
        destinations = [pkg['destination'] for pkg in vehicle['packages']]
        if not destinations:
            continue
        x, y = zip(*destinations)
        plt.scatter(x, y, s=30, label=f"CSP - Vehicle {vehicle['id']}")
        x, y = [shop_location[0]] + list(x) + [shop_location[0]], [shop_location[1]] + list(y) + [shop_location[1]]
        plt.plot(x, y, color=colors[vehicle['id'] % len(colors)], linestyle='--')

    plt.scatter(shop_location[0], shop_location[1], s=80, c='black', marker='X', label='Shop Location')
    plt.title("CSP Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Plot genetic solution
    plt.subplot(gs[0, 1])  # Create the second subplot for genetic solution
    for vehicle in genetic_solution:
        destinations = [pkg['destination'] for pkg in vehicle['packages']]
        if not destinations:
            continue
        x, y = zip(*destinations)
        plt.scatter(x, y, s=30, label=f"Genetic - Vehicle {vehicle['id']}")
        x, y = [shop_location[0]] + list(x) + [shop_location[0]], [shop_location[1]] + list(y) + [shop_location[1]]
        plt.plot(x, y, color=colors[vehicle['id'] % len(colors)], linestyle='--')

    plt.scatter(shop_location[0], shop_location[1], s=80, c='black', marker='X', label='Shop Location')
    plt.title("Genetic Algorithm Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    # Print CSP distance on the left
    plt.subplot(gs[1, 0])  # Create the third subplot for CSP distance
    plt.axis('off')  # Turn off axis for this subplot
    plt.text(0.5, 0.5, f"CSP Total Distance: {csp_distance:.2f}", ha="center", fontsize=10)

    # Print genetic algorithm distance on the right
    plt.subplot(gs[1, 1])  # Create the fourth subplot for genetic distance
    plt.axis('off')  # Turn off axis for this subplot
    plt.text(0.5, 0.5, f"Genetic Total Distance: {genetic_distance:.2f}", ha="center", fontsize=10)

    plt.tight_layout(w_pad=3)  # Ensure subplots do not overlap and add padding between them
    plt.show()


def calc_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def calculate_total_distance(vehicle, shop_location):
    distance = 0
    last_location = shop_location
    for package in vehicle['packages']:
        distance += calc_distance(last_location, package['destination'])
        last_location = package['destination']
    distance += calc_distance(last_location, shop_location)
    return distance


# --------------------------------------------------------------------------------------------------------------#
# -----------------------------------genetic_algorithm implementation-------------------------------------------#
# --------------------------------------------------------------------------------------------------------------#

def initialize_population(packages, vehicle_count, vehicle_capacity, population_size):
    population = []

    for _ in range(population_size):
        random.shuffle(packages)
        vehicles = [{'id': i + 1, 'packages': [], 'remaining_capacity': vehicle_capacity} for i in range(vehicle_count)]
        for pkg in packages:
            for vehicle in vehicles:
                if vehicle['remaining_capacity'] >= pkg['weight']:
                    vehicle['packages'].append(pkg)
                    vehicle['remaining_capacity'] -= pkg['weight']

                    break
        population.append(vehicles)

    return population


def select_parents(population, tournament_size=10):
    # first parent
    parent1 = min(population, key=lambda v: sum(calculate_total_distance(vehicle, shop_location) for vehicle in v))
    # second parent
    parent2 = min(population, key=lambda v: sum(calculate_total_distance(vehicle, shop_location) for vehicle in v))

    while parent1 != parent2:
        tournament2 = random.sample(population, tournament_size)
        parent2 = min(tournament2, key=lambda v: sum(calculate_total_distance(vehicle, shop_location) for vehicle in v))

    return parent1, parent2


def crossover(parent1, parent2):
    # Choose two crossover points
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    # Copy segments from parents
    child1 = [-1] * size
    child2 = [-1] * size
    child1[start:end] = copy.deepcopy(parent1[start:end])
    child2[start:end] = copy.deepcopy(parent2[start:end])

    # Fill the remaining spots
    fill_remaining(child1, parent2)
    fill_remaining(child2, parent1)

    return child1, child2


def fill_remaining(child, parent):
    size = len(child)
    unused_vehicles = [v for v in parent if v not in child]

    for i in range(size):
        if child[i] == -1 and unused_vehicles:
            child[i] = unused_vehicles.pop(0)


def mutate(child):
    if len(child) < 2:  # can not swap if there is 1 vehicle
        return

    vehicle1, vehicle2 = random.sample(child, 2)

    if not vehicle1['packages'] or not vehicle2['packages']:  # an empty vehicle
        return

    pkg1 = random.choice(vehicle1['packages'])
    pkg2 = random.choice(vehicle2['packages'])

    if pkg1['weight'] <= vehicle2['remaining_capacity'] and pkg2['weight'] <= vehicle1['remaining_capacity']:
        vehicle1['packages'].remove(pkg1)
        vehicle1['packages'].append(pkg2)
        vehicle2['packages'].remove(pkg2)
        vehicle2['packages'].append(pkg1)

        vehicle1['remaining_capacity'] += pkg1['weight'] - pkg2['weight']
        vehicle2['remaining_capacity'] += pkg2['weight'] - pkg1['weight']


def genetic_algorithm(packages, vehicle_count, vehicle_capacity, iterations=50, population_size=20):
    population = initialize_population(packages, vehicle_count, vehicle_capacity, population_size)

    for _ in range(iterations):

        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.append(child1)
            new_population.append(child2)

        population = new_population
        # print("--------------------------\n")
        # print(new_population)

    best_solution = min(population,
                        key=lambda v: sum(calculate_total_distance(vehicle, shop_location) for vehicle in v))

    return best_solution


# --------------------------------------------------------------------------------------------------------------#
# ---------------------------------------CSP algorithm implementation-------------------------------------------#
# --------------------------------------------------------------------------------------------------------------#

def csp(packages, vehicle_count, vehicle_capacity, shop_location):
    vehicles = [{'id': i + 1, 'packages': [], 'remaining_capacity': vehicle_capacity} for i in range(vehicle_count)]
    best_solution = None
    best_distance = float('inf')

    def backtrack(pkg_idx=0):
        nonlocal best_solution, best_distance
        if pkg_idx == len(packages):
            curr_distance = sum(calculate_total_distance(vehicle, shop_location) for vehicle in vehicles)
            if curr_distance < best_distance:
                best_solution = copy.deepcopy(vehicles)
                best_distance = curr_distance
            return

        pkg = packages[pkg_idx]
        for vehicle in vehicles:
            if vehicle['remaining_capacity'] >= pkg['weight']:
                vehicle['packages'].append(pkg)
                vehicle['remaining_capacity'] -= pkg['weight']
                backtrack(pkg_idx + 1)
                vehicle['packages'].remove(pkg)
                vehicle['remaining_capacity'] += pkg['weight']

    backtrack()
    return best_solution


def menu():
    print("************Welcome to package delivery shop**************")
    print()

    choice = input("""
             A: Genetic algorithms
             B: Backtracking search 
             C: Both genetic and backtracking
             Q: Exit

                 Please enter your choice: """)

    if choice == "A" or choice == "a":
        return 1
    elif choice == "B" or choice == "b":
        return 2
    elif choice == "C" or choice == "c":
        return 3
    elif choice == "Q" or choice == "q":
        sys.exit
    else:
        print("OoOps.. invalid choice....")
        print("Please try again")
        menu()


if __name__ == '__main__':
    packages = [
        {'id': 1, 'destination': (4, 5), 'weight': 5},
        {'id': 2, 'destination': (-2, -2), 'weight': 50},
        {'id': 3, 'destination': (5, -1), 'weight': 30},
        {'id': 4, 'destination': (-4, -1), 'weight': 10},
        {'id': 5, 'destination': (1, -1), 'weight': 50},
        {'id': 6, 'destination': (-2, 2), 'weight': 7}
    ]

    vehicle_count = 5
    vehicle_capacity = 50
    shop_location = (0, 0)

    algo_select = menu()

    if algo_select == 1:
        print("Vehicles Path's using genetic_algorithm \n")
        genetic_solution = genetic_algorithm(packages, vehicle_count, vehicle_capacity)
        for vehicle in genetic_solution:
            print(f"Vehicle {vehicle['id']} Path: {[pkg['destination'] for pkg in vehicle['packages']]}")
        genetic_final_total_distance = sum(
            calculate_total_distance(vehicle, shop_location) for vehicle in genetic_solution)
        print(f"Total Distance:{genetic_final_total_distance}")
        interface1(genetic_solution, algo_select, genetic_final_total_distance, 0)


    elif algo_select == 2:

        print("Vehicles Path's using CSP \n")
        csp_solution = csp(packages, vehicle_count, vehicle_capacity, shop_location)
        for vehicle in csp_solution:
            print(f"Vehicle {vehicle['id']} Path: {[pkg['destination'] for pkg in vehicle['packages']]}")
        csp_final_total_distance = sum(calculate_total_distance(vehicle, shop_location) for vehicle in csp_solution)
        print(f"Total Distance:{csp_final_total_distance}")
        interface1(csp_solution, algo_select, 0, csp_final_total_distance)


    elif algo_select == 3:
        print("Vehicles Path's using genetic_algorithm \n")
        genetic_solution = genetic_algorithm(packages, vehicle_count, vehicle_capacity)
        for vehicle in genetic_solution:
            print(f"Vehicle {vehicle['id']} Path: {[pkg['destination'] for pkg in vehicle['packages']]}")
        genetic_final_total_distance = sum(
            calculate_total_distance(vehicle, shop_location) for vehicle in genetic_solution)
        print(f"Total Distance:{genetic_final_total_distance}")

        print("\nVehicles Path's using CSP \n")
        csp_solution = csp(packages, vehicle_count, vehicle_capacity, shop_location)
        for vehicle in csp_solution:
            print(f"Vehicle {vehicle['id']} Path: {[pkg['destination'] for pkg in vehicle['packages']]}")
        csp_final_total_distance = sum(calculate_total_distance(vehicle, shop_location) for vehicle in csp_solution)
        print(f"Total Distance:{csp_final_total_distance}")

        interface2(csp_solution, genetic_solution, shop_location, csp_final_total_distance,
                   genetic_final_total_distance)
