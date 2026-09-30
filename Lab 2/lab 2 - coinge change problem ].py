import random
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Coin Change Problem using Genetic Algorithm
# ---------------------------------------------------------

# Coin denominations
COINS = [1, 5, 10, 25]

# Target amount
TARGET = 63

# Number of different coins
NUM_COINS = len(COINS)

# GA parameters
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.80


# ---------------------------------------------------------
# Create an Individual
# ---------------------------------------------------------

def create_individual():

    # Maximum possible number of each coin
    individual = []

    for coin in COINS:

        max_count = TARGET // coin

        individual.append(
            random.randint(0, max_count)
        )

    return individual


# ---------------------------------------------------------
# Create Population
# ---------------------------------------------------------

def create_population():

    return [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]


# ---------------------------------------------------------
# Calculate Total Value
# ---------------------------------------------------------

def calculate_value(individual):

    return sum(
        individual[i] * COINS[i]
        for i in range(NUM_COINS)
    )


# ---------------------------------------------------------
# Calculate Number of Coins
# ---------------------------------------------------------

def calculate_coin_count(individual):

    return sum(individual)


# ---------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------

def fitness(individual):

    value = calculate_value(individual)
    coin_count = calculate_coin_count(individual)

    # Difference from target
    difference = abs(TARGET - value)

    # Exact solutions should have very high fitness
    if difference == 0:

        # Fewer coins = better fitness
        return 100000 - coin_count

    else:

        # Penalize incorrect amounts
        return 10000 - (difference * 100) - coin_count


# ---------------------------------------------------------
# Selection - Tournament Selection
# ---------------------------------------------------------

def selection(population):

    tournament_size = 3

    participants = random.sample(
        population,
        tournament_size
    )

    winner = max(
        participants,
        key=fitness
    )

    return winner.copy()


# ---------------------------------------------------------
# Crossover
# ---------------------------------------------------------

def crossover(parent1, parent2):

    point = random.randint(
        1,
        NUM_COINS - 1
    )

    child1 = (
        parent1[:point] +
        parent2[point:]
    )

    child2 = (
        parent2[:point] +
        parent1[point:]
    )

    return child1, child2


# ---------------------------------------------------------
# Mutation
# ---------------------------------------------------------

def mutation(individual):

    individual = individual.copy()

    for i in range(NUM_COINS):

        if random.random() < MUTATION_RATE:

            max_count = TARGET // COINS[i]

            # Randomly change coin count
            individual[i] = random.randint(
                0,
                max_count
            )

    return individual


# ---------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------

def genetic_algorithm():

    population = create_population()

    best_solution = None
    best_fitness = -float("inf")

    history = []

    for generation in range(GENERATIONS):

        # Calculate fitness
        fitness_values = [
            fitness(individual)
            for individual in population
        ]

        # Find best individual
        best_index = np.argmax(fitness_values)

        generation_best = population[best_index]
        generation_best_fitness = fitness_values[best_index]

        # Update global best
        if generation_best_fitness > best_fitness:

            best_fitness = generation_best_fitness
            best_solution = generation_best.copy()

        history.append(best_fitness)

        # New generation
        new_population = []

        # Elitism
        new_population.append(
            best_solution.copy()
        )

        while len(new_population) < POPULATION_SIZE:

            # Selection
            parent1 = selection(population)
            parent2 = selection(population)

            # Crossover
            if random.random() < CROSSOVER_RATE:

                child1, child2 = crossover(
                    parent1,
                    parent2
                )

            else:

                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:

                new_population.append(child2)

        population = new_population

    return best_solution, best_fitness, history


# ---------------------------------------------------------
# Run GA
# ---------------------------------------------------------

best_solution, best_fitness, history = genetic_algorithm()


# ---------------------------------------------------------
# Display Result
# ---------------------------------------------------------

total_value = calculate_value(best_solution)
total_coins = calculate_coin_count(best_solution)

print("\nOptimal Coin Change Solution")
print("--------------------------------")

for i in range(NUM_COINS):

    print(
        f"Coin {COINS[i]}: "
        f"{best_solution[i]} coins"
    )

print("--------------------------------")

print(f"Target amount : {TARGET}")
print(f"Total amount  : {total_value}")
print(f"Total coins   : {total_coins}")

if total_value == TARGET:
    print("Status        : Exact solution")
else:
    print("Status        : Approximate solution")


# ---------------------------------------------------------
# Plot GA Convergence
# ---------------------------------------------------------

plt.plot(history)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Genetic Algorithm - Coin Change Optimization")

plt.grid(True)
plt.show()
