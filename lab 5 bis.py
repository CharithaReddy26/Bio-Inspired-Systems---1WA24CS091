
import numpy as np
import math

# ==========================================
# CUCKOO SEARCH FOR RESOURCE ALLOCATION
# ==========================================

# Problem parameters
NUM_USERS = 3
NUM_RESOURCES = 3

# Total available resources
# CPU, Memory, Bandwidth
TOTAL_RESOURCES = np.array([100, 200, 100], dtype=float)

# Minimum resource requirements
REQUIREMENTS = np.array([
    [20, 40, 10],
    [25, 50, 20],
    [15, 30, 15]
], dtype=float)

# Cuckoo Search parameters
NESTS = 25
PA = 0.25
MAX_ITER = 200
ALPHA = 0.01
BETA = 1.5


# ==========================================
# 1. INITIALIZE POPULATION
# ==========================================

def initialize_nests():

    nests = []

    upper_limits = np.maximum(
        REQUIREMENTS,
        TOTAL_RESOURCES / NUM_USERS
    )

    for i in range(NESTS):

        solution = np.random.uniform(
            REQUIREMENTS,
            upper_limits
        )

        solution = repair_solution(solution)

        nests.append(solution)

    return np.array(nests)


# ==========================================
# 2. REPAIR RESOURCE ALLOCATION
# ==========================================

def repair_solution(solution):

    solution = np.maximum(
        solution,
        REQUIREMENTS
    )

    for j in range(NUM_RESOURCES):

        total = np.sum(solution[:, j])

        if total > TOTAL_RESOURCES[j]:

            minimum = np.sum(REQUIREMENTS[:, j])

            available = TOTAL_RESOURCES[j] - minimum

            extra = solution[:, j] - REQUIREMENTS[:, j]

            if np.sum(extra) > 0:

                solution[:, j] = (
                    REQUIREMENTS[:, j]
                    + extra * available / np.sum(extra)
                )

    return solution


# ==========================================
# 3. FITNESS FUNCTION
# ==========================================

def fitness(solution):

    solution = repair_solution(solution)

    utilization = (
        np.sum(solution, axis=0) / TOTAL_RESOURCES
    )

    # Calculate resource wastage
    wastage = np.sum(1 - utilization)

    # Calculate load imbalance
    imbalance = np.std(utilization)

    # Calculate allocation cost
    cost = np.sum(solution / TOTAL_RESOURCES)

    # Objective function
    score = (
        0.50 * wastage
        + 0.30 * imbalance
        + 0.20 * cost
    )

    return score


# ==========================================
# 4. LEVY FLIGHT
# ==========================================

def levy_flight():

    sigma = (
        math.gamma(1 + BETA)
        * math.sin(math.pi * BETA / 2)
        /
        (
            math.gamma((1 + BETA) / 2)
            * BETA
            * 2 ** ((BETA - 1) / 2)
        )
    ) ** (1 / BETA)

    u = np.random.normal(0, sigma)

    v = np.random.normal(0, 1)

    step = u / (
        abs(v) ** (1 / BETA) + 1e-10
    )

    return step


# ==========================================
# 5. CUCKOO SEARCH ALGORITHM
# ==========================================

def cuckoo_search():

    # Initialize nests
    nests = initialize_nests()

    # Evaluate fitness
    fitness_values = np.array([
        fitness(nest) for nest in nests
    ])

    # Find best nest
    best_index = np.argmin(fitness_values)

    best_solution = nests[best_index].copy()

    best_fitness = fitness_values[best_index]

    history = []

    # Iteration process
    for iteration in range(MAX_ITER):

        # Generate new solutions using Levy flight
        for i in range(NESTS):

            step = levy_flight()

            new_solution = (
                nests[i]
                + ALPHA * step
                * np.random.randn(
                    NUM_USERS,
                    NUM_RESOURCES
                )
            )

            # Repair constraints
            new_solution = repair_solution(
                new_solution
            )

            # Calculate fitness
            new_fitness = fitness(new_solution)

            # Select random nest
            j = np.random.randint(NESTS)

            # Replace if better
            if new_fitness < fitness_values[j]:

                nests[j] = new_solution.copy()

                fitness_values[j] = new_fitness

        # Abandon worst nests
        abandon_count = max(
            1,
            int(PA * NESTS)
        )

        worst_indices = np.argsort(
            fitness_values
        )[-abandon_count:]

        for index in worst_indices:

            new_nest = initialize_nests()[0]

            nests[index] = new_nest.copy()

            fitness_values[index] = fitness(new_nest)

        # Update best solution
        current_best = np.argmin(fitness_values)

        if fitness_values[current_best] < best_fitness:

            best_fitness = fitness_values[current_best]

            best_solution = nests[current_best].copy()

        history.append(best_fitness)

        print(
            "Iteration:", iteration + 1,
            "| Best Fitness:", round(best_fitness, 6)
        )

    return best_solution, best_fitness, history


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    np.random.seed(42)

    print("\n====================================")
    print(" CUCKOO SEARCH RESOURCE ALLOCATION")
    print("====================================")

    # Run algorithm
    best_allocation, best_fitness, history = cuckoo_search()

    # Display allocation
    resources = ["CPU", "Memory", "Bandwidth"]

    print("\n========== OPTIMAL RESOURCE ALLOCATION ==========")

    for i in range(NUM_USERS):

        print("\nUser", i + 1)

        for j in range(NUM_RESOURCES):

            print(
                resources[j], ":",
                round(best_allocation[i, j], 2)
            )

    # Display resource usage
    print("\n========== TOTAL RESOURCE USAGE ==========")

    for j in range(NUM_RESOURCES):

        used = np.sum(best_allocation[:, j])

        print(
            resources[j], ":",
            round(used, 2),
            "/",
            TOTAL_RESOURCES[j]
        )

    # Display fitness
    print("\n========== FINAL RESULT ==========")

    print(
        "Best Fitness Value:",
        round(best_fitness, 6)
    )

    print(
        "Total Iterations:",
        MAX_ITER
    )

    # Check resource constraints
    print("\n========== CONSTRAINT CHECK ==========")

    for j in range(NUM_RESOURCES):

        used = np.sum(best_allocation[:, j])

        if used <= TOTAL_RESOURCES[j] + 1e-8:

            print(resources[j], ": Satisfied")

        else:

            print(resources[j], ": Violated")

    print("\n========== ALGORITHM COMPLETED ==========")
