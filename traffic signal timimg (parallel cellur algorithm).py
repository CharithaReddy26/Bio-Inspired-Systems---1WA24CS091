
import numpy as np
from multiprocessing import Pool

DIRECTIONS = ["North", "South", "East", "West"]

MIN_GREEN = 10
MAX_GREEN = 60
TOTAL_GREEN = 120

ITERATIONS = 50
LEARNING_RATE = 0.3


# Cellular update function
def update_cell(args):

    index, old_times, traffic = args

    left = (index - 1) % 4
    right = (index + 1) % 4

    current = old_times[index]

    # Neighbor information
    neighbor_avg = (
        old_times[left] + old_times[right]
    ) / 2

    # Calculate desired green time
    desired = (
        traffic[index] / np.sum(traffic)
    ) * TOTAL_GREEN

    # PCA cellular update
    new_time = (
        current
        + LEARNING_RATE * (desired - current)
        + 0.05 * (neighbor_avg - current)
    )

    return index, new_time


# Ensure signal timings satisfy constraints
def allocate_time(values):

    # Find a feasible allocation with minimum
    # and maximum green time constraints.
    low = 0.0
    high = TOTAL_GREEN

    for _ in range(100):
        mid = (low + high) / 2

        allocated = np.clip(
            values * mid,
            MIN_GREEN,
            MAX_GREEN
        )

        if allocated.sum() > TOTAL_GREEN:
            high = mid
        else:
            low = mid

    return np.clip(
        values * low,
        MIN_GREEN,
        MAX_GREEN
    )


# PCA optimization
def parallel_cellular_algorithm(traffic):

    green_times = allocate_time(
        traffic / np.sum(traffic)
    )

    with Pool(processes=4) as pool:

        for iteration in range(ITERATIONS):

            tasks = [
                (i, green_times.copy(), traffic.copy())
                for i in range(4)
            ]

            # Parallel processing
            results = pool.map(update_cell, tasks)

            new_times = np.zeros(4)

            for index, value in results:
                new_times[index] = value

            # Enforce feasible signal allocation
            green_times = allocate_time(new_times)

            print(
                f"Iteration {iteration + 1}:",
                np.round(green_times, 2)
            )

    return green_times


# Main program
if __name__ == "__main__":

    print("===================================")
    print(" TRAFFIC SIGNAL OPTIMIZATION PCA")
    print("===================================")

    traffic = []

    # Get user input
    for direction in DIRECTIONS:

        while True:
            try:
                vehicles = int(input(
                    f"Enter number of vehicles in {direction}: "
                ))

                if vehicles < 0:
                    print("Enter a non-negative number.")
                    continue

                traffic.append(vehicles)
                break

            except ValueError:
                print("Please enter a valid integer.")

    traffic = np.array(traffic, dtype=float)

    if traffic.sum() == 0:
        print("All roads are empty. No optimization needed.")
        exit()

    print("\nInitial Traffic Density")

    for i in range(4):
        print(DIRECTIONS[i], ":", int(traffic[i]), "vehicles")

    print("\nStarting PCA Optimization...\n")

    optimized = parallel_cellular_algorithm(traffic)

    print("\n===================================")
    print(" OPTIMIZED TRAFFIC SIGNAL TIMINGS")
    print("===================================")

    for i in range(4):
        print(
            f"{DIRECTIONS[i]}: "
            f"{optimized[i]:.2f} seconds"
        )

    print("-----------------------------------")
    print(
        "Total Green Time:",
        round(optimized.sum(), 2),
        "seconds"
    )

    print("\nOptimization Completed!")
