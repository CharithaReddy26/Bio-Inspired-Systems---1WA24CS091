import numpy as np

# Get user input
n = int(input("Enter number of cities: "))

cities = []
for i in range(n):
    city = input(f"Enter city {i+1} name: ")
    cities.append(city)

# Get distance matrix
dist = np.zeros((n, n))

print("\nEnter distance between cities:")

for i in range(n):
    for j in range(i + 1, n):
        d = float(input(
            f"Distance between {cities[i]} and {cities[j]}: "
        ))
        dist[i][j] = d
        dist[j][i] = d

# GWO parameters
wolves = 30
iterations = 200

# Fitness function
def fitness(route):
    total = 0
    for i in range(n):
        total += dist[route[i]][route[(i + 1) % n]]
    return total

# Initialize population
population = [
    np.random.permutation(n)
    for _ in range(wolves)
]

best_route = None
best_distance = float("inf")

# Optimization
for iteration in range(iterations):

    # Evaluate all wolves and sort by fitness
    population.sort(key=fitness)

    alpha = population[0].copy()
    beta = population[1].copy()
    delta = population[2].copy()

    if fitness(alpha) < best_distance:
        best_distance = fitness(alpha)
        best_route = alpha.copy()

    a = 2 - 2 * iteration / iterations

    new_population = [alpha.copy()]

    # Generate new routes
    for wolf in population[1:]:

        route = wolf.copy()

        for leader in [alpha, beta, delta]:

            if np.random.rand() < (a / 2 + 0.1):

                i, j = sorted(
                    np.random.choice(n, 2, replace=False)
                )

                segment = list(leader[i:j+1])

                remaining = [
                    city for city in route
                    if city not in segment
                ]

                route = (
                    remaining[:i] + segment + remaining[i:]
                )

        new_population.append(np.array(route))

    population = new_population

# Display result
print("\n========== GWO TSP RESULT ==========")

print("\nBest Travelling Route:")

route_names = [cities[i] for i in best_route]
route_names.append(route_names[0])

print(" -> ".join(route_names))

print("\nMinimum Distance:",
      best_distance)

print("\n====================================")

