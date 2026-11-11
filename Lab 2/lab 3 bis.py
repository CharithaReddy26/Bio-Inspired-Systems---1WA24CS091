
import numpy as np

# ==========================================
# PARTICLE SWARM OPTIMIZATION FOR JOB SCHEDULING
# ==========================================

# Problem parameters
NUM_JOBS = 10
NUM_MACHINES = 3

# Processing time of each job
processing_times = np.array([
    10, 5, 8, 7, 12, 6, 9, 4, 11, 3
])

# PSO parameters
NUM_PARTICLES = 30
MAX_ITERATIONS = 100

W = 0.7       # Inertia weight
C1 = 1.5      # Cognitive coefficient
C2 = 1.5      # Social coefficient

# Velocity limits
V_MAX = 2.0
V_MIN = -2.0


# ==========================================
# 1. FITNESS FUNCTION
# ==========================================

def fitness(position):

    # Convert continuous position into machine assignments
    assignments = np.rint(position).astype(int)

    assignments = np.clip(
        assignments, 0, NUM_MACHINES - 1
    )

    # Calculate workload of each machine
    machine_loads = np.zeros(NUM_MACHINES)

    for job in range(NUM_JOBS):

        machine = assignments[job]

        machine_loads[machine] += processing_times[job]

    # Makespan = maximum machine workload
    makespan = np.max(machine_loads)

    return makespan


# ==========================================
# 2. INITIALIZE PARTICLES
# ==========================================

def initialize_particles():

    positions = np.random.uniform(
        0, NUM_MACHINES - 1,
        (NUM_PARTICLES, NUM_JOBS)
    )

    velocities = np.random.uniform(
        V_MIN, V_MAX,
        (NUM_PARTICLES, NUM_JOBS)
    )

    return positions, velocities


# ==========================================
# 3. PARTICLE SWARM OPTIMIZATION
# ==========================================

def pso_job_scheduling():

    # Initialize population
    positions, velocities = initialize_particles()

    # Initialize personal best positions
    pbest_positions = positions.copy()

    pbest_fitness = np.array([
        fitness(p) for p in positions
    ])

    # Initialize global best
    best_index = np.argmin(pbest_fitness)

    gbest_position = pbest_positions[
        best_index
    ].copy()

    gbest_fitness = pbest_fitness[
        best_index
    ]

    history = []

    # Main PSO iterations
    for iteration in range(MAX_ITERATIONS):

        for i in range(NUM_PARTICLES):

            # Generate random numbers
            r1 = np.random.random(NUM_JOBS)
            r2 = np.random.random(NUM_JOBS)

            # Velocity update equation
            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (
                    pbest_positions[i] - positions[i]
                )
                + C2 * r2 * (
                    gbest_position - positions[i]
                )
            )

            # Limit velocity
            velocities[i] = np.clip(
                velocities[i], V_MIN, V_MAX
            )

            # Update particle position
            positions[i] += velocities[i]

            # Keep positions within search space
            positions[i] = np.clip(
                positions[i],
                0,
                NUM_MACHINES - 1
            )

            # Evaluate new fitness
            current_fitness = fitness(positions[i])

            # Update personal best
            if current_fitness < pbest_fitness[i]:

                pbest_fitness[i] = current_fitness

                pbest_positions[i] = positions[i].copy()

        # Update global best
        best_index = np.argmin(pbest_fitness)

        if pbest_fitness[best_index] < gbest_fitness:

            gbest_fitness = pbest_fitness[best_index]

            gbest_position = pbest_positions[
                best_index
            ].copy()

        history.append(gbest_fitness)

        print(
            "Iteration:", iteration + 1,
            "| Best Makespan:", gbest_fitness
        )

    return gbest_position, gbest_fitness, history


# ==========================================
# 4. DISPLAY JOB SCHEDULE
# ==========================================

def display_schedule(best_position, best_makespan):

    # Convert position into machine assignment
    assignments = np.rint(
        best_position
    ).astype(int)

    assignments = np.clip(
        assignments, 0, NUM_MACHINES - 1
    )

    machine_loads = np.zeros(NUM_MACHINES)

    print("\n======================================")
    print("       OPTIMAL JOB SCHEDULING")
    print("======================================")

    for machine in range(NUM_MACHINES):

        print("\nMachine", machine + 1)

        current_time = 0

        for job in range(NUM_JOBS):

            if assignments[job] == machine:

                start_time = current_time

                current_time += processing_times[job]

                print(
                    "Job", job + 1,
                    "| Processing Time:", processing_times[job],
                    "| Start:", start_time,
                    "| Finish:", current_time
                )

        machine_loads[machine] = current_time

        print("Total Machine Load:", current_time)

    # Final results
    print("\n======================================")
    print("           FINAL RESULTS")
    print("======================================")

    for i in range(NUM_MACHINES):

        print(
            "Machine", i + 1,
            "Load:", machine_loads[i]
        )

    print("\nBest Makespan:", best_makespan)

    print(
        "Total Processing Time:",
        np.sum(processing_times)
    )

    print(
        "Average Machine Load:",
        round(np.sum(processing_times) / NUM_MACHINES, 2)
    )

    # Machine utilization
    print("\nMachine Utilization:")

    for i in range(NUM_MACHINES):

        utilization = (
            machine_loads[i] / best_makespan
        ) * 100

        print(
            "Machine", i + 1,
            ":", round(utilization, 2), "%"
        )


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    np.random.seed(42)

    print("PARTICLE SWARM OPTIMIZATION")
    print("JOB SCHEDULING PROBLEM")

    print("\nNumber of Jobs:", NUM_JOBS)

    print("Number of Machines:", NUM_MACHINES)

    print("\nProcessing Times:", processing_times)

    # Execute PSO
    best_position, best_makespan, history = (
        pso_job_scheduling()
    )

    # Display optimal schedule
    display_schedule(
        best_position,
        best_makespan
    )

    print("\n======================================")
    print("       ALGORITHM COMPLETED")
    print("======================================")
