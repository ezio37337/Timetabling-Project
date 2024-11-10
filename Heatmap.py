# main.py

import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Read input data from Excel file
file_path = 'input_university_large.xlsx'
df_units = pd.read_excel(file_path, sheet_name='Units')
df_students = pd.read_excel(file_path, sheet_name='Students')
df_rooms = pd.read_excel(file_path, sheet_name='Rooms')

# Ensure data consistency
df_units['ClassType'] = df_units['ClassType'].astype(int)
df_units['ClassID'] = df_units['ClassID'].astype(int)
df_rooms['Type'] = df_rooms['Type'].str.strip().str.title()

valid_room_types = ['Lecture Hall', 'Lab', 'Room']
if not df_rooms['Type'].isin(valid_room_types).all():
    raise ValueError("df_rooms['Type'] contains invalid values.")

# Create list of units, including unit, class type, and class ID
units = df_units[['Unit', 'ClassType', 'ClassID']].values.tolist()
teachers = df_units['Teacher'].tolist()
capacities = df_units['Capacity'].tolist()

# Define days and time slots
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
time_slots = list(range(1, 13))  # Assume there are 12 time slots per day
rooms = df_rooms['Room'].tolist()

# Create a mapping for quick lookup
unit_class_to_index = {tuple(units[i]): i for i in range(len(units))}

# Process student data
students = {}
for index, row in df_students.iterrows():
    student = row['StudentID']
    unit = row['Unit']
    class_type = row['ClassType']
    class_id = row['ClassID']
    key = (unit, class_type, class_id)
    if student not in students:
        students[student] = []
    students[student].append(key)

# Preprocess: create room type and capacity mappings
room_capacities = df_rooms.set_index('Room')['Capacity'].to_dict()
room_types = df_rooms.set_index('Room')['Type'].to_dict()

def get_allowed_room_types(class_type):
    if class_type == 1:
        return ['Lecture Hall']
    elif class_type == 2:
        return ['Lecture Hall', 'Room', 'Lab']
    elif class_type == 3:
        return ['Lab']
    else:
        return ['Room']

# Create chromosome, considering lectures occupy two consecutive time slots
def create_chromosome(num_units, num_days, num_time_slots, units, rooms, room_types):
    chromosome = []
    for i in range(num_units):
        unit, class_type, class_id = units[i]
        day_idx = random.randint(0, num_days - 1)
        if class_type == 1:  # Lecture
            time_slot_idx = random.randint(0, num_time_slots - 2)  # Ensure enough time slots
        else:
            time_slot_idx = random.randint(0, num_time_slots - 1)
        # Choose rooms matching the class type
        allowed_room_types = get_allowed_room_types(class_type)
        possible_rooms = [room for room in rooms if room_types[room] in allowed_room_types]
        room = random.choice(possible_rooms)
        chromosome.append((day_idx, time_slot_idx, room))
    return chromosome


# Initialize population with a mix of greedy and random chromosomes
def initialize_population(units, teachers, days, time_slots, df_rooms, rooms, room_types, students, POPULATION_SIZE):
    population = []
    num_random = POPULATION_SIZE

    # Random initialization
    for _ in range(num_random):
        chromosome = create_chromosome(len(units), len(days), len(time_slots), units, rooms, room_types)
        population.append(chromosome)
    return population

# Fitness function
def fitness(chromosome, units, teachers, days, time_slots, rooms, students,
            room_capacities, room_types, unit_class_to_index, HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY):
    score = 10000  # Adjusted max score
    hard_constraint_violations = 0
    soft_constraint_violations = 0

    teacher_time_slots = set()
    room_time_slots = set()
    student_time_slots = {}
    unit_times = {}
    time_slot_violations = {}

    num_time_slots = len(time_slots)

    # Precompute class_students mapping
    class_students = {}
    for student, classes in students.items():
        for key in classes:
            if key not in class_students:
                class_students[key] = set()
            class_students[key].add(student)

    for i in range(len(chromosome)):
        day_idx, time_slot_idx, room = chromosome[i]
        unit, class_type, class_id = units[i]
        teacher = teachers[i]
        capacity_required = None  # Will be checked based on room capacity

        # Handle Lecture occupying two consecutive time slots
        if class_type == 1:  # Lecture
            if time_slot_idx + 1 >= num_time_slots:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
                # Record violation
                ts_list = [time_slot_idx]
                for ts in ts_list:
                    time_slot_violations[(day_idx, ts)] = time_slot_violations.get((day_idx, ts), 0) + 1
                continue
            time_slots_needed = [time_slot_idx, time_slot_idx + 1]
        else:
            time_slots_needed = [time_slot_idx]

        # Check room type constraints
        allowed_room_types = get_allowed_room_types(class_type)
        room_type = room_types[room]
        if room_type not in allowed_room_types:
            score -= HARD_CONSTRAINT_PENALTY
            hard_constraint_violations += 1
            # Record violation
            for ts in time_slots_needed:
                time_slot_violations[(day_idx, ts)] = time_slot_violations.get((day_idx, ts), 0) + 1

        # Check room capacity
        room_capacity = room_capacities[room]
        if class_type == 1:
            # For Lectures, capacity is total students enrolled in the lecture
            capacity_required = len(class_students.get((unit, class_type, class_id), []))
            if capacity_required > room_capacity:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
                # Record violation
                for ts in time_slots_needed:
                    time_slot_violations[(day_idx, ts)] = time_slot_violations.get((day_idx, ts), 0) + 1
        else:
            # For Practicals and Labs, capacity is determined by room capacity
            capacity_required = room_capacity  # Capacity is effectively the room capacity

        # Check teacher time conflicts (only for Lectures)
        if pd.notnull(teacher) and teacher != 'None':
            for ts in time_slots_needed:
                teacher_time_slot_key = (teacher, day_idx, ts)
                if teacher_time_slot_key in teacher_time_slots:
                    score -= HARD_CONSTRAINT_PENALTY
                    hard_constraint_violations += 1
                    # Record violation
                    time_slot_violations[(day_idx, ts)] = time_slot_violations.get((day_idx, ts), 0) + 1
                else:
                    teacher_time_slots.add(teacher_time_slot_key)

        # Check room time conflicts
        for ts in time_slots_needed:
            room_time_slot_key = (room, day_idx, ts)
            if room_time_slot_key in room_time_slots:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
                # Record violation
                time_slot_violations[(day_idx, ts)] = time_slot_violations.get((day_idx, ts), 0) + 1
            else:
                room_time_slots.add(room_time_slot_key)

        # Record unit times for soft constraint checks
        unit_times[(unit, class_type, class_id)] = (day_idx, time_slot_idx)

    # Student conflicts
    for student, class_keys in students.items():
        if student not in student_time_slots:
            student_time_slots[student] = set()
        for key in class_keys:
            if key in unit_class_to_index:
                course_idx = unit_class_to_index[key]
                day_idx, time_slot_idx, room = chromosome[course_idx]
                class_type = units[course_idx][1]
                # Handle Lecture occupying two time slots
                if class_type == 1:
                    time_slots_needed = [time_slot_idx, time_slot_idx + 1]
                else:
                    time_slots_needed = [time_slot_idx]
                for ts in time_slots_needed:
                    student_time_slot = (day_idx, ts)
                    if student_time_slot in student_time_slots[student]:
                        # Student time conflict
                        score -= HARD_CONSTRAINT_PENALTY
                        hard_constraint_violations += 1
                        # Record violation
                        time_slot_violations[(day_idx, ts)] = time_slot_violations.get((day_idx, ts), 0) + 1
                    else:
                        student_time_slots[student].add(student_time_slot)

    # Soft Constraints
    # 1. Practicals and Labs should be scheduled after the Lecture
    for student, class_keys in students.items():
        unit_classes = {}
        for key in class_keys:
            unit = key[0]
            class_type = key[1]
            if unit not in unit_classes:
                unit_classes[unit] = {}
            unit_classes[unit][class_type] = key

        for unit, classes in unit_classes.items():
            lecture_time = unit_times.get(classes.get(1))
            practical_time = unit_times.get(classes.get(2))
            lab_time = unit_times.get(classes.get(3))

            if lecture_time:
                lecture_total_time = lecture_time[0] * num_time_slots + lecture_time[1]
                if practical_time:
                    practical_total_time = practical_time[0] * num_time_slots + practical_time[1]
                    if practical_total_time <= lecture_total_time:
                        score -= SOFT_CONSTRAINT_PENALTY
                        soft_constraint_violations += 1
                if lab_time:
                    lab_total_time = lab_time[0] * num_time_slots + lab_time[1]
                    if lab_total_time <= lecture_total_time:
                        score -= SOFT_CONSTRAINT_PENALTY
                        soft_constraint_violations += 1

    # 2. Students' classes should be spread throughout the week
    for student, times in student_time_slots.items():
        days_with_classes = set([t[0] for t in times])
        if len(days_with_classes) <= 2:
            score -= SOFT_CONSTRAINT_PENALTY
            soft_constraint_violations += 1

    # 3. Avoid too many consecutive classes in a day
    for student, times in student_time_slots.items():
        day_time_slots = {}
        for day_idx, ts in times:
            if day_idx not in day_time_slots:
                day_time_slots[day_idx] = []
            day_time_slots[day_idx].append(ts)
        for day_idx, ts_list in day_time_slots.items():
            ts_list.sort()
            consecutive_classes = 1
            for i in range(len(ts_list) - 1):
                if ts_list[i + 1] == ts_list[i] + 1:
                    consecutive_classes += 1
                    if consecutive_classes > 4:  # More than 4 consecutive classes
                        score -= SOFT_CONSTRAINT_PENALTY
                        soft_constraint_violations += 1
                        break
                else:
                    consecutive_classes = 1

    return score, hard_constraint_violations, soft_constraint_violations, time_slot_violations

# Crossover function
def crossover(parent1, parent2, CROSSOVER_RATE):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation function considering Lecture time slots
def mutate(chromosome, num_days, num_time_slots, units, rooms, room_types, MUTATION_RATE):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            unit, class_type, class_id = units[i]
            day_idx = random.randint(0, num_days - 1)
            if class_type == 1:  # Lecture
                time_slot_idx = random.randint(0, num_time_slots - 2)
            else:
                time_slot_idx = random.randint(0, num_time_slots - 1)
            # Choose rooms matching the class type
            allowed_room_types = get_allowed_room_types(class_type)
            possible_rooms = [room for room in rooms if room_types[room] in allowed_room_types]
            room = random.choice(possible_rooms)
            chromosome[i] = (day_idx, time_slot_idx, room)

# Tournament selection function
def tournament_selection(population, fitness_values):
    tournament_size = 5
    selected = random.sample(list(zip(population, fitness_values)), tournament_size)
    selected.sort(key=lambda x: x[1][0], reverse=True)
    return selected[0][0]

def genetic_algorithm(units, teachers, days, time_slots, rooms, students, df_rooms,
                      POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE, PATIENCE,
                      HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY):
    num_units = len(units)
    num_days = len(days)
    num_time_slots = len(time_slots)

    # Preprocess data
    unit_class_to_index = {tuple(units[i]): i for i in range(len(units))}
    room_capacities = df_rooms.set_index('Room')['Capacity'].to_dict()
    room_types = df_rooms.set_index('Room')['Type'].to_dict()

    # Initialize population
    population = initialize_population(units, teachers, days, time_slots, df_rooms, rooms, room_types, students, POPULATION_SIZE)

    # Save initial schedule
    initial_chromosome = population[0]
    initial_schedule_df = chromosome_to_schedule(initial_chromosome, units, teachers, days, time_slots)

    # Compute initial fitness and violations
    initial_fitness, _, _, initial_time_slot_violations = fitness(
        initial_chromosome, units, teachers, days, time_slots, rooms, students,
        room_capacities, room_types, unit_class_to_index, HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY)

    best_fitness = None
    generations_without_improvement = 0
    fitness_history = []

    # Record best chromosome violations
    best_time_slot_violations = {}

    for generation in range(GENERATIONS):
        fitness_values = [
            fitness(
                chromosome, units, teachers, days, time_slots, rooms, students,
                room_capacities, room_types, unit_class_to_index,
                HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY
            )
            for chromosome in population
        ]

        # Extract fitness scores
        fitness_scores = [fv[0] for fv in fitness_values]

        # Sort population based on fitness
        sorted_population = [
            x for _, x in sorted(
                zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True
            )
        ]

        # Implement elitism
        num_elites = int(POPULATION_SIZE * ELITISM_RATE)
        elites = sorted_population[:num_elites]

        # Get current best fitness and corresponding indices
        current_best_fitness = fitness_scores[0]  # Since population is sorted, the first one is the best
        hard_constraints = fitness_values[0][1]
        soft_constraints = fitness_values[0][2]
        fitness_history.append(current_best_fitness)

        # Add progress logging every 10 generations
        if generation % 10 == 0 or generation == GENERATIONS - 1:
            logging.info(f"Generation {generation}: Best Fitness = {current_best_fitness}, Hard Violations = {hard_constraints}, Soft Violations = {soft_constraints}")

        # Check for new best solution
        if best_fitness is None or current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = sorted_population[0]
            # Save violations of the best chromosome
            best_time_slot_violations = fitness_values[0][3]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Early stopping if no improvement
        if generations_without_improvement >= PATIENCE:
            logging.info("Early stopping: no improvement in fitness")
            break

        # Generate new population
        new_population = elites.copy()

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            mutate(child1, num_days, num_time_slots, units, rooms, room_types, MUTATION_RATE)
            mutate(child2, num_days, num_time_slots, units, rooms, room_types, MUTATION_RATE)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    return (
        best_chromosome, best_fitness, hard_constraints, soft_constraints,
        initial_chromosome, initial_schedule_df, initial_time_slot_violations, best_time_slot_violations
    )
# Convert chromosome to schedule DataFrame
def chromosome_to_schedule(chromosome, units, teachers, days, time_slots):
    schedule = []

    for unit_idx, (day_idx, time_slot_idx, room) in enumerate(chromosome):
        day = days[day_idx]
        time_slot = time_slots[time_slot_idx]
        unit, class_type, class_id = units[unit_idx]
        teacher = teachers[unit_idx]
        if class_type == 1:
            duration = 2  # Lecture occupies two time slots
        else:
            duration = 1
        schedule.append((day, time_slot, duration, unit, class_type, class_id, teacher, room))
        # If Lecture, add next time slot
        if class_type == 1:
            schedule.append((day, time_slot + 1, 0, unit, class_type, class_id, teacher, room))

    df_schedule = pd.DataFrame(schedule, columns=['Day', 'Time Slot', 'Duration', 'Unit', 'Class Type', 'ClassID', 'Teacher', 'Room'])
    df_schedule['Class Type'] = df_schedule['Class Type'].map({1: 'Lecture', 2: 'Practical', 3: 'Lab'})
    df_schedule['Teacher'] = df_schedule['Teacher'].fillna('No Teacher')
    return df_schedule

# Plot heatmap of schedule violations
def plot_heatmap(time_slot_violations, title, num_time_slots, vmax=None):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    time_slots = list(range(1, num_time_slots + 1))

    # Create a DataFrame to hold violation counts
    data = pd.DataFrame(0, index=time_slots, columns=days)

    for (day_idx, ts_idx), violation_count in time_slot_violations.items():
        day = days[day_idx]
        ts = ts_idx + 1  # Adjust index to match time slot numbering
        data.loc[ts, day] = violation_count

    # Create custom colormap from green to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green', 'yellow', 'red'])

    plt.figure(figsize=(12, 8))

    # Normalize colors using vmax to ensure consistent scales between plots
    if vmax is None:
        vmax = data.max().max()
    norm = plt.Normalize(vmin=0, vmax=vmax)

    # Annotate cells with violation counts
    annot = data.astype(str)
    annot[data == 0] = ''  # Remove zeros

    # Plot heatmap
    sns.heatmap(data.astype(float), annot=annot, fmt='', cmap=cmap, cbar=True, linewidths=0.5, linecolor='black',
                annot_kws={"size": 8}, norm=norm)

    plt.title(title)
    plt.ylabel('Time Slot')
    plt.xlabel('Day')
    plt.yticks(rotation=0)
    plt.show()

if __name__ == '__main__':
    # Parameters
    GENERATIONS = 200
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.9
    ELITISM_RATE = 0.1
    PATIENCE = 200
    HARD_CONSTRAINT_PENALTY = 500
    SOFT_CONSTRAINT_PENALTY = 10
    POPULATION_SIZE = 100

    # Run genetic algorithm
    (best_chromosome, best_fitness, hard_constraints, soft_constraints,
     initial_chromosome, initial_schedule_df, initial_time_slot_violations, best_time_slot_violations) = genetic_algorithm(
        units, teachers, days, time_slots, rooms, students, df_rooms,
        POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE, PATIENCE,
        HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY)

    # Display final schedule
    final_schedule_df = chromosome_to_schedule(best_chromosome, units, teachers, days, time_slots)

    print(f'\nBest Fitness: {best_fitness}')
    print(f'Hard Constraint Violations: {hard_constraints}')
    print(f'Soft Constraint Violations: {soft_constraints}')

    # Get the maximum violation count for consistent color scales
    max_violation = max(
        max(initial_time_slot_violations.values(), default=0),
        max(best_time_slot_violations.values(), default=0)
    )

    # Plot initial schedule heatmap
    plot_heatmap(initial_time_slot_violations, 'Initial Schedule Heatmap', len(time_slots), vmax=max_violation)

    # Plot final schedule heatmap
    plot_heatmap(best_time_slot_violations, 'Final Schedule Heatmap', len(time_slots), vmax=max_violation)
