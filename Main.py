# main.py

import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Read input data from Excel file
file_path = 'input_university.xlsx'
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
time_slots = list(range(1, 13))  # Assuming 12 time slots
rooms = df_rooms['Room'].tolist()

# Create mapping for quick lookup
unit_class_to_index = {tuple(units[i]): i for i in range(len(units))}

# Process students data
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

# Preprocess: create room types and capacities mapping
room_capacities = df_rooms.set_index('Room')['Capacity'].to_dict()
room_types = df_rooms.set_index('Room')['Type'].to_dict()

# Genetic algorithm parameters 
POPULATION_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.9
ELITISM_RATE = 0.1
PATIENCE = 50

def get_allowed_room_types(class_type):
    if class_type == 1:
        return ['Lecture Hall']
    elif class_type == 2:
        return ['Lecture Hall', 'Room', 'Lab']
    elif class_type == 3:
        return ['Lab']
    else:
        return ['Room']

# Create chromosome, considering that Lectures occupy two consecutive time slots
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

# Greedy initialization for better starting solutions
def greedy_initialization(units, teachers, days, time_slots, rooms, room_types, students):
    num_units = len(units)
    num_days = len(days)
    num_time_slots = len(time_slots)
    chromosome = [None] * num_units
    teacher_schedule = {}
    room_schedule = {}
    student_schedule = {}

    # Precompute class_students mapping
    class_students = {}
    for student, classes in students.items():
        for key in classes:
            if key not in class_students:
                class_students[key] = set()
            class_students[key].add(student)

    # Sort units based on class type (Lectures first)
    units_sorted = sorted(range(num_units), key=lambda i: units[i][1])

    for i in units_sorted:
        unit, class_type, class_id = units[i]
        teacher = teachers[i]
        allowed_room_types = get_allowed_room_types(class_type)
        possible_rooms = [room for room in rooms if room_types[room] in allowed_room_types]
        random.shuffle(possible_rooms)

        # Handle Lecture occupying two consecutive time slots
        if class_type == 1:
            duration = 2
        else:
            duration = 1

        assigned = False
        for day_idx in range(num_days):
            for time_slot_idx in range(num_time_slots - (duration - 1)):
                time_slots_needed = [time_slot_idx + d for d in range(duration)]
                # Check teacher availability
                if pd.notnull(teacher) and teacher != 'None':
                    teacher_busy = any((teacher, day_idx, ts) in teacher_schedule for ts in time_slots_needed)
                else:
                    teacher_busy = False
                if teacher_busy:
                    continue
                # Check student conflicts
                students_in_class = class_students.get((unit, class_type, class_id), set())
                student_conflict = False
                for student in students_in_class:
                    if any((day_idx, ts) in student_schedule.get(student, set()) for ts in time_slots_needed):
                        student_conflict = True
                        break
                if student_conflict:
                    continue
                # Try to find a room
                for room in possible_rooms:
                    room_busy = any((room, day_idx, ts) in room_schedule for ts in time_slots_needed)
                    if room_busy:
                        continue
                    # Assign the class
                    chromosome[i] = (day_idx, time_slot_idx, room)
                    # Mark teacher as busy
                    if pd.notnull(teacher) and teacher != 'None':
                        for ts in time_slots_needed:
                            teacher_schedule[(teacher, day_idx, ts)] = True
                    # Mark room as busy
                    for ts in time_slots_needed:
                        room_schedule[(room, day_idx, ts)] = True
                    # Mark students as busy
                    for student in students_in_class:
                        if student not in student_schedule:
                            student_schedule[student] = set()
                        for ts in time_slots_needed:
                            student_schedule[student].add((day_idx, ts))
                    assigned = True
                    break
                if assigned:
                    break
            if assigned:
                break
        if not assigned:
            # If not assigned, assign randomly (fallback)
            day_idx = random.randint(0, num_days - 1)
            if class_type == 1:
                time_slot_idx = random.randint(0, num_time_slots - 2)
            else:
                time_slot_idx = random.randint(0, num_time_slots - 1)
            room = random.choice(possible_rooms)
            chromosome[i] = (day_idx, time_slot_idx, room)
    return chromosome

# Initialize population with a mix of greedy and random chromosomes
def initialize_population(units, teachers, days, time_slots, df_rooms, rooms, room_types, students):
    population = []
    num_greedy = int(POPULATION_SIZE * 0.2)  # 20% greedy initialization
    num_random = POPULATION_SIZE - num_greedy

    # Greedy initialization
    for _ in range(num_greedy):
        chromosome = greedy_initialization(units, teachers, days, time_slots, rooms, room_types, students)
        population.append(chromosome)

    # Random initialization
    for _ in range(num_random):
        chromosome = create_chromosome(len(units), len(days), len(time_slots), units, rooms, room_types)
        population.append(chromosome)
    return population

# Fitness function
def fitness(chromosome, units, teachers, days, time_slots, rooms, students,
            room_capacities, room_types, unit_class_to_index):
    score = 10000  # Adjusted max score
    hard_constraint_violations = 0
    soft_constraint_violations = 0

    HARD_CONSTRAINT_PENALTY = 500  # 提高硬约束惩罚力度
    SOFT_CONSTRAINT_PENALTY = 10

    teacher_time_slots = set()
    room_time_slots = set()
    student_time_slots = {}
    unit_times = {}

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

        # Check room type constraints
        allowed_room_types = get_allowed_room_types(class_type)
        room_type = room_types[room]
        if room_type not in allowed_room_types:
            score -= HARD_CONSTRAINT_PENALTY
            hard_constraint_violations += 1

        # Check room capacity
        room_capacity = room_capacities[room]
        if class_type == 1:
            # For Lectures, capacity is total students enrolled in the lecture
            capacity_required = len(class_students.get((unit, class_type, class_id), []))
            if capacity_required > room_capacity:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
        else:
            # For Practicals and Labs, capacity is determined by room capacity
            capacity_required = room_capacity  # Capacity is effectively the room capacity

        # Handle Lecture occupying two consecutive time slots
        if class_type == 1:  # Lecture
            if time_slot_idx + 1 >= num_time_slots:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
                continue
            time_slots_needed = [time_slot_idx, time_slot_idx + 1]
        else:
            time_slots_needed = [time_slot_idx]

        # Check teacher time conflicts (only for Lectures)
        if pd.notnull(teacher) and teacher != 'None':
            for ts in time_slots_needed:
                teacher_time_slot_key = (teacher, day_idx, ts)
                if teacher_time_slot_key in teacher_time_slots:
                    score -= HARD_CONSTRAINT_PENALTY
                    hard_constraint_violations += 1
                else:
                    teacher_time_slots.add(teacher_time_slot_key)

        # Check room time conflicts
        for ts in time_slots_needed:
            room_time_slot_key = (room, day_idx, ts)
            if room_time_slot_key in room_time_slots:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
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

    return score, hard_constraint_violations, soft_constraint_violations, {}

# Crossover function
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation function considering Lecture time slots
def mutate(chromosome, num_days, num_time_slots, units, rooms, room_types):
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
    tournament_size = 7
    selected = random.sample(list(zip(population, fitness_values)), tournament_size)
    selected.sort(key=lambda x: x[1][0], reverse=True)
    return selected[0][0]

# Genetic algorithm main function
def genetic_algorithm(units, teachers, days, time_slots, rooms, students, df_rooms):
    num_units = len(units)
    num_days = len(days)
    num_time_slots = len(time_slots)

    # Preprocess data
    unit_class_to_index = {tuple(units[i]): i for i in range(len(units))}
    room_capacities = df_rooms.set_index('Room')['Capacity'].to_dict()
    room_types = df_rooms.set_index('Room')['Type'].to_dict()

    # Initialize population
    population = initialize_population(units, teachers, days, time_slots, df_rooms, rooms, room_types, students)

    # Save initial schedule
    initial_chromosome = population[0]
    initial_schedule_df = chromosome_to_schedule(initial_chromosome, units, teachers, days, time_slots)

    # Compute initial fitness and violations
    initial_fitness, _, _, initial_time_slot_violations = fitness(
        initial_chromosome, units, teachers, days, time_slots, rooms, students,
        room_capacities, room_types, unit_class_to_index)

    best_fitness = None
    generations_without_improvement = 0
    fitness_history = []

    # Record best chromosome violations
    best_time_slot_violations = {}

    for generation in range(GENERATIONS):
        fitness_values = [
            fitness(
                chromosome, units, teachers, days, time_slots, rooms, students,
                room_capacities, room_types, unit_class_to_index
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

        # Check for new best solution
        current_best_fitness = fitness_scores[fitness_scores.index(max(fitness_scores))]
        fitness_history.append(current_best_fitness)

        if best_fitness is None or current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = sorted_population[0]
            best_index = population.index(best_chromosome)
            # Save violations of the best chromosome
            best_time_slot_violations = fitness_values[best_index][3]
            hard_constraints = fitness_values[best_index][1]
            soft_constraints = fitness_values[best_index][2]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Early stopping if no improvement
        if generations_without_improvement >= PATIENCE:
            logging.info("Early stopping: no improvement in fitness")
            break

        # Logging
        print(
            f'Generation {generation}: Best Fitness = {best_fitness}, '
            f'Hard Constraints = {hard_constraints}, Soft Constraints = {soft_constraints}'
        )

        # Generate new population
        new_population = elites.copy()

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, num_days, num_time_slots, units, rooms, room_types)
            mutate(child2, num_days, num_time_slots, units, rooms, room_types)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    # Plot fitness over generations
    plt.plot(fitness_history)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.show()

    return (
        best_chromosome, best_fitness, hard_constraints, soft_constraints,
        initial_schedule_df, {}, best_time_slot_violations
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

# Display final schedule
def display_schedule(chromosome, units, teachers, days, time_slots, rooms, students, best_fitness, hard_constraints, soft_constraints):
    schedule_df = chromosome_to_schedule(chromosome, units, teachers, days, time_slots)

    # Print schedule
    print("\nFinal Schedule:")
    print(schedule_df.to_string(index=False))

    print(f'\nBest Fitness: {best_fitness}')
    print(f'Hard Constraint Violations: {hard_constraints}')
    print(f'Soft Constraint Violations: {soft_constraints}')

    return schedule_df

if __name__ == '__main__':
    # Run genetic algorithm and save results
    start_time = time.time()
    (best_schedule, best_fitness, hard_constraints, soft_constraints,
     initial_schedule_df, initial_time_slot_violations, best_time_slot_violations) = genetic_algorithm(
        units, teachers, days, time_slots, rooms, students, df_rooms)
    end_time = time.time()

    print(f"\nExecution Time: {end_time - start_time} seconds")
    final_schedule_df = display_schedule(best_schedule, units, teachers, days, time_slots, rooms, students, best_fitness, hard_constraints, soft_constraints)

    # Save initial and final schedules
    initial_schedule_df.to_csv('initial_schedule.csv', index=False)
    final_schedule_df.to_csv('final_schedule.csv', index=False)

