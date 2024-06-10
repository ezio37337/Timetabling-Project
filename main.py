import pandas as pd
import random
import time

# Read the input data from Excel file
file_path = 'input2.xlsx'
df_courses = pd.read_excel(file_path, sheet_name='Courses')
df_students = pd.read_excel(file_path, sheet_name='Students')

# Assume the Courses sheet has the following columns: Course, Professor, Room
courses = df_courses['Course'].tolist()
professors = df_courses['Professor'].tolist()
rooms = df_courses['Room'].tolist()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']  # 使用星期的英文简写
time_slots = list(range(1, 6))  # 每天有5个时间段

# Assume the Students sheet has the following columns: Student, Course
students = df_students.groupby('Student')['Course'].apply(list).to_dict()

# Define the parameters of the genetic algorithm
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8

def create_chromosome(num_courses, num_days, num_time_slots):
    return [(random.randint(0, num_days - 1), random.randint(0, num_time_slots - 1)) for _ in range(num_courses)]

def initialize_population(num_courses, num_days, num_time_slots):
    return [create_chromosome(num_courses, num_days, num_time_slots) for _ in range(POPULATION_SIZE)]

def fitness(chromosome, courses, professors, days, time_slots, rooms, students):
    # Initial fitness value set to a large number
    score = 1000
    schedule = []

    # 将染色体中的基因转换为可读的课程安排
    for course_idx, (day_idx, time_slot_idx) in enumerate(chromosome):
        schedule.append((courses[course_idx], professors[course_idx], days[day_idx], time_slots[time_slot_idx], rooms[course_idx]))

    hard_constraint_constraints = 0
    soft_constraint_constraints = 0

    # 硬约束：同一时间段内一个教授只能教一门课
    prof_time_slot = {}
    for course, professor, day, time_slot, room in schedule:
        if (professor, day, time_slot) in prof_time_slot:
            score -= 200  # 违反硬约束，扣200分
            hard_constraint_constraints += 1
        else:
            prof_time_slot[(professor, day, time_slot)] = True

    # 硬约束：同一时间段内一个教室只能被占用一次
    room_time_slot = {}
    for course, professor, day, time_slot, room in schedule:
        if (room, day, time_slot) in room_time_slot:
            score -= 200  # 违反硬约束，扣200分
            hard_constraint_constraints += 1
        else:
            room_time_slot[(room, day, time_slot)] = True

    # 软约束：避免教授在同一天内连续上课时间过长
    prof_teaching_hours = {}
    for course, professor, day, time_slot, room in schedule:
        if professor not in prof_teaching_hours:
            prof_teaching_hours[professor] = {}
        if day not in prof_teaching_hours[professor]:
            prof_teaching_hours[professor][day] = []
        prof_teaching_hours[professor][day].append(time_slot)

    for professor, days in prof_teaching_hours.items():
        for day, slots in days.items():
            slots.sort()
            if len(slots) > 3:  # 如果一天内上课超过3节课，扣分
                score -= 20  # 软约束，轻微扣分
                soft_constraint_constraints += 1
            for i in range(1, len(slots)):
                if slots[i] - slots[i - 1] == 1:  # 连续时间段
                    score -= 10  # 软约束，轻微扣分
                    soft_constraint_constraints += 1

    # 学生约束：学生不能在同一时间段内有两门课程
    student_time_slot = {}
    for student, student_courses in students.items():
        student_time_slot[student] = {}
        for course in student_courses:
            if course in courses:
                course_idx = courses.index(course)
                day, time_slot = chromosome[course_idx]
                if day not in student_time_slot[student]:
                    student_time_slot[student][day] = []
                student_time_slot[student][day].append(time_slot)

        for day, slots in student_time_slot[student].items():
            slots.sort()
            if len(slots) == 0:  # 如果学生某一天没课，扣分
                score -= 5  # 软约束，轻微扣分
                soft_constraint_constraints += 1
            if len(slots) > 3:  # 如果学生一天内有超过3节课，扣分
                score -= 20  # 软约束，轻微扣分
                soft_constraint_constraints += 1
            for i in range(1, len(slots)):
                if slots[i] == slots[i - 1]:  # 时间冲突
                    score -= 200  # 违反硬约束，扣200分
                    hard_constraint_constraints += 1

    return score, hard_constraint_constraints, soft_constraint_constraints

def tournament_selection(population, courses, professors, days, time_slots, rooms, students):
    tournament_size = 5
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda x: fitness(x, courses, professors, days, time_slots, rooms, students)[0], reverse=True)
    return selected[0]

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2

    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(chromosome, num_days, num_time_slots):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = (random.randint(0, num_days - 1), random.randint(0, num_time_slots - 1))

def genetic_algorithm(courses, professors, days, time_slots, rooms, students):
    num_courses = len(courses)
    num_days = len(days)
    num_time_slots = len(time_slots)
    population = initialize_population(num_courses, num_days, num_time_slots)
    
    for generation in range(GENERATIONS):
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1 = tournament_selection(population, courses, professors, days, time_slots, rooms, students)
            parent2 = tournament_selection(population, courses, professors, days, time_slots, rooms, students)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, num_days, num_time_slots)
            mutate(child2, num_days, num_time_slots)
            new_population.extend([child1, child2])

        population = new_population

        # Track the best solution
        best_chromosome = max(population, key=lambda x: fitness(x, courses, professors, days, time_slots, rooms, students)[0])
        best_fitness, hard_constraints, soft_constraints = fitness(best_chromosome, courses, professors, days, time_slots, rooms, students)
        print(f'Generation {generation}: Best Fitness = {best_fitness}, Hard Constraints = {hard_constraints}, Soft Constraints = {soft_constraints}')

    return best_chromosome, best_fitness, hard_constraints, soft_constraints

def display_schedule(chromosome, courses, professors, days, time_slots, rooms, students, best_fitness, hard_constraints, soft_constraints):
    schedule = []

    for course_idx, (day_idx, time_slot_idx) in enumerate(chromosome):
        schedule.append((courses[course_idx], professors[course_idx], days[day_idx], time_slots[time_slot_idx], rooms[course_idx]))

    for student, student_courses in students.items():
        print(f'Student: {student}')
        for course in student_courses:
            if course in courses:
                course_idx = courses.index(course)
                course, professor, day, time_slot, room = schedule[course_idx]
                print(f'  Course: {course}, Professor: {professor}, Day: {day}, TimeSlot: {time_slot}, Room: {room}')
    
    print(f'Best Fitness: {best_fitness}')
    print(f'Hard Constraint Constraints: {hard_constraints}')
    print(f'Soft Constraint Constraints: {soft_constraints}')

start_time = time.time()
best_schedule, best_fitness, hard_constraints, soft_constraints = genetic_algorithm(courses, professors, days, time_slots, rooms, students)
end_time = time.time()

print(f"Execution Time: {end_time - start_time} seconds")
display_schedule(best_schedule, courses, professors, days, time_slots, rooms, students, best_fitness, hard_constraints, soft_constraints)
