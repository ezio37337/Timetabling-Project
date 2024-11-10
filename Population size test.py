# main.py

import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 从 Excel 文件读取输入数据
file_path = 'input_university_large.xlsx'
df_units = pd.read_excel(file_path, sheet_name='Units')
df_students = pd.read_excel(file_path, sheet_name='Students')
df_rooms = pd.read_excel(file_path, sheet_name='Rooms')

# 确保数据一致性
df_units['ClassType'] = df_units['ClassType'].astype(int)
df_units['ClassID'] = df_units['ClassID'].astype(int)
df_rooms['Type'] = df_rooms['Type'].str.strip().str.title()

valid_room_types = ['Lecture Hall', 'Lab', 'Room']
if not df_rooms['Type'].isin(valid_room_types).all():
    raise ValueError("df_rooms['Type'] contains invalid values.")

# 创建单位列表，包括单元、班级类型和班级 ID
units = df_units[['Unit', 'ClassType', 'ClassID']].values.tolist()
teachers = df_units['Teacher'].tolist()
capacities = df_units['Capacity'].tolist()

# 定义天数和时间段
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
time_slots = list(range(1, 13))  # 假设每天有 12 个时间段
rooms = df_rooms['Room'].tolist()

# 创建快速查找的映射
unit_class_to_index = {tuple(units[i]): i for i in range(len(units))}

# 处理学生数据
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

# 预处理：创建教室类型和容量映射
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

# 创建染色体，考虑讲座占用连续两个时间段
def create_chromosome(num_units, num_days, num_time_slots, units, rooms, room_types):
    chromosome = []
    for i in range(num_units):
        unit, class_type, class_id = units[i]
        day_idx = random.randint(0, num_days - 1)
        if class_type == 1:  # 讲座
            time_slot_idx = random.randint(0, num_time_slots - 2)  # 确保有足够的时间段
        else:
            time_slot_idx = random.randint(0, num_time_slots - 1)
        # 选择符合班级类型的教室
        allowed_room_types = get_allowed_room_types(class_type)
        possible_rooms = [room for room in rooms if room_types[room] in allowed_room_types]
        room = random.choice(possible_rooms)
        chromosome.append((day_idx, time_slot_idx, room))
    return chromosome

# 贪心初始化以获得更好的起始解
def greedy_initialization(units, teachers, days, time_slots, rooms, room_types, students):
    num_units = len(units)
    num_days = len(days)
    num_time_slots = len(time_slots)
    chromosome = [None] * num_units
    teacher_schedule = {}
    room_schedule = {}
    student_schedule = {}

    # 预先计算 class_students 映射
    class_students = {}
    for student, classes in students.items():
        for key in classes:
            if key not in class_students:
                class_students[key] = set()
            class_students[key].add(student)

    # 根据班级类型对单位进行排序（先讲座）
    units_sorted = sorted(range(num_units), key=lambda i: units[i][1])

    for i in units_sorted:
        unit, class_type, class_id = units[i]
        teacher = teachers[i]
        allowed_room_types = get_allowed_room_types(class_type)
        possible_rooms = [room for room in rooms if room_types[room] in allowed_room_types]
        random.shuffle(possible_rooms)

        # 处理讲座占用连续两个时间段
        if class_type == 1:
            duration = 2
        else:
            duration = 1

        assigned = False
        for day_idx in range(num_days):
            for time_slot_idx in range(num_time_slots - (duration - 1)):
                time_slots_needed = [time_slot_idx + d for d in range(duration)]
                # 检查教师可用性
                if pd.notnull(teacher) and teacher != 'None':
                    teacher_busy = any((teacher, day_idx, ts) in teacher_schedule for ts in time_slots_needed)
                else:
                    teacher_busy = False
                if teacher_busy:
                    continue
                # 检查学生冲突
                students_in_class = class_students.get((unit, class_type, class_id), set())
                student_conflict = False
                for student in students_in_class:
                    if any((day_idx, ts) in student_schedule.get(student, set()) for ts in time_slots_needed):
                        student_conflict = True
                        break
                if student_conflict:
                    continue
                # 尝试找到一个教室
                for room in possible_rooms:
                    room_busy = any((room, day_idx, ts) in room_schedule for ts in time_slots_needed)
                    if room_busy:
                        continue
                    # 分配班级
                    chromosome[i] = (day_idx, time_slot_idx, room)
                    # 标记教师为忙碌
                    if pd.notnull(teacher) and teacher != 'None':
                        for ts in time_slots_needed:
                            teacher_schedule[(teacher, day_idx, ts)] = True
                    # 标记教室为忙碌
                    for ts in time_slots_needed:
                        room_schedule[(room, day_idx, ts)] = True
                    # 标记学生为忙碌
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
            # 如果未分配，随机分配（后备方案）
            day_idx = random.randint(0, num_days - 1)
            if class_type == 1:
                time_slot_idx = random.randint(0, num_time_slots - 2)
            else:
                time_slot_idx = random.randint(0, num_time_slots - 1)
            room = random.choice(possible_rooms)
            chromosome[i] = (day_idx, time_slot_idx, room)
    return chromosome

# 使用贪心和随机染色体混合初始化种群
def initialize_population(units, teachers, days, time_slots, df_rooms, rooms, room_types, students, POPULATION_SIZE):
    population = []
    num_greedy = int(POPULATION_SIZE * 0.2)  # 20% 贪心初始化
    num_random = POPULATION_SIZE - num_greedy

    # 贪心初始化
    for _ in range(num_greedy):
        chromosome = greedy_initialization(units, teachers, days, time_slots, rooms, room_types, students)
        population.append(chromosome)

    # 随机初始化
    for _ in range(num_random):
        chromosome = create_chromosome(len(units), len(days), len(time_slots), units, rooms, room_types)
        population.append(chromosome)
    return population

# 适应度函数
def fitness(chromosome, units, teachers, days, time_slots, rooms, students,
            room_capacities, room_types, unit_class_to_index, HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY):
    score = 10000  # 调整后的最大得分
    hard_constraint_violations = 0
    soft_constraint_violations = 0

    teacher_time_slots = set()
    room_time_slots = set()
    student_time_slots = {}
    unit_times = {}

    num_time_slots = len(time_slots)

    # 预先计算 class_students 映射
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
        capacity_required = None  # 将根据教室容量进行检查

        # 检查教室类型约束
        allowed_room_types = get_allowed_room_types(class_type)
        room_type = room_types[room]
        if room_type not in allowed_room_types:
            score -= HARD_CONSTRAINT_PENALTY
            hard_constraint_violations += 1

        # 检查教室容量
        room_capacity = room_capacities[room]
        if class_type == 1:
            # 对于讲座，容量是参加讲座的学生总数
            capacity_required = len(class_students.get((unit, class_type, class_id), []))
            if capacity_required > room_capacity:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
        else:
            # 对于实践和实验，容量由教室容量决定
            capacity_required = room_capacity  # 容量实际上是教室容量

        # 处理讲座占用连续两个时间段
        if class_type == 1:  # 讲座
            if time_slot_idx + 1 >= num_time_slots:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
                continue
            time_slots_needed = [time_slot_idx, time_slot_idx + 1]
        else:
            time_slots_needed = [time_slot_idx]

        # 检查教师时间冲突（仅针对讲座）
        if pd.notnull(teacher) and teacher != 'None':
            for ts in time_slots_needed:
                teacher_time_slot_key = (teacher, day_idx, ts)
                if teacher_time_slot_key in teacher_time_slots:
                    score -= HARD_CONSTRAINT_PENALTY
                    hard_constraint_violations += 1
                else:
                    teacher_time_slots.add(teacher_time_slot_key)

        # 检查教室时间冲突
        for ts in time_slots_needed:
            room_time_slot_key = (room, day_idx, ts)
            if room_time_slot_key in room_time_slots:
                score -= HARD_CONSTRAINT_PENALTY
                hard_constraint_violations += 1
            else:
                room_time_slots.add(room_time_slot_key)

        # 记录单位时间以进行软约束检查
        unit_times[(unit, class_type, class_id)] = (day_idx, time_slot_idx)

    # 学生冲突
    for student, class_keys in students.items():
        if student not in student_time_slots:
            student_time_slots[student] = set()
        for key in class_keys:
            if key in unit_class_to_index:
                course_idx = unit_class_to_index[key]
                day_idx, time_slot_idx, room = chromosome[course_idx]
                class_type = units[course_idx][1]
                # 处理讲座占用两个时间段
                if class_type == 1:
                    time_slots_needed = [time_slot_idx, time_slot_idx + 1]
                else:
                    time_slots_needed = [time_slot_idx]
                for ts in time_slots_needed:
                    student_time_slot = (day_idx, ts)
                    if student_time_slot in student_time_slots[student]:
                        # 学生时间冲突
                        score -= HARD_CONSTRAINT_PENALTY
                        hard_constraint_violations += 1
                    else:
                        student_time_slots[student].add(student_time_slot)

    # 软约束
    # 1. 实践和实验应安排在讲座之后
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

    # 2. 学生的课程应分布在一周内
    for student, times in student_time_slots.items():
        days_with_classes = set([t[0] for t in times])
        if len(days_with_classes) <= 2:
            score -= SOFT_CONSTRAINT_PENALTY
            soft_constraint_violations += 1

    # 3. 避免一天内过多的连续课程
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
                    if consecutive_classes > 4:  # 超过 4 个连续课程
                        score -= SOFT_CONSTRAINT_PENALTY
                        soft_constraint_violations += 1
                        break
                else:
                    consecutive_classes = 1

    return score, hard_constraint_violations, soft_constraint_violations, {}

# 交叉函数
def crossover(parent1, parent2, CROSSOVER_RATE):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异函数，考虑讲座时间段
def mutate(chromosome, num_days, num_time_slots, units, rooms, room_types, MUTATION_RATE):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            unit, class_type, class_id = units[i]
            day_idx = random.randint(0, num_days - 1)
            if class_type == 1:  # 讲座
                time_slot_idx = random.randint(0, num_time_slots - 2)
            else:
                time_slot_idx = random.randint(0, num_time_slots - 1)
            # 选择符合班级类型的教室
            allowed_room_types = get_allowed_room_types(class_type)
            possible_rooms = [room for room in rooms if room_types[room] in allowed_room_types]
            room = random.choice(possible_rooms)
            chromosome[i] = (day_idx, time_slot_idx, room)

# 锦标赛选择函数
def tournament_selection(population, fitness_values):
    tournament_size = 7
    selected = random.sample(list(zip(population, fitness_values)), tournament_size)
    selected.sort(key=lambda x: x[1][0], reverse=True)
    return selected[0][0]

# 遗传算法主函数
def genetic_algorithm(units, teachers, days, time_slots, rooms, students, df_rooms,
                      POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE, PATIENCE,
                      HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY):
    num_units = len(units)
    num_days = len(days)
    num_time_slots = len(time_slots)

    # 预处理数据
    unit_class_to_index = {tuple(units[i]): i for i in range(len(units))}
    room_capacities = df_rooms.set_index('Room')['Capacity'].to_dict()
    room_types = df_rooms.set_index('Room')['Type'].to_dict()

    # 初始化种群
    population = initialize_population(units, teachers, days, time_slots, df_rooms, rooms, room_types, students, POPULATION_SIZE)

    # 保存初始时间表
    initial_chromosome = population[0]
    initial_schedule_df = chromosome_to_schedule(initial_chromosome, units, teachers, days, time_slots)

    # 计算初始适应度和违背
    initial_fitness, _, _, initial_time_slot_violations = fitness(
        initial_chromosome, units, teachers, days, time_slots, rooms, students,
        room_capacities, room_types, unit_class_to_index, HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY)

    best_fitness = None
    generations_without_improvement = 0
    fitness_history = []

    # 记录最佳染色体的违背
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

        # 提取适应度得分
        fitness_scores = [fv[0] for fv in fitness_values]

        # 根据适应度对种群进行排序
        sorted_population = [
            x for _, x in sorted(
                zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True
            )
        ]

        # 实施精英保留
        num_elites = int(POPULATION_SIZE * ELITISM_RATE)
        elites = sorted_population[:num_elites]

        # 检查新的最佳解
        current_best_fitness = fitness_scores[fitness_scores.index(max(fitness_scores))]
        fitness_history.append(current_best_fitness)

        if best_fitness is None or current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = sorted_population[0]
            best_index = population.index(best_chromosome)
            # 保存最佳染色体的违背
            best_time_slot_violations = fitness_values[best_index][3]
            hard_constraints = fitness_values[best_index][1]
            soft_constraints = fitness_values[best_index][2]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # 提前停止，如果没有改进
        if generations_without_improvement >= PATIENCE:
            logging.info("早停：适应度没有改进")
            break

        # 生成新种群
        new_population = elites.copy()

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            mutate(child1, num_days, num_time_slots, units, rooms, room_types, MUTATION_RATE)
            mutate(child2, num_days, num_time_slots, units, rooms, room_types, MUTATION_RATE)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

    # 返回结果
    return (
        best_chromosome, best_fitness, hard_constraints, soft_constraints,
        initial_schedule_df, {}, best_time_slot_violations
    )

# 将染色体转换为时间表 DataFrame
def chromosome_to_schedule(chromosome, units, teachers, days, time_slots):
    schedule = []

    for unit_idx, (day_idx, time_slot_idx, room) in enumerate(chromosome):
        day = days[day_idx]
        time_slot = time_slots[time_slot_idx]
        unit, class_type, class_id = units[unit_idx]
        teacher = teachers[unit_idx]
        if class_type == 1:
            duration = 2  # 讲座占用两个时间段
        else:
            duration = 1
        schedule.append((day, time_slot, duration, unit, class_type, class_id, teacher, room))
        # 如果是讲座，添加下一个时间段
        if class_type == 1:
            schedule.append((day, time_slot + 1, 0, unit, class_type, class_id, teacher, room))

    df_schedule = pd.DataFrame(schedule, columns=['Day', 'Time Slot', 'Duration', 'Unit', 'Class Type', 'ClassID', 'Teacher', 'Room'])
    df_schedule['Class Type'] = df_schedule['Class Type'].map({1: 'Lecture', 2: 'Practical', 3: 'Lab'})
    df_schedule['Teacher'] = df_schedule['Teacher'].fillna('No Teacher')
    return df_schedule

# 显示最终时间表
def display_schedule(chromosome, units, teachers, days, time_slots, rooms, students, best_fitness, hard_constraints, soft_constraints):
    schedule_df = chromosome_to_schedule(chromosome, units, teachers, days, time_slots)

    print(f'\nBest Fitness: {best_fitness}')
    print(f'Hard Constraint Violations: {hard_constraints}')
    print(f'Soft Constraint Violations: {soft_constraints}')

    return schedule_df

if __name__ == '__main__':
    # 定义要测试的种群规模范围
    population_sizes = list(range(20, 201, 10))  # 从20到200，每次增加20

    results = []

    # 固定参数
    GENERATIONS = 200
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.9
    ELITISM_RATE = 0.1
    PATIENCE = 30
    HARD_CONSTRAINT_PENALTY = 500
    SOFT_CONSTRAINT_PENALTY = 10

    # 用于绘图的数据
    population_size_list = []
    avg_fitness_list = []
    avg_execution_time_list = []

    for POPULATION_SIZE in population_sizes:
        fitness_values = []
        execution_times = []
        for run in range(10):  # 每个种群规模下运行10次
            start_time = time.time()
            (best_schedule, best_fitness, hard_constraints, soft_constraints,
             initial_schedule_df, initial_time_slot_violations, best_time_slot_violations) = genetic_algorithm(
                units, teachers, days, time_slots, rooms, students, df_rooms,
                POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITISM_RATE, PATIENCE,
                HARD_CONSTRAINT_PENALTY, SOFT_CONSTRAINT_PENALTY)
            end_time = time.time()
            execution_time = end_time - start_time

            fitness_values.append(best_fitness)
            execution_times.append(execution_time)

            print(f"Population Size: {POPULATION_SIZE}, Run {run + 1}/10, Best Fitness: {best_fitness}, Execution Time: {execution_time:.2f}s")

        # 计算平均值
        avg_fitness = np.mean(fitness_values)
        avg_execution_time = np.mean(execution_times)

        population_size_list.append(POPULATION_SIZE)
        avg_fitness_list.append(avg_fitness)
        avg_execution_time_list.append(avg_execution_time)

        # 记录结果
        results.append({
            'Population Size': POPULATION_SIZE,
            'Average Best Fitness': avg_fitness,
            'Average Execution Time': avg_execution_time
        })

    # 将结果保存到 CSV 文件
    df_results = pd.DataFrame(results)
    df_results.to_csv('population_size_results.csv', index=False)

    # 绘制双 Y 轴图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Average Execution Time (s)', color=color)
    ax1.plot(population_size_list, avg_execution_time_list, color=color, marker='o', label='Execution Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Population Size vs Average Execution Time and Best Fitness')
    ax1.grid(True)

    ax2 = ax1.twinx()  # 共享 x 轴

    color = 'tab:blue'
    ax2.set_ylabel('Average Best Fitness', color=color)
    ax2.plot(population_size_list, avg_fitness_list, color=color, marker='s', label='Best Fitness')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

    plt.tight_layout()
    plt.savefig('population_size_vs_execution_time_and_fitness.png')
    plt.show()
