# generate_dataset.py

import pandas as pd
import numpy as np
import random
import math

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# 1. Generate Units (Courses)
def generate_units(num_units):
    units = []
    prefixes = ["Math", "Phys", "Comp", "Chem", "Biol", "Eng", "Hist", "Phil", "Econ", "Psych"]
    unit_numbers = list(range(1001, 2000))  # Expanded range to accommodate more units
    random.shuffle(unit_numbers)
    unit_names = []
    prefix_index = 0

    while len(unit_names) < num_units:
        prefix = prefixes[prefix_index % len(prefixes)]
        number = unit_numbers.pop() if unit_numbers else random.randint(2000, 9999)
        unit_name = f"{prefix}{number}"
        if unit_name not in unit_names:
            unit_names.append(unit_name)
        prefix_index += 1

    units_list = []
    teacher_ids = [f"Teacher_{i}" for i in range(1, 101)]  # Increased to 100 teachers
    teacher_index = 0

    for unit in unit_names:
        # Assign a teacher to each Lecture
        teacher = teacher_ids[teacher_index % len(teacher_ids)]
        teacher_index += 1

        # Create Lecture class
        units_list.append({
            'Unit': unit,
            'ClassType': 1,
            'ClassID': 1,  # Only one Lecture per unit
            'Teacher': teacher,
            'Capacity': None  # Will be updated later
        })

        # Practical and Lab classes will be added later
    df_units = pd.DataFrame(units_list)
    return df_units, unit_names

# 2. Generate Students
def generate_students(num_students, unit_names):
    students = []
    student_ids = set()
    while len(student_ids) < num_students:
        student_id = str(random.randint(10000000, 99999999))
        student_ids.add(student_id)
    student_ids = list(student_ids)

    student_courses = {}
    for student_id in student_ids:
        num_courses = 4  # Each student takes 4 courses
        selected_units = random.sample(unit_names, num_courses)
        student_courses[student_id] = selected_units
        for unit in selected_units:
            # Students register for Lecture classes
            students.append({
                'StudentID': student_id,
                'Unit': unit,
                'ClassType': 1,
                'ClassID': 1
            })
    df_students = pd.DataFrame(students)
    return df_students, student_courses

# 3. Update Units with Practical and Lab Sessions
def update_units_with_sessions(df_units, df_students, unit_names, student_courses):
    units_list = df_units.to_dict('records')

    practical_room_capacity = 50  # Updated practical room capacity
    lab_room_capacity = 45        # Updated lab room capacity

    for unit in unit_names:
        # Get total number of students enrolled in the unit
        total_students = len(df_students[df_students['Unit'] == unit]['StudentID'].unique())

        # Calculate required number of practical sessions
        num_practical_sessions = max(1, math.ceil(total_students / practical_room_capacity))

        # Create practical sessions
        for i in range(1, num_practical_sessions + 1):
            units_list.append({
                'Unit': unit,
                'ClassType': 2,
                'ClassID': i,
                'Teacher': None,        # Practicals have no teacher
                'Capacity': practical_room_capacity
            })

        # Calculate required number of lab sessions
        num_lab_sessions = max(1, math.ceil(total_students / lab_room_capacity))

        # Create lab sessions
        for i in range(1, num_lab_sessions + 1):
            units_list.append({
                'Unit': unit,
                'ClassType': 3,
                'ClassID': i,
                'Teacher': None,        # Labs have no teacher
                'Capacity': lab_room_capacity
            })

    df_units = pd.DataFrame(units_list)
    return df_units

# 4. Assign Students to Practical and Lab Sessions
def assign_students_to_sessions(df_students, df_units, student_courses):
    students_list = df_students.to_dict('records')
    for student_id, units in student_courses.items():
        for unit in units:
            # Assign student to a practical session
            practical_sessions = df_units[(df_units['Unit'] == unit) & (df_units['ClassType'] == 2)]
            practical_session_ids = practical_sessions['ClassID'].tolist()
            practical_session_id = random.choice(practical_session_ids)
            students_list.append({
                'StudentID': student_id,
                'Unit': unit,
                'ClassType': 2,
                'ClassID': practical_session_id
            })

            # Assign student to a lab session
            lab_sessions = df_units[(df_units['Unit'] == unit) & (df_units['ClassType'] == 3)]
            lab_session_ids = lab_sessions['ClassID'].tolist()
            lab_session_id = random.choice(lab_session_ids)
            students_list.append({
                'StudentID': student_id,
                'Unit': unit,
                'ClassType': 3,
                'ClassID': lab_session_id
            })

    df_students = pd.DataFrame(students_list)
    return df_students

# 5. Update Lecture Capacities
def update_lecture_capacities(df_units, df_students):
    for idx, row in df_units.iterrows():
        if row['ClassType'] == 1:
            unit = row['Unit']
            total_students = len(df_students[(df_students['Unit'] == unit) & (df_students['ClassType'] == 1)])
            df_units.at[idx, 'Capacity'] = total_students + 10  # Add buffer
    return df_units

# 6. Generate Rooms
def generate_rooms():
    rooms = []

    # Lecture Halls
    num_lecture_halls = 30  # Increased number of lecture halls
    for i in range(1, num_lecture_halls + 1):
        capacity = 350  # Increased lecture hall capacity
        rooms.append({
            'Room': f"LectureHall_{i}",
            'Type': 'Lecture Hall',
            'Capacity': capacity
        })

    # Labs
    num_labs = 100  # Increased number of labs
    for i in range(1, num_labs + 1):
        capacity = 45  # Updated lab capacity
        rooms.append({
            'Room': f"Lab_{i}",
            'Type': 'Lab',
            'Capacity': capacity
        })

    # General Rooms
    num_rooms = 100  # Increased number of general rooms
    for i in range(1, num_rooms + 1):
        capacity = 50  # Updated room capacity
        rooms.append({
            'Room': f"Room_{i}",
            'Type': 'Room',
            'Capacity': capacity
        })

    df_rooms = pd.DataFrame(rooms)
    return df_rooms

# 7. Main Function
def main():
    num_units = 80    # Increased number of units
    num_students = 5000  # Increased number of students

    # Generate Units
    df_units, unit_names = generate_units(num_units)

    # Generate Students
    df_students, student_courses = generate_students(num_students, unit_names)

    # Update Units with Practical and Lab Sessions
    df_units = update_units_with_sessions(df_units, df_students, unit_names, student_courses)

    # Assign Students to Practical and Lab Sessions
    df_students = assign_students_to_sessions(df_students, df_units, student_courses)

    # Update Lecture Capacities
    df_units = update_lecture_capacities(df_units, df_students)

    # Generate Rooms
    df_rooms = generate_rooms()

    # Write data to Excel file
    with pd.ExcelWriter('input_university_large.xlsx') as writer:
        df_units.to_excel(writer, sheet_name='Units', index=False)
        df_students.to_excel(writer, sheet_name='Students', index=False)
        df_rooms.to_excel(writer, sheet_name='Rooms', index=False)

    print("Data generation completed. The dataset has been saved to 'input_university_large.xlsx'.")

if __name__ == '__main__':
    main()
