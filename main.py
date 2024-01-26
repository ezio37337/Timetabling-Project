import random
import time
import pandas as pd

# Define Global Variables
populationSize = 5
maxGenerations = 500
crossoverRate = 0.8
mutationRate = 0.2
tournamentSize = 5  
baseScore = 100
highPenalty = 20
lowPenalty = 5
flag = 1000

# courses = ["Course1", "Course2", "Course3","Course4","Course5","Course6"]  
# rooms = ["Room1","Room2","Room3","Room4"]  
# timeslots = ["Time1", "Time2", "Time3","Time4","Time5"] 

csv_path = "./schedule.csv"
file = pd.read_csv(csv_path)
courses = list(file.iloc[0][1:].dropna())
rooms = list(file.iloc[1][1:].dropna())
timeslots = list(file.iloc[2][1:].dropna())

# for complex test
# courses = ["Course" + str(i) for i in range(1, 7)]  
# rooms = ["Room" + str(i) for i in range(1, 21)]  
# timeslots = ["Time" + str(i) for i in range(1,7)]  


# hard cons
# Check if all courses are scheduled
def all_courses_scheduled(timetable):
    # Create a set to store the courses found in the timetable
    scheduled_courses = set()
    
    # Check each slot in the timetable
    for (room, timeslot), course in timetable['slots'].items():
        # Add the course to the set of scheduled courses
        scheduled_courses.add(course)
    
    # Check if all courses have been scheduled
    for course in courses:
        if course not in scheduled_courses:
            # If a course is missing, return True (violated)
            return True
    
    # If all courses are scheduled, return False (not violated)
    return False

# hard cons
# Course schedule uniqueness: Each course can only be scheduled once in the timetable.
def course_scheduled_once_violated(timetable):
    # Create a dictionary to store the count of each course
    course_count = {}
    
    # Check each slot in the timetable
    for (room, timeslot), course in timetable['slots'].items():
        # If the course is not in the dictionary, initialize its count to 1
        if course not in course_count:
            course_count[course] = 1
        else:
            # If the course is already in the dictionary, increment its count
            course_count[course] += 1
    
    # Check if any course has been scheduled more than once
    for count in course_count.values():
        if count > 1:
            # If a course is scheduled more than once, return True (violated)
            return True
    
    # If no violations were found, return False
    return False

# hard cons
# Check if the same course is scheduled in the same time slot in the timetable
def same_course_in_multiple_slots_violated(timetable):
    # Create a dictionary to store the timeslots for each course
    course_timeslots = {}
    
    # Check each slot in the timetable
    for (room, timeslot), course in timetable['slots'].items():
        # If the course is not in the dictionary, add it with a set containing the current timeslot
        if course not in course_timeslots:
            course_timeslots[course] = {timeslot}
        else:
            # If the course is already in the dictionary, check if the timeslot is already in the set
            if timeslot in course_timeslots[course]:
                # If the course is already in the same timeslot, return True (violated)
                return True
            else:
                # Otherwise, add the timeslot to the set for this course
                course_timeslots[course].add(timeslot)
    
    # If no violations were found, return False
    return False

# hard cons
# Check if any classrooms are being reused in the same time slot
def room_availability_violated(timetable):
    # Create a set to store occupied slots
    occupied_slots = set()
    
    # Check each slot in the timetable
    for (room, timeslot), course in timetable['slots'].items():
        if (room, timeslot) in occupied_slots:
            # If the room is already occupied in this timeslot, return True (violated)
            return True
        else:
            # Otherwise, add the slot to the occupied set
            occupied_slots.add((room, timeslot))
    
    # If no violations were found, return False
    return False

# soft constraits
# Check the schedule timetable to see if any courses are scheduled in consecutive time slots without intervening breaks.
def breaks_between_lectures_violated(timetable):
    # Create a dictionary to store the timeslots for each course
    course_timeslots = {}
    
    # Populate the course_timeslots dictionary
    for (room, timeslot), course in timetable['slots'].items():
        if course in course_timeslots:
            course_timeslots[course].append(timeslot)
        else:
            course_timeslots[course] = [timeslot]
    
    # Check for each course if there are consecutive timeslots
    for timeslots in course_timeslots.values():
        # Sort the timeslots for this course
        timeslots.sort()
        
        for i in range(len(timeslots) - 1):
            # If two consecutive timeslots are found, return True (violated)
            if timeslots.index(timeslots[i]) + 1 == timeslots.index(timeslots[i + 1]):
                return True
    
    # If no violations were found, return False
    return False

# soft cons
# Balanced use of classrooms: try to distribute the use of classrooms as much as possible
def classroom_usage_balance_violated(timetable):
    # Create a dictionary to store the count of usage for each room
    room_usage_count = {room: 0 for room in rooms}
    
    # Check each slot in the timetable
    for (room, timeslot), course in timetable['slots'].items():
        # Increment the count for the room used
        room_usage_count[room] += 1
    
    # Calculate the average usage of a room
    average_usage = sum(room_usage_count.values()) / len(room_usage_count)
    
    # Calculate the standard deviation of the room usage
    variance = sum((usage - average_usage) ** 2 for usage in room_usage_count.values()) / len(room_usage_count)
    std_deviation = variance ** 0.5
    
    # Define a threshold for acceptable standard deviation
    threshold = 0.5  # This can be adjusted based on the specific requirements
    
    # Check if the standard deviation of room usage exceeds the threshold
    if std_deviation > threshold:
        # If exceeded, return the standard deviation as a penalty score (violated)
        return std_deviation * 5
    
    # If the standard deviation is within the acceptable range, return 0 (not violated)
    return 0

def GeneticAlgorithmForTimetabling():
    population = InitializePopulation(populationSize)
    EvaluateFitnessOfAll(population)

    for generation in range(maxGenerations):
        new_population = []

        while len(new_population) < len(population):
            parent1 = TournamentSelection(population, tournamentSize)
            parent2 = TournamentSelection(population, tournamentSize)

            if random.random() < crossoverRate:
                child1, child2 = OnePointCrossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = UniformMutate(child1)
            child1 = UniformMutate(child1)

            new_population.extend([child1, child2])

        population = new_population
        EvaluateFitnessOfAll(population)

    best_timetable = SelectBestFromPopulation(population)
    return best_timetable


def InitializePopulation(size):
    population = []
    for i in range(size):
        timetable = GenerateRandomTimetable()
        population.append(timetable)
    return population


def EvaluateFitnessOfAll(population):
    for timetable in population:
        timetable["fitness"] = CalculateFitness(timetable)


def GenerateRandomTimetable():
    timetable = {"slots": {}, "fitness": 0}
    for course in courses:
        randomRoom = random.choice(rooms)
        randomTimeSlot = random.choice(timeslots)

        while (randomRoom, randomTimeSlot) in timetable["slots"]:
            randomRoom = random.choice(rooms)
            randomTimeSlot = random.choice(timeslots)
        timetable["slots"][(randomRoom, randomTimeSlot)] = course
    return timetable


def CalculateFitness(timetable):
    fitness = baseScore

    # hard
    if all_courses_scheduled(timetable):
        fitness -= highPenalty

    if room_availability_violated(timetable):
        fitness -= highPenalty
    
    if same_course_in_multiple_slots_violated(timetable):
        fitness -= highPenalty

    if course_scheduled_once_violated(timetable):
        fitness -= highPenalty

    # soft 
    if breaks_between_lectures_violated(timetable):
        fitness -= lowPenalty

    if classroom_usage_balance_violated(timetable):
        fitness -= classroom_usage_balance_violated(timetable)

    return fitness


def TournamentSelection(population, tournamentSize):
    best = random.choice(population)
    for i in range(tournamentSize - 1):
        individual = random.choice(population)
        if individual["fitness"] > best["fitness"]:
            best = individual
    return best

def OnePointCrossover(parent1, parent2):
    # one-point crossover
    crossoverPoint = random.randint(0, len(courses) - 1) 
    child1 = {"slots": {}, "fitness": 0}
    child2 = {"slots": {}, "fitness": 0}

    items1 = list(parent1["slots"].items())
    items2 = list(parent2["slots"].items())

    child1["slots"] = dict(items1[:crossoverPoint] + items2[crossoverPoint:])
    child2["slots"] = dict(items2[:crossoverPoint] + items1[crossoverPoint:])
    # check if child's length < len(courses), crossover again
    flag = 1
    while (len(child1["slots"]) < len(courses) or len(child2["slots"]) < len(courses)) and flag < 1000:
        crossoverPoint = random.randint(0, len(courses) - 1)
        child1["slots"] = dict(items1[:crossoverPoint] + items2[crossoverPoint:])
        child2["slots"] = dict(items2[:crossoverPoint] + items1[crossoverPoint:])
        flag += 1
    return child1, child2

def TwoPointCrossover(parent1, parent2):
    # two-point crossover
    # Ensure that the first crossover point is less than the second one
    crossoverPoint1 = random.randint(0, len(courses) - 2)
    crossoverPoint2 = random.randint(crossoverPoint1 + 1, len(courses) - 1)
    
    child1 = {"slots": {}, "fitness": 0}
    child2 = {"slots": {}, "fitness": 0}

    items1 = list(parent1["slots"].items())
    items2 = list(parent2["slots"].items())

    # Swap the gene segments between the two crossover points
    child1["slots"] = dict(items1[:crossoverPoint1] + items2[crossoverPoint1:crossoverPoint2] + items1[crossoverPoint2:])
    child2["slots"] = dict(items2[:crossoverPoint1] + items1[crossoverPoint1:crossoverPoint2] + items2[crossoverPoint2:])
    # check if child's length < len(courses), crossover again
    flag = 1
    while (len(child1["slots"]) < len(courses) or len(child2["slots"]) < len(courses)) and flag < 1000:
        crossoverPoint1 = random.randint(0, len(courses) - 2)
        crossoverPoint2 = random.randint(crossoverPoint1 + 1, len(courses) - 1)
        child1["slots"] = dict(items1[:crossoverPoint1] + items2[crossoverPoint1:crossoverPoint2] + items1[crossoverPoint2:])
        child2["slots"] = dict(items2[:crossoverPoint1] + items1[crossoverPoint1:crossoverPoint2] + items2[crossoverPoint2:])
        flag += 1
    return child1, child2

def UniformCrossover(parent1, parent2):
    # Uniform crossover
    child1 = {"slots": {}, "fitness": 0}
    child2 = {"slots": {}, "fitness": 0}

    items1 = list(parent1["slots"].items())
    items2 = list(parent2["slots"].items())

    # For each gene position, randomly choose the gene from one of the parents
    flag = 1
    while (len(child1["slots"]) < len(courses) or len(child2["slots"]) < len(courses)) and flag < 1000:
        for i in range(len(courses)):
            if random.random() < 0.5:
                child1["slots"][items1[i][0]] = items1[i][1]
                child2["slots"][items2[i][0]] = items2[i][1]
            else:
                child1["slots"][items1[i][0]] = items2[i][1]
                child2["slots"][items2[i][0]] = items1[i][1]
        flag += 1
    return child1, child2

# simple mutate, mutate course
def Mutate(timetable):
    for slot in list(timetable["slots"].keys()):
        if random.random() < mutationRate:
            original_course = timetable["slots"][slot]
            newCourse = random.choice(courses)
            timetable["slots"][slot] = newCourse
    return timetable

def UniformMutate(timetable):
    # Create a copy of the keys to avoid modifying the dictionary during iteration
    slots_to_mutate = list(timetable["slots"].keys())
    
    # Iterate over each slot in the timetable
    for slot in slots_to_mutate:
        # Mutate the course
        if random.random() < mutationRate:
            newCourse = random.choice(courses)
            timetable["slots"][slot] = newCourse
        
        # Mutate the room
        if random.random() < mutationRate:
            newRoom = random.choice(rooms)
            newSlot = (newRoom, slot[1])
            # Check if the new slot is already occupied and mutate until it's not
            while newSlot in timetable["slots"] and newSlot != slot:
                newRoom = random.choice(rooms)
                newSlot = (newRoom, slot[1])
            # If the new slot is different, update the slot with the new room
            if newSlot != slot:
                timetable["slots"][newSlot] = timetable["slots"].pop(slot)
                # Update the slot variable to reflect the new key for timeslot mutation
                slot = newSlot
        
        # Mutate the timeslot
        if random.random() < mutationRate:
            newTimeslot = random.choice(timeslots)
            newSlot = (slot[0], newTimeslot)
            # Check if the new slot is already occupied and mutate until it's not
            while newSlot in timetable["slots"] and newSlot != slot:
                newTimeslot = random.choice(timeslots)
                newSlot = (slot[0], newTimeslot)
            # If the new slot is different, update the slot with the new timeslot
            if newSlot != slot:
                timetable["slots"][newSlot] = timetable["slots"].pop(slot)

    return timetable


def SelectBestFromPopulation(population): 
    return max(population, key=lambda x: x["fitness"])


if __name__ == "__main__":
    all_best_timetable= []
    # best_timetable = GeneticAlgorithmForTimetabling()
    # print(best_timetable)

    # eval operator
    start_time = time.time()  # Start timing
    for i in range(1,101):
        best_timetable = GeneticAlgorithmForTimetabling()
        all_best_timetable.append(best_timetable)
    end_time = time.time()  # End timing
    total_time = end_time - start_time  # Calculate total runtime
    print(all_best_timetable[0])
     # Calculate the average fitness score from all best timetables
    average_fitness = sum(timetable["fitness"] for timetable in all_best_timetable) / len(all_best_timetable)
    print("Run GA with OnePointCrossover + UniformMutate for 100 times")
    print("Average Fitness Score:", average_fitness)
    print("Total Runtime:", total_time, "seconds")