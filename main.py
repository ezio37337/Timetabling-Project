import random
import pandas as pd

# Define Global Variables
populationSize = 5
maxGenerations = 50
crossoverRate = 0.8
mutationRate = 0.2
tournamentSize = 5
numSlots = 50
baseScore = 100
highPenalty = 20
lowPenalty = 5

csv_path = "./schedule.csv"
file = pd.read_csv(csv_path)
courses = list(file.iloc[0][1:].dropna())
rooms = list(file.iloc[1][1:].dropna())
timeslots = list(file.iloc[2][1:].dropna())


# courses = ["Course1", "Course2", "Course3","Course4","Course5","Course6"]
# rooms = ["Room1","Room2","Room3","Room4"]
# timeslots = ["Time1", "Time2", "Time3","Time4","Time5"]


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


def GeneticAlgorithmForTimetabling():
    population = InitializePopulation(populationSize)
    EvaluateFitnessOfAll(population)

    for generation in range(maxGenerations):
        new_population = []

        while len(new_population) < len(population):
            parent1 = TournamentSelection(population, tournamentSize)
            parent2 = TournamentSelection(population, tournamentSize)

            if random.random() < crossoverRate:
                child1, child2 = Crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = Mutate(child1)
            child2 = Mutate(child2)

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

    ##### check if there are duplicate courses schdule
    # print(timetable["slots"])
    use = timetable["slots"].values()
    if len(set(use)) != len(use):  # have duplicate
        return 0
    #####

    if room_availability_violated(timetable):
        fitness -= highPenalty

    if breaks_between_lectures_violated(timetable):
        fitness -= lowPenalty

    return fitness


def TournamentSelection(population, tournamentSize):
    best = random.choice(population)
    for i in range(tournamentSize - 1):
        individual = random.choice(population)
        if individual["fitness"] > best["fitness"]:
            best = individual
    return best


def Crossover(parent1, parent2):
    crossoverPoint = random.randint(0, numSlots - 1)
    child1 = {"slots": {}, "fitness": 0}
    child2 = {"slots": {}, "fitness": 0}

    items1 = list(parent1["slots"].items())
    items2 = list(parent2["slots"].items())

    child1["slots"] = dict(items1[:crossoverPoint] + items2[crossoverPoint:])
    child2["slots"] = dict(items2[:crossoverPoint] + items1[crossoverPoint:])

    return child1, child2


def Mutate(timetable):
    for slot in list(timetable["slots"].keys()):
        if random.random() < mutationRate:
            original_course = timetable["slots"][slot]
            newCourse = random.choice(courses)
            timetable["slots"][slot] = newCourse

            if room_availability_violated(timetable):
                timetable["slots"][slot] = original_course

    return timetable


def SelectBestFromPopulation(population):
    return max(population, key=lambda x: x["fitness"])


if __name__ == "__main__":
    best_timetable = GeneticAlgorithmForTimetabling()
    print("Best Timetable:", best_timetable)
