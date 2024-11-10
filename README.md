# University Timetabling Problem

## Overview
This project implements a University Course Scheduling System using a Genetic Algorithm (GA). The main objective is to generate optimized course schedules based on various constraints such as room capacity, course type, and student conflicts. The project provides multiple testing scripts to evaluate different parameters and configurations of the Genetic Algorithm.

## Features
- Supports scheduling for multiple course types: Lecture, Practical, and Lab.
- Incorporates constraints for room capacity and type (e.g., Lecture Halls, Lab Rooms).
- Optimizes schedules by minimizing student conflicts and improving room utilization.
- Includes various test scripts for analyzing the impact of algorithm parameters such as population size, mutation rate, crossover rate, and elitism.

## Repository Structure
The repository includes the following files:

- **Main Scripts:**
  - **Main.py**: Main program file implementing the Genetic Algorithm for course scheduling.
  - **Dataset creation.py**: Script for generating synthetic datasets with students, courses, teachers, and rooms.

- **Testing Scripts:**
  - **Average fitness test.py**: Evaluates the average fitness of the population over multiple generations.
  - **Crossover&Mutation rate test.py**: Analyzes the impact of different crossover and mutation rates.
  - **Elitism test.py**: Tests the impact of elitism on the algorithm's performance.
  - **Greed initialization test.py**: Tests the effect of using a greedy initialization strategy.
  - **Population size test.py**: Evaluates the effect of varying the population size.
  - **Tournament size test.py**: Analyzes the impact of different tournament sizes on the selection process.
  - **Heatmap.py**: Generates heatmaps for visualizing the performance metrics.

- **Datasets:**
  - **input_university_large.xlsx**: A comprehensive dataset for large-scale testing, including detailed information on courses, students, rooms, and teachers.
  - **input_university_testcase.xlsx**: A smaller, simplified dataset for initial testing and debugging purposes.

## Prerequisites
This project requires Python 3 and the following Python libraries:

- `numpy`
- `matplotlib`
- `pandas`
- `openpyxl`

You can install the required dependencies with the following command:
```bash
pip install numpy matplotlib pandas openpyxl
