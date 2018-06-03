import time
import numpy as np
import random
import matplotlib.pyplot as plt

# Settings
numPoints = 3
alpha = 1
beta = 1
minMaxX = [0.01, 5]

# Break Conditions
eBreak = 1e-10
maxIterations = 10000
globalBestDidNotChangeFor = 0
maxGlobalBestDidNotChangeFor = 10

fitnessGlobal = 0
fitnessGlobalOld = 0.1
bestXGlobal = 0


# Function to search global maximum in
def fx(x):
    return x**(3-x)


# Fitness function
def fitness(x):
    return fx(x)


# Condition to check for better fitness, if looking for mininum this has to be inverted
def new_fitness_better(newFitness, oldFitness):
    return newFitness > oldFitness


if __name__ == "__main__":
    # Create random starting points, init starting parameters
    points = np.array([[p, fx(p)] for p in [random.uniform(minMaxX[0], minMaxX[1]) for i in range(numPoints)]])
    print("Init ", end='')
    for p in points:
        print("& (%.3f, %.3f)" %(p[0], p[1]), end='')
    print("\\\\ \\hline")
    velocities = np.zeros(numPoints)
    bestFitnessValues = np.zeros(numPoints)
    bestX = np.zeros(numPoints)

    # Calculate reference graph
    xValues = np.arange(minMaxX[0], minMaxX[1], 0.1)
    yValues = fx(xValues)

    currentIteration = 0
    for currentIteration in range(maxIterations):
        for p in range(numPoints):
            f = fitness(points[p][0])

            if new_fitness_better(f, bestFitnessValues[p]):
                bestFitnessValues[p] = f
                bestX[p] = points[p][0]

                if new_fitness_better(f, fitness(bestXGlobal)):
                    bestXGlobal = bestX[p]
                    fitnessGlobal = f

            v = velocities[p] + alpha * random.random() * (bestXGlobal - points[p][0]) + beta * random.random() * (bestX[p] - points[p][0])
            velocities[p] = v
            points[p][0] = points[p][0] + v

            if points[p][0] < minMaxX[0]:
                points[p][0] = minMaxX[0]
            elif points[p][0] > minMaxX[1]:
                points[p][0] = minMaxX[1]

            points[p][1] = fx(points[p][0])

        print("%d " % (currentIteration), end='')
        for p in points:
            print("& (%.3f, %.3f)" %(p[0], p[1]), end='')
        print("\\\\ \\hline")

        # Plot everything
        plt.clf()
        plt.plot(points[:, 0], points[:, 1], 'ro')
        plt.plot(xValues, yValues)
        plt.plot([bestXGlobal], [fx(bestXGlobal)], 'bo')
        plt.pause(0.05)

        # Break if global best hasn't changed for a couple of rounds
        if abs(fitnessGlobal - fitnessGlobalOld) < eBreak:
            globalBestDidNotChangeFor += 1
        else:
            globalBestDidNotChangeFor = 0
        if globalBestDidNotChangeFor >= maxGlobalBestDidNotChangeFor:
            print("Stop due to global fitness not improving anymore")
            break
        fitnessGlobalOld = fitnessGlobal

    print("Best point: (%f, %f); Needed iterations: %d" % (bestXGlobal, fx(bestXGlobal), currentIteration))
    plt.show()
