import gym_Pole as pole
import gym_MCar as cart
import gym_MCarCont as cartC
import math
import datetime
import statistics
def startEnv(choice, gens, neat, testing):
    if choice == "1":
        score = pole.start(gens, neat, testing)

    if choice == "2":
        score = cart.start(gens, neat, testing)
    if choice == "3":
        score = cartC.start(gens, neat, testing)
    return score

def run():
    options = ["1","2","3","4"]
    choice = "7"
    print("\n\n")
    print("###############################################################")
    print("# Prototype application for comparing neural network training #")
    print("#      Written by Andrei Radulescu for the Honours Module     #")
    print("###############################################################\n\n\n\n")
    while choice not in options:
        print("1. Pole Balancing")
        print("2. Mountain Cart")
        print("3. Mountain Cart Continuous")
        choice = input("Choose environment: ")
    filename = "data/" + str(choice) +" " + datetime.datetime.now().strftime("%m-%d %H.%M.%S") +".txt"
    testing = "f"
    while testing not in ["y", "n"]:
       testing =  input("Do you want to run testing mode (does 10 tests)? [y/n]: ")
    neat = "f"
    while neat not in ["y", "n"]:
       neat =  input("Do you want to run with neat or not? [y/n]: ")
    gens = int(input("How many generations to train?: "))
    if testing == "y":
        file = open(filename, "w")
        file.write(str(choice)+"\n")
        scores = []
        for x in range(10):
            score = startEnv(choice, gens, neat, testing)
            lineToWrite = str(x+1) + " " + str(score) + "\n"
            file.write(lineToWrite)
            scores.append(score)
        mean = sum(scores) / len(scores)
        median = statistics.median(scores)
        print("Average over 10 trainings: ",mean)
        sd = statistics.stdev(scores)
        print("standard dev: ", sd)
        file.write("generations: %s \nneat: %s \naverage: %s \nstandard dev: %s \nmedian: %s" % (gens, neat, str(mean), str(sd), str(median)))
        file.close()
    else:
        startEnv(choice, gens, neat, testing)



if __name__ == '__main__':
   run()