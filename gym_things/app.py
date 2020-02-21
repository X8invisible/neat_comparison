import gym_Pole as pole
import gym_MCar as cart
import math
import datetime
def startEnv(choice, gens, neat):
    if choice == "1":
        score = pole.start(gens, neat)

    if choice == "2":
        score = cart.start(gens, neat)
    return score

def run():
    options = ["1","2","3","4"]
    choice = "7"
    while choice not in options:
        print("1. Pole Balancing")
        print("2. Mountain Cart")
        print("3. Pendulum")
        print("4. Lunar Lander")
        choice = input("Choose environment:")
    filename = "data/" + str(choice) +" " + datetime.datetime.now().strftime("%m-%d %H.%M.%S") +".txt"
    testing = "f"
    while testing not in ["y", "n"]:
       testing =  input("Do you want to run testing mode (does 10 tests)? [y/n]: ")
    neat = "f"
    while neat not in ["y", "n"]:
       neat =  input("Do you want to run with neat or not? [y/n]: ")
    gens = int(input("How many generations to train?"))
    if testing == "y":
        file = open(filename, "w")
        file.write(str(choice)+"\n")
        scores = []
        for x in range(10):
            score = startEnv(choice, gens, neat)
            lineToWrite = str(x+1) + " " + str(score) + "\n"
            file.write(lineToWrite)
            scores.append(score)
        avg = sum(scores)/10

        print("Average over 10 trainings: ",avg)

        sd = 0
        mean = sum(scores) / len(scores)
        for p in scores:
            sd += (mean - p) * (mean - p)
        sd /= len(scores)
        sd = math.sqrt(sd)
        print("standard dev: ", sd)
        file.write("average: "+ str(avg)+ "\nstandard dev: "+ str(sd))
        file.close()
    else:
        startEnv(choice, gens, neat)


   


    


if __name__ == '__main__':
   run()