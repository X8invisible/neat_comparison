import gym_Pole as pole
import gym_MCar as cart
import math
def startEnv(choice):
    if choice == "1":
        score = pole.start()

    if choice == "2":
        score = cart.start()
    return score

def run():
    print("hey")
    options = ["1","2","3","4"]
    choice = "7"
    while choice not in options:
        print("1. Pole Balancing")
        print("2. Mountain Cart")
        print("3. Pendulum")
        print("4. Lunar Lander")
        choice = input("Choose environment:")

    testing = "f"
    while testing not in ["y", "n"]:
       testing =  input("Do you want to run testing mode (does 10 tests)? [y/n]: ")

    if testing == "y":
        scores = []
        for x in range(10):
            score = startEnv(choice)
            scores.append(score)
        print(scores)
        print("Average over 10 trainings: ", (sum(scores))/10)
        sd = 0
        mean = sum(scores) / len(scores)
        for p in scores:
            sd += (mean - p) * (mean - p)
        sd /= len(scores)
        sd = math.sqrt(sd)
        print("standard dev: ", sd)
    else:
        startEnv(choice)


   


    


if __name__ == '__main__':
   run()