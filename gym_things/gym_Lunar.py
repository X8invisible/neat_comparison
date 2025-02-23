import gym
import os
import neat
import numpy as np


env = gym.make('LunarLander-v2')
print(env.observation_space)
print(env.action_space)
env.reset()

def highestVal(vals):
    index = 0
    for x in range(0,len(vals)):
        if vals[x] > vals[index]:
            index = x
    return index

def distanceToSolution(x,y):
    #euclidean distance to (0,0) (where our landing pad is)
    return np.sqrt( x**2 + y**2 )
def eval_genomes(genomes, config):

    distance = 10000
    for _, genome in genomes:
        observation = env.reset()  #Inital observation
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, config) #Create net for genome with configs
        genome.fitness = 0  #Starting fitness of 0
        while not done:
            while not done:
                action = net.activate(observation)
                action = highestVal(action)
                observation, reward, done, info = env.step(action)
                currDistance = distanceToSolution(observation[0], observation[1])
                if(currDistance > distance):
                    genome.fitness +=1
                    distance = currDistance
                else:
                    genome.fitness -=1
                if done:
                    #if solved, reward gives 200
                    genome.fitness += reward
                    break
        env.reset()

def run(config_file):
    print(env.action_space)
    print(env.observation_space)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 30)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #test winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    test_model(winner_net)  #Tests model 100 times and prints result

def test_model(winner):
    input()
    observation = env.reset()
    score = 0
    reward = 0
    for i in range(100):
        done = False
        observation = env.reset()
        t = 0
        while not done:
            t=t+1
            env.render()
            action = winner.activate(observation)
            action = highestVal(action)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        env.reset()
    print("Score Over 100 tries:")
    print(score/100)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-gymLunar')
    run(config_path)