import gym
import os
import neat
import math
env = gym.make('CartPole-v1')
env.reset()
steps = 1000



def eval_genomes(genomes, config):

    for _, genome in genomes:
        observation = [0, 0, 0, 0]  #Inital observation
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, config) #Creat net for genome with configs
        genome.fitness = 0  #Starting fitness of 0
        while not done:
            #env.render()   #Render game   
            output = net.activate(observation)  #Getting output from net based on observations
            action = max(output)
            if output[0] == action:
                action = 1
            else:
                action = 0

            observation, reward, done, info = env.step(action)  #Performs action
            if observation[3] > 0.5 or observation[3] < -0.5:   
                done = True
                reward = -5
                env.reset()
            genome.fitness += reward    #Rewards genome for each turn
        env.reset()

def run(config_file):
    print(env.action_space)
    print(env.observation_space)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,config_file)
    scores = []
    for x in range(10):
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        #p.add_reporter(neat.Checkpointer(5))

        # Run for up to x generations.
        winner = p.run(eval_genomes, 50)

        # show final stats
        print('\nBest genome:\n{!s}'.format(winner))

        #test winner
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        score = test_model(winner_net)  #Tests model 100 times and prints result
        scores.append(score)
    print("average over 10 trainings: ", (sum(scores))/10)

    sd = 0
    mean = sum(scores) / len(scores)
    for p in scores:
        sd += (mean - p) * (mean - p)
    sd /= len(scores)
    sd = math.sqrt(sd)
    print("standard dev: ", sd)

def test_model(winner):
    
    observation = [0, 0, 0, 0]
    score = 0
    reward = 0
    for i in range(100):
        done = False
        observation = [0, 0, 0, 0]
        while not done:
            #env.render()   #Render game      
            output = winner.activate(observation)
            action = max(output)
            if output[0] == action:
                action = 1
            else:
                action = 0

            observation, reward, done, info = env.step(action)
            if observation[3] > 0.5 or observation[3] < -0.5:
                done = True
                env.reset()
            score += reward
        env.reset()

    print("Score Over 100 tries:")
    print(score/100)
    return score/100



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    #config_path = os.path.join(local_dir, 'config-gym1')
    config_path = os.path.join(local_dir, 'config-no-neat1')
    run(config_path)