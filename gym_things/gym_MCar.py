import gym
import os
import neat


env = gym.make('MountainCar-v0')
env.reset()
steps = 200
score_requirement = -198
intial_games = 10000

def highestVal(vals):
    index = 0
    for x in range(1,len(vals)):
        if vals[x] > vals[index]:
            index = x
    return index



def eval_genomes(genomes, config):

    max_position = -.4
    for _, genome in genomes:
        runningReward = 0
        observation = env.reset()  #Inital observation
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, config) #Creat net for genome with configs
        genome.fitness = 0  #Starting fitness of 0
        while not done:
            action = net.activate(observation)
            #print(action)
            action = highestVal(action)
            observation, reward, done, info = env.step(action)
            # Give a reward for reaching a new maximum position
            if observation[0] > -0.2:
                genome.fitness += 1
            else:
                genome.fitness -= 1
            if done: 
                if observation[0] >= 0.5:
                    genome.fitness +=20
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
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #test winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    test_model(winner_net)  #Tests model 100 times and prints result

def test_model(winner):
    
    observation = [0, 0]
    score = 0
    reward = 0
    for i in range(100):
        done = False
        observation = [0, 0]
        t = 0
        while not done:
            t=t+1
            #env.render()
            output = winner.activate(observation)
            action = highestVal(output)
            observation, reward, done, info = env.step(action)
            if done:
                #print("Finished after {} timesteps".format(t+1))
                score += t
                break
        env.reset()
    print("Score Over 100 tries:")
    print(score/100)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-gymCartNo')
    run(config_path)