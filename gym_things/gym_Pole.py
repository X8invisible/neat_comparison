import gym
import os
import neat
import visualize
env = gym.make('CartPole-v1')
env.reset()
steps = 1000
generations = 20


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
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, generations)

    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    #visualize.draw_net(config, winner, True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)
    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #test winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    score = test_model(winner_net)  #Tests model 100 times and prints result
    return score

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
            score += reward
            if done:
                break
        env.reset()

    print("Score Over 100 tries:")
    print(score/100)
    return (score/100)

def start(gens, neat):
    global generations 
    generations = gens
     # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    if neat == "n":
        config_path = os.path.join(local_dir, 'config-gymPoleNo')
    else:
        config_path = os.path.join(local_dir, 'config-gymPole')
    score = run(config_path)
    return score

if __name__ == '__main__':
   start()