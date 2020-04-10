import gym
import os
import neat
import visualize

env = gym.make('MountainCar-v0')
env.reset()
generations = 20
showGraph = "n"

def highestVal(vals):
    index = 0
    for x in range(0,len(vals)):
        if vals[x] > vals[index]:
            index = x
    return index



def eval_genomes(genomes, config):

    max_position = -.4
    for _, genome in genomes:
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
            if observation[0] > max_position:
                genome.fitness += 1
                max_position = observation[0]
            else:
                genome.fitness -= 1
            if done: 
                if observation[0] >= 0.5:
                    genome.fitness +=20
                break
        env.reset()

def run(config_file):
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
    winner = p.run(eval_genomes, generations)

    if(showGraph == "n"):

        #Visualization of the winner NN
        node_names = {-1:'Pos', -2: 'Vel'}
        visualize.draw_net(config, winner, True, node_names= node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    #test winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    score = test_model(winner_net)  #Tests model 100 times and prints result
    return score

def test_model(winner):
    
    score = 0
    for i in range(100):
        done = False
        observation = env.reset()
        t = 0
        while not done:
            t = t+1
            #env.render()
            output = winner.activate(observation)
            action = highestVal(output)
            observation, reward, done, info = env.step(action)
            if done:
                #print("Finished after {} timesteps".format(t+1))
                score += t
                break
        
    print("Score Over 100 tries:")
    print(score/100)

    return (score/100)

def start(gens, neat, graph):
    global generations
    global showGraph

    showGraph = graph
    generations = gens
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    if neat == "n":
        config_path = os.path.join(local_dir, 'config-gymCartNo')
    else:
        config_path = os.path.join(local_dir, 'config-gymCart')

    score = run(config_path)
    return score

if __name__ == '__main__':
    start(30, 'y', "n")