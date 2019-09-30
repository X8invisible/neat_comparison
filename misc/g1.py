import numpy
import ga

equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

num_weights = 6

sol_per_pop = 8


pop_size = (sol_per_pop, num_weights)


new_population = numpy.random.uniform(low = -4.0, high = 4.0, size = pop_size)



num_generations = 100
num_parents_mating = 4

for generation in range(num_generations):
    #measuring fitness
    fitness = ga.cal_pop_fitness(equation_inputs,new_population)
    #selecting best parents in the population
    parents = ga.select_mating_pool(new_population,fitness,num_parents_mating)

    #generate next population using crossover
    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

    #adding variantions using mutation
    offspring_mutation = ga.mutation(offspring_crossover)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
     # The best result in the current iteration.
    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))


# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])





