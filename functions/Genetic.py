import math
from Problem import AthletePerformanceProblem
import Problem
import random
import numpy as np
import time
class GeneticAlgorithm:

    def __init__(self, problem, population_size=100, num_generations=30, mutation_rate=0.01):
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
    
        print(self.population_size)
        print(self.num_generations)
        print(self.mutation_rate)
        
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.problem.random_individual()
            population.append(individual)
        return population
    

    def evaluate_individual(self, individual):
        # Evaluate the individual using the problem's evaluate method
        fitness_value = self.problem.evaluate_individual(individual)
        if fitness_value is not None: 
            return fitness_value
        else:
            return 0.0

    def evaluate_population(self, population):
        """Evaluates the entire population and returns list of (individual, fitness) tuples"""
        fitness_indiv = []
        for individual in population:
            ft = self.evaluate_individual(individual)
            fitness_value = 0.5*ft[3] - 0.3 * ft[2] - 0.2 * ft[1]
            fitness_indiv.append([individual, fitness_value, ft])
        return fitness_indiv

    # SELECTION ALGORITHMS
    
    def roulette_wheel_selection(self, evaluated_population, num_parents):
        """
        Roulette wheel selection (fitness proportionate selection)
        Higher fitness individuals have higher probability of selection

        """
        # Extract fitness values
        population, fitness_values = zip(*evaluated_population)
        population = list(population)
        
        # Handle negative fitness values if they exist by shifting all values
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 1e-5 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values
            
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            # If all fitness values are 0, select randomly
            return random.choices(population, k=num_parents)
            
        selection_probs = [f/total_fitness for f in adjusted_fitness]
        
        # Select parents based on probabilities
        mating_pool = random.choices(population, weights=selection_probs, k=num_parents)
        return mating_pool
    
    def tournament_selection(self, evaluated_population, num_parents, tournament_size=3):
        """
        Tournament selection
        Randomly select tournament_size individuals and pick the best one
        Repeat until we have num_parents individuals
        """
        mating_pool = []
        
        for _ in range(num_parents):
            # Randomly select tournament_size individuals
            tournament = random.sample(evaluated_population, min(tournament_size, len(evaluated_population)))
            
            # Find the winner (individual with highest fitness)
            winner = max(tournament, key=lambda x: x[1])[0]
            mating_pool.append(winner)
            
        return mating_pool
    
    def rank_selection(self, evaluated_population, num_parents):
        """
        Rank selection
        Selection probability is based on rank rather than absolute fitness
        Helps maintain diversity when fitness differences are extreme
        """
        # Sort population by fitness (ascending)
        sorted_population = sorted(evaluated_population, key=lambda x: x[1])
        
        # Assign ranks (higher rank = higher fitness)
        population_size = len(sorted_population)
        ranks = list(range(1, population_size + 1))
        
        # Calculate selection probabilities based on rank
        total_rank = sum(ranks)
        selection_probs = [r/total_rank for r in ranks]
        
        # Extract individuals from sorted population
        individuals = [individual for individual, _, _ in sorted_population]
        
        # Select parents based on rank probabilities
        mating_pool = random.choices(individuals, weights=selection_probs, k=num_parents)
        return mating_pool
    
    # PAIRING LOGIC
    
    def random_pairing(self, mating_pool):
        """
            random pairing method for the selected parents
            return list of pairs
        """
        # Make a copy of the mating pool to shuffle
        shuffled_pool = mating_pool.copy()
        random.shuffle(shuffled_pool)
        
        # Create pairs
        pairs = []
        for i in range(0, len(shuffled_pool) - 1, 2):
            pairs.append((shuffled_pool[i], shuffled_pool[i+1]))
            
        # If odd number of individuals, pair the last one with a random individual
        if len(shuffled_pool) % 2 != 0:
            pairs.append((shuffled_pool[-1], random.choice(shuffled_pool[:-1])))
            
        return pairs
    
    def fitness_based_pairing(self, evaluated_population, mating_pool):
        """
        Pairs individuals based on fitness similarity
        
        return: List of parent pairs (tuples)
        """
        # Create a fitness dictionary for quick lookup
        fitness_dict = {individual: fitness for individual, fitness in evaluated_population}
        
        # Sort mating pool by fitness
        sorted_pool = sorted(mating_pool, key=lambda ind: fitness_dict[ind])
        
        # Create pairs of adjacent individuals
        pairs = []
        for i in range(0, len(sorted_pool) - 1, 2):
            pairs.append((sorted_pool[i], sorted_pool[i+1]))
            
        # If odd number of individuals, pair the last one with a random individual
        if len(sorted_pool) % 2 != 0:
            pairs.append((sorted_pool[-1], random.choice(sorted_pool[:-1])))
            
        return pairs
    
    def diverse_pairing(self, evaluated_population, mating_pool):
        """
        Pairs individuals to maximize diversity
        Individuals with different fitness are paired together
        """
        
        fitness_dict = {tuple(individual): fitness for individual, fitness, ind in evaluated_population}
        
        # Sort mating pool by fitness
        sorted_pool = sorted(mating_pool, key=lambda ind: fitness_dict[tuple(ind)])
        
        
        pairs = []
        left, right = 0, len(sorted_pool) - 1
        
        while left < right:
            pairs.append((sorted_pool[left], sorted_pool[right]))
            left += 1
            right -= 1
            
        # If odd number of individuals, pair the middle one with a random individual
        if len(sorted_pool) % 2 != 0:
            middle = len(sorted_pool) // 2
            pairs.append((sorted_pool[middle], random.choice(sorted_pool[:middle] + sorted_pool[middle+1:])))
            
        return pairs
    
    # CROSSOVER
    
    def two_point_crossover(self, parent1, parent2, crossover_rate=0.8):
        """
        Two-point crossover operation
        Selects two random points and exchanges the segments between parents
        
        Args:
            parent1, parent2: The parent individuals
            crossover_rate: Probability of performing crossover
            
        Returns:
            Two offspring individuals
        """
        # Check if crossover should be performed
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Get parent length and ensure it's the same for both parents
        length = len(parent1)
        if length != len(parent2):
            raise ValueError("Parents must be of the same length for crossover")
        
        # Select two random crossover points
        if length < 3:
            # handle short chromosome
            point1 = random.randint(1, length - 1)
            point2 = point1
        else:
            points = sorted(random.sample(range(1, length), 2))
            point1, point2 = points
        
      
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return offspring1, offspring2

    
    def mutate(self, offspring):
        """
        apply muatation on a given individual under the mutation rate

        """
        mutated = offspring[:]
        if random.random() < self.mutation_rate:
            ind = random.randint(0, len(mutated) - 1)
            mutated[ind] = random.choice(self.problem.actions())
        return mutated
    
    def run(self):
        """Returns standardized search result format"""
        start_time = time.time()
        population = self.initialize_population()
        best_fitness = -float('inf')
        best_individual = None

        for generation in range(self.num_generations):
            evaluated_pop = self.evaluate_population(population)
            current_best = max(evaluated_pop, key=lambda x: x[1])
            
            if current_best[1] > best_fitness:
                best_fitness = current_best[1]
                best_individual = current_best[0]

            mating_pool = self.rank_selection(evaluated_pop, self.population_size)
            parent_pairs = self.random_pairing(mating_pool)
            
            new_population = []
            for p1, p2 in parent_pairs:
                o1, o2 = self.two_point_crossover(p1, p2)
                new_population.extend([self.mutate(o1), self.mutate(o2)])
            
            population = new_population[:self.population_size]

        # Build final schedule
        final_state = self.problem.initial_state
        schedule = []
        for action in best_individual:
            final_state = self.problem.apply_action(final_state, action)
            day, fatigue, risk, performance, _ = final_state
            schedule.append({
                'day': day,
                'intensity': action[0],
                'duration': action[1],
                'performance': performance,
                'fatigue': fatigue,
                'risk': risk
            })

        self.execution_time = time.time() - start_time
        return self._format_result(schedule, final_state, best_individual)

    def _format_result(self, schedule, final_state, best_individual):
        """Standardized result format for API"""
        if not best_individual:
            return {
                'success': False,
                'message': 'No solution found',
                'stats': self._get_stats()
            }

        # Calculate metrics
        rest_days = sum(1 for a in best_individual if a[0] == 0)
        high_intensity_days = sum(1 for a in best_individual if a[0] >= 0.7)
        total_workload = sum(a[0]*a[1] for a in best_individual)

        return {
            'success': True,
            'message': 'Solution found with Genetic Algorithm',
            'schedule': schedule,
            'finalState': {
                'day': final_state[0],
                'performance': final_state[3],
                'fatigue': final_state[1],
                'risk': final_state[2]
            },
            'stats': self._get_stats(),
            'metrics': {
                'total_days': final_state[0],
                'rest_days': rest_days,
                'high_intensity_days': high_intensity_days,
                'total_workload': total_workload
            }
        }

    def _get_stats(self):
        return {
            'nodesExplored': self.num_generations * self.population_size,
            'maxQueueSize': self.population_size,
            'executionTime': self.execution_time
        }