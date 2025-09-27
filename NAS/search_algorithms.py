"""
Search algorithms for Neural Architecture Search
"""
import random
import numpy as np
from typing import List, Tuple, Callable
from NAS.search_space import StarNetSearchSpace


class RandomSearch:
    """
    Random search algorithm for NAS
    """
    
    def __init__(self, search_space: StarNetSearchSpace, max_evaluations: int = 100):
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        self.history = []
    
    def search(self, evaluate_func: Callable):
        """
        Perform random search
        
        Args:
            evaluate_func: Function to evaluate an architecture
                          Should take an architecture config and return a score
        """
        best_arch = None
        best_score = -float('inf')
        
        for i in range(self.max_evaluations):
            # Sample random architecture
            arch_config = self.search_space.sample_architecture()
            
            # Evaluate architecture
            score = evaluate_func(arch_config)
            
            # Update history
            self.history.append((arch_config, score))
            
            # Update best architecture
            if score > best_score:
                best_score = score
                best_arch = arch_config
            
            print(f"Random Search [{i+1}/{self.max_evaluations}]: Score = {score:.4f}, Best = {best_score:.4f}")
        
        return best_arch, best_score


class EvolutionarySearch:
    """
    Evolutionary search algorithm for NAS
    """
    
    def __init__(self, search_space: StarNetSearchSpace, population_size: int = 20, 
                 generations: int = 10, mutation_prob: float = 0.2):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.history = []
    
    def _initialize_population(self) -> List:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            arch_config = self.search_space.sample_architecture()
            population.append(arch_config)
        return population
    
    def _mutate(self, arch_config):
        """Mutate an architecture configuration"""
        mutated = arch_config.copy()
        
        # Randomly select which parameter to mutate
        param_to_mutate = random.choice([
            'dims', 'depths', 'mlp_ratio', 'wt_type', 'learnable_wavelet'
        ])
        
        if param_to_mutate == 'dims':
            mutated['dims'] = random.choice(self.search_space.dim_choices)
        elif param_to_mutate == 'depths':
            mutated['depths'] = random.choice(self.search_space.depth_choices)
        elif param_to_mutate == 'mlp_ratio':
            mutated['mlp_ratio'] = random.choice(self.search_space.mlp_ratio_choices)
        elif param_to_mutate == 'wt_type':
            mutated['wt_type'] = random.choice(self.search_space.wt_type_choices)
        elif param_to_mutate == 'learnable_wavelet':
            mutated['learnable_wavelet'] = random.choice(self.search_space.learnable_wavelet_choices)
        
        return mutated
    
    def _crossover(self, parent1, parent2):
        """Crossover two architectures"""
        child = {}
        
        # For each parameter, randomly select from either parent
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def search(self, evaluate_func: Callable):
        """
        Perform evolutionary search
        
        Args:
            evaluate_func: Function to evaluate an architecture
                          Should take an architecture config and return a score
        """
        # Initialize population
        population = self._initialize_population()
        fitness_scores = []
        
        best_arch = None
        best_score = -float('inf')
        
        # Evaluate initial population
        for arch in population:
            score = evaluate_func(arch)
            fitness_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_arch = arch
        
        self.history.append((best_arch, best_score))
        print(f"Generation 0: Best Score = {best_score:.4f}")
        
        # Evolutionary process
        for gen in range(self.generations):
            # Select parents (tournament selection)
            new_population = []
            
            # Keep best individual (elitism)
            best_idx = np.argmax(fitness_scores)
            new_population.append(population[best_idx])
            
            # Generate new population
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent1 = population[parent1_idx]
                
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent2 = population[parent2_idx]
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_prob:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            # Evaluate new population
            population = new_population
            fitness_scores = []
            
            for arch in population:
                score = evaluate_func(arch)
                fitness_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_arch = arch
            
            self.history.append((best_arch, best_score))
            print(f"Generation {gen+1}: Best Score = {best_score:.4f}")
        
        return best_arch, best_score


class BayesianOptimizationSearch:
    """
    Bayesian optimization search (simplified version)
    """
    
    def __init__(self, search_space: StarNetSearchSpace, max_evaluations: int = 50):
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        self.history = []
        self.evaluated_points = []  # Store evaluated architecture vectors
        self.evaluated_scores = []  # Store corresponding scores
    
    def _arch_to_vector(self, arch_config):
        """Convert architecture to vector representation"""
        return self.search_space.get_architecture_vector(arch_config)
    
    def _vector_to_arch(self, vector):
        """Convert vector to architecture configuration"""
        return self.search_space.get_architecture_from_vector(vector)
    
    def _acquisition_function(self, arch_vector):
        """
        Simple acquisition function (random for this simplified version)
        In a full implementation, this would use a Gaussian Process model
        """
        # For this simplified version, we just return a random value
        # A real implementation would use the GP model to predict
        # the expected improvement or upper confidence bound
        return random.random()
    
    def search(self, evaluate_func: Callable):
        """
        Perform Bayesian optimization search (simplified)
        
        Args:
            evaluate_func: Function to evaluate an architecture
                          Should take an architecture config and return a score
        """
        best_arch = None
        best_score = -float('inf')
        
        # Initial random samples
        initial_samples = min(10, self.max_evaluations)
        
        for i in range(initial_samples):
            arch_config = self.search_space.sample_architecture()
            score = evaluate_func(arch_config)
            
            # Store for history
            self.history.append((arch_config, score))
            self.evaluated_points.append(self._arch_to_vector(arch_config))
            self.evaluated_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_arch = arch_config
            
            print(f"Bayesian Opt. Initial [{i+1}/{initial_samples}]: Score = {score:.4f}, Best = {best_score:.4f}")
        
        # Bayesian optimization iterations
        for i in range(initial_samples, self.max_evaluations):
            # In a real implementation, we would:
            # 1. Fit a Gaussian Process model to the evaluated points
            # 2. Use the acquisition function to select the next point
            # 3. Evaluate that point
            
            # For this simplified version, we'll just sample randomly
            arch_config = self.search_space.sample_architecture()
            score = evaluate_func(arch_config)
            
            # Store for history
            self.history.append((arch_config, score))
            self.evaluated_points.append(self._arch_to_vector(arch_config))
            self.evaluated_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_arch = arch_config
            
            print(f"Bayesian Opt. [{i+1}/{self.max_evaluations}]: Score = {score:.4f}, Best = {best_score:.4f}")
        
        return best_arch, best_score


# Example usage
if __name__ == "__main__":
    search_space = StarNetSearchSpace()
    
    # Define a dummy evaluation function
    def dummy_evaluate(arch_config):
        # In practice, this would train and evaluate the model
        # For demonstration, we'll just return a random score
        return random.random()
    
    # Test random search
    print("Testing Random Search...")
    random_search = RandomSearch(search_space, max_evaluations=5)
    best_arch, best_score = random_search.search(dummy_evaluate)
    print(f"Best architecture: {best_arch}")
    print(f"Best score: {best_score:.4f}")
    
    print("\nTesting Evolutionary Search...")
    # Test evolutionary search
    evolutionary_search = EvolutionarySearch(search_space, population_size=5, generations=3)
    best_arch, best_score = evolutionary_search.search(dummy_evaluate)
    print(f"Best architecture: {best_arch}")
    print(f"Best score: {best_score:.4f}")
