"""
Main NAS Controller - ties all components together
"""
import os
import json
import torch
import random
from typing import Dict, Any, Optional
from NAS.search_space import StarNetSearchSpace
from NAS.search_algorithms import RandomSearch, EvolutionarySearch, BayesianOptimizationSearch
from NAS.evaluator import ModelEvaluator
from NAS.utils import save_architecture_config, save_search_history, create_experiment_directory, log_search_progress


class NASController:
    """
    Main controller for Neural Architecture Search
    """
    
    def __init__(self, search_space=None, evaluator=None, experiment_dir=None):
        self.search_space = search_space or StarNetSearchSpace()
        self.evaluator = evaluator or ModelEvaluator()
        self.experiment_dir = experiment_dir or create_experiment_directory()
        
        # Create experiment directory if it doesn't exist
        if self.experiment_dir:
            os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(self.experiment_dir, "search_progress.log") if self.experiment_dir else None
    
    def run_random_search(self, max_evaluations=100, save_best=True):
        """
        Run random search for architecture optimization
        
        Args:
            max_evaluations: Maximum number of architectures to evaluate
            save_best: Whether to save the best architecture found
        
        Returns:
            Best architecture configuration and score
        """
        print("Running Random Search...")
        
        # Initialize search algorithm
        random_search = RandomSearch(self.search_space, max_evaluations)
        
        # Define evaluation function
        def evaluate_func(arch_config):
            score = self.evaluator.evaluate_architecture_simple(arch_config)
            return score
        
        # Run search
        best_arch, best_score = random_search.search(evaluate_func)
        
        # Save results
        if self.experiment_dir:
            save_search_history(random_search.history, 
                              os.path.join(self.experiment_dir, "random_search_history.pkl"))
            
            if save_best and best_arch:
                save_architecture_config(best_arch, 
                                       os.path.join(self.experiment_dir, "best_random_arch.json"))
        
        print(f"Random Search completed. Best score: {best_score:.4f}")
        return best_arch, best_score
    
    def run_evolutionary_search(self, population_size=20, generations=10, 
                              mutation_prob=0.2, save_best=True):
        """
        Run evolutionary search for architecture optimization
        
        Args:
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            mutation_prob: Probability of mutation
            save_best: Whether to save the best architecture found
        
        Returns:
            Best architecture configuration and score
        """
        print("Running Evolutionary Search...")
        
        # Initialize search algorithm
        evolutionary_search = EvolutionarySearch(
            self.search_space, population_size, generations, mutation_prob
        )
        
        # Define evaluation function
        def evaluate_func(arch_config):
            score = self.evaluator.evaluate_architecture_simple(arch_config)
            return score
        
        # Run search
        best_arch, best_score = evolutionary_search.search(evaluate_func)
        
        # Save results
        if self.experiment_dir:
            save_search_history(evolutionary_search.history, 
                              os.path.join(self.experiment_dir, "evolutionary_search_history.pkl"))
            
            if save_best and best_arch:
                save_architecture_config(best_arch, 
                                       os.path.join(self.experiment_dir, "best_evolutionary_arch.json"))
        
        print(f"Evolutionary Search completed. Best score: {best_score:.4f}")
        return best_arch, best_score
    
    def run_bayesian_optimization(self, max_evaluations=50, save_best=True):
        """
        Run Bayesian optimization search for architecture optimization
        
        Args:
            max_evaluations: Maximum number of architectures to evaluate
            save_best: Whether to save the best architecture found
        
        Returns:
            Best architecture configuration and score
        """
        print("Running Bayesian Optimization...")
        
        # Initialize search algorithm
        bayesian_search = BayesianOptimizationSearch(self.search_space, max_evaluations)
        
        # Define evaluation function
        def evaluate_func(arch_config):
            score = self.evaluator.evaluate_architecture_simple(arch_config)
            return score
        
        # Run search
        best_arch, best_score = bayesian_search.search(evaluate_func)
        
        # Save results
        if self.experiment_dir:
            save_search_history(bayesian_search.history, 
                              os.path.join(self.experiment_dir, "bayesian_search_history.pkl"))
            
            if save_best and best_arch:
                save_architecture_config(best_arch, 
                                       os.path.join(self.experiment_dir, "best_bayesian_arch.json"))
        
        print(f"Bayesian Optimization completed. Best score: {best_score:.4f}")
        return best_arch, best_score
    
    def run_search(self, algorithm="random", **kwargs):
        """
        Run NAS search with specified algorithm
        
        Args:
            algorithm: Search algorithm to use ("random", "evolutionary", "bayesian")
            **kwargs: Algorithm-specific parameters
        
        Returns:
            Best architecture configuration and score
        """
        if algorithm == "random":
            return self.run_random_search(**kwargs)
        elif algorithm == "evolutionary":
            return self.run_evolutionary_search(**kwargs)
        elif algorithm == "bayesian":
            return self.run_bayesian_optimization(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def benchmark_architecture(self, arch_config: Dict[str, Any], 
                             save_results=True) -> Dict[str, Any]:
        """
        Benchmark a specific architecture configuration
        
        Args:
            arch_config: Architecture configuration to benchmark
            save_results: Whether to save benchmark results
        
        Returns:
            Dictionary with benchmark results
        """
        print("Benchmarking architecture...")
        
        # Full evaluation
        results = self.evaluator.evaluate_architecture(arch_config)
        
        if results['success'] and self.experiment_dir and save_results:
            # Save benchmark results
            benchmark_file = os.path.join(self.experiment_dir, "benchmark_results.json")
            with open(benchmark_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to {benchmark_file}")
        
        return results
    
    def compare_search_algorithms(self, algorithms=None, iterations=1):
        """
        Compare different search algorithms
        
        Args:
            algorithms: List of algorithms to compare (default: all)
            iterations: Number of times to run each algorithm
        
        Returns:
            Dictionary with comparison results
        """
        if algorithms is None:
            algorithms = ["random", "evolutionary", "bayesian"]
        
        comparison_results = {}
        
        for algorithm in algorithms:
            print(f"\nComparing {algorithm} search...")
            scores = []
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}")
                _, score = self.run_search(algorithm)
                scores.append(score)
            
            comparison_results[algorithm] = {
                'mean_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'scores': scores
            }
        
        # Save comparison results
        if self.experiment_dir:
            comparison_file = os.path.join(self.experiment_dir, "algorithm_comparison.json")
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"Comparison results saved to {comparison_file}")
        
        return comparison_results


# Example usage
if __name__ == "__main__":
    # Create NAS controller
    controller = NASController()
    
    # Run random search
    print("=== Random Search ===")
    best_random_arch, best_random_score = controller.run_search("random", max_evaluations=5)
    
    # Run evolutionary search
    print("\n=== Evolutionary Search ===")
    best_evo_arch, best_evo_score = controller.run_search(
        "evolutionary", population_size=5, generations=3
    )
    
    # Run Bayesian optimization
    print("\n=== Bayesian Optimization ===")
    best_bayes_arch, best_bayes_score = controller.run_search("bayesian", max_evaluations=5)
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"Random Search Best Score: {best_random_score:.4f}")
    print(f"Evolutionary Search Best Score: {best_evo_score:.4f}")
    print(f"Bayesian Optimization Best Score: {best_bayes_score:.4f}")
    
    # Benchmark best architecture
    if best_evo_score >= max(best_random_score, best_bayes_score):
        best_arch = best_evo_arch
        print("\n=== Benchmarking Best Architecture ===")
        benchmark_results = controller.benchmark_architecture(best_arch)
        print("Benchmark results:", benchmark_results)
