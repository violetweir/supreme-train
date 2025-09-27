"""
Main script to run Neural Architecture Search
"""
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NAS.controller import NASController
from NAS.search_space import StarNetSearchSpace
from NAS.evaluator import ModelEvaluator
from NAS.utils import load_architecture_config, print_architecture_summary


def main():
    parser = argparse.ArgumentParser(description="Neural Architecture Search for StarNet_NEW_CONV")
    parser.add_argument("--algorithm", choices=["random", "evolutionary", "bayesian"], 
                       default="random", help="Search algorithm to use")
    parser.add_argument("--max_evaluations", type=int, default=50, 
                       help="Maximum number of evaluations")
    parser.add_argument("--population_size", type=int, default=20, 
                       help="Population size for evolutionary search")
    parser.add_argument("--generations", type=int, default=10, 
                       help="Number of generations for evolutionary search")
    parser.add_argument("--mutation_prob", type=float, default=0.2, 
                       help="Mutation probability for evolutionary search")
    parser.add_argument("--experiment_dir", type=str, 
                       help="Directory to save experiment results")
    parser.add_argument("--benchmark", type=str, 
                       help="Path to architecture config file to benchmark")
    parser.add_argument("--compare_algorithms", action="store_true",
                       help="Compare different search algorithms")
    
    args = parser.parse_args()
    
    # If benchmarking a specific architecture
    if args.benchmark:
        if not os.path.exists(args.benchmark):
            print(f"Error: Architecture config file {args.benchmark} not found")
            return
        
        # Load architecture configuration
        arch_config = load_architecture_config(args.benchmark)
        if not arch_config:
            print("Error: Failed to load architecture configuration")
            return
        
        print("Benchmarking architecture from file:", args.benchmark)
        print_architecture_summary(arch_config)
        
        # Create evaluator and benchmark
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_architecture(arch_config)
        
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))
        return
    
    # Create NAS components
    search_space = StarNetSearchSpace()
    evaluator = ModelEvaluator()
    controller = NASController(
        search_space=search_space,
        evaluator=evaluator,
        experiment_dir=args.experiment_dir
    )
    
    # Compare algorithms if requested
    if args.compare_algorithms:
        print("Comparing search algorithms...")
        comparison_results = controller.compare_search_algorithms(
            iterations=3  # Run each algorithm 3 times for comparison
        )
        
        print("\nAlgorithm Comparison Results:")
        print("=" * 50)
        for algorithm, results in comparison_results.items():
            print(f"{algorithm:15}: Mean={results['mean_score']:.4f}, "
                  f"Max={results['max_score']:.4f}, Min={results['min_score']:.4f}")
        return
    
    # Run search with specified algorithm
    print(f"Running {args.algorithm} search...")
    
    if args.algorithm == "random":
        best_arch, best_score = controller.run_search(
            "random", 
            max_evaluations=args.max_evaluations
        )
    elif args.algorithm == "evolutionary":
        best_arch, best_score = controller.run_search(
            "evolutionary",
            population_size=args.population_size,
            generations=args.generations,
            mutation_prob=args.mutation_prob
        )
    elif args.algorithm == "bayesian":
        best_arch, best_score = controller.run_search(
            "bayesian",
            max_evaluations=args.max_evaluations
        )
    
    # Print results
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    print(f"Best Score: {best_score:.4f}")
    print("\nBest Architecture:")
    print_architecture_summary(best_arch)
    
    # Benchmark the best architecture
    print("\nBenchmarking best architecture...")
    benchmark_results = controller.benchmark_architecture(best_arch)
    
    if benchmark_results['success']:
        print("\nDetailed Benchmark Results:")
        for key, value in benchmark_results.items():
            if key not in ['arch_config', 'success']:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
