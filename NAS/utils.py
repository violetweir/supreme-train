"""
Utility functions for Neural Architecture Search
"""
import json
import pickle
import torch
import os
from typing import Dict, Any
from datetime import datetime


def save_architecture_config(arch_config: Dict[str, Any], filepath: str):
    """
    Save architecture configuration to a JSON file
    
    Args:
        arch_config: Architecture configuration dictionary
        filepath: Path to save the configuration
    """
    try:
        # Convert any numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            else:
                return obj
        
        arch_config = convert_numpy_types(arch_config)
        
        with open(filepath, 'w') as f:
            json.dump(arch_config, f, indent=2)
        print(f"Architecture configuration saved to {filepath}")
    except Exception as e:
        print(f"Error saving architecture configuration: {e}")


def load_architecture_config(filepath: str) -> Dict[str, Any]:
    """
    Load architecture configuration from a JSON file
    
    Args:
        filepath: Path to the configuration file
    
    Returns:
        Architecture configuration dictionary
    """
    try:
        with open(filepath, 'r') as f:
            arch_config = json.load(f)
        print(f"Architecture configuration loaded from {filepath}")
        return arch_config
    except Exception as e:
        print(f"Error loading architecture configuration: {e}")
        return None


def save_search_history(history: list, filepath: str):
    """
    Save search history to a pickle file
    
    Args:
        history: List of (architecture, score) tuples
        filepath: Path to save the history
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)
        print(f"Search history saved to {filepath}")
    except Exception as e:
        print(f"Error saving search history: {e}")


def load_search_history(filepath: str) -> list:
    """
    Load search history from a pickle file
    
    Args:
        filepath: Path to the history file
    
    Returns:
        List of (architecture, score) tuples
    """
    try:
        with open(filepath, 'rb') as f:
            history = pickle.load(f)
        print(f"Search history loaded from {filepath}")
        return history
    except Exception as e:
        print(f"Error loading search history: {e}")
        return []


def save_model_checkpoint(model, optimizer, epoch, best_score, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        best_score: Best evaluation score
        filepath: Path to save the checkpoint
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'best_score': best_score
        }
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")


def load_model_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to the checkpoint file
    
    Returns:
        checkpoint dictionary
    """
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None


def create_experiment_directory(base_dir="experiments"):
    """
    Create a directory for the current experiment with a timestamp
    
    Args:
        base_dir: Base directory for experiments
    
    Returns:
        Path to the experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(base_dir, f"nas_experiment_{timestamp}")
    
    try:
        os.makedirs(exp_dir, exist_ok=True)
        print(f"Experiment directory created: {exp_dir}")
        return exp_dir
    except Exception as e:
        print(f"Error creating experiment directory: {e}")
        return None


def log_search_progress(algorithm_name, iteration, best_score, current_score, 
                       arch_config, log_file):
    """
    Log search progress to a file
    
    Args:
        algorithm_name: Name of the search algorithm
        iteration: Current iteration number
        best_score: Best score so far
        current_score: Current evaluation score
        arch_config: Current architecture configuration
        log_file: Path to the log file
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "algorithm": algorithm_name,
            "iteration": iteration,
            "best_score": best_score,
            "current_score": current_score,
            "arch_config": str(arch_config)  # Convert to string for logging
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Error logging search progress: {e}")


def print_architecture_summary(arch_config):
    """
    Print a summary of the architecture configuration
    
    Args:
        arch_config: Architecture configuration dictionary
    """
    print("=" * 50)
    print("Architecture Configuration Summary")
    print("=" * 50)
    
    for key, value in arch_config.items():
        print(f"{key:20}: {value}")
    
    print("=" * 50)


def compare_architectures(arch1, arch2):
    """
    Compare two architecture configurations
    
    Args:
        arch1: First architecture configuration
        arch2: Second architecture configuration
    """
    print("=" * 60)
    print("Architecture Comparison")
    print("=" * 60)
    
    keys = set(arch1.keys()) | set(arch2.keys())
    
    for key in sorted(keys):
        val1 = arch1.get(key, "N/A")
        val2 = arch2.get(key, "N/A")
        
        status = "==" if val1 == val2 else "!="
        print(f"{key:20}: {str(val1):20} {status} {str(val2):20}")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Test architecture saving/loading
    test_arch = {
        'dims': [32, 64, 128, 256],
        'depths': [1, 2, 4, 5],
        'mlp_ratio': 2.0,
        'wt_type': 'db1',
        'learnable_wavelet': True
    }
    
    # Save and load architecture
    save_architecture_config(test_arch, "test_arch.json")
    loaded_arch = load_architecture_config("test_arch.json")
    print("Original:", test_arch)
    print("Loaded:", loaded_arch)
    print("Match:", test_arch == loaded_arch)
    
    # Test experiment directory creation
    exp_dir = create_experiment_directory()
    if exp_dir:
        print(f"Created experiment directory: {exp_dir}")
