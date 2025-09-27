"""
Model evaluator for Neural Architecture Search
"""
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
import time
import numpy as np
from typing import Dict, Any
from NAS.search_space import StarNetSearchSpace


class ModelEvaluator:
    """
    Evaluator for StarNet_NEW_CONV architectures
    """
    
    def __init__(self, input_shape=(1, 3, 224, 224), num_classes=1000):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.search_space = StarNetSearchSpace()
    
    def compute_flops(self, model):
        """
        Compute FLOPs and parameters of the model
        """
        try:
            input_tensor = torch.randn(self.input_shape)
            macs, params = profile(model, inputs=(input_tensor,))
            macs, params = clever_format([macs, params], "%.3f")
            return macs, params
        except Exception as e:
            print(f"Error computing FLOPs: {e}")
            return "N/A", "N/A"
    
    def compute_inference_time(self, model, device='cpu', iterations=10):
        """
        Compute average inference time of the model
        """
        try:
            model.eval()
            model.to(device)
            
            input_tensor = torch.randn(self.input_shape).to(device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            return avg_time * 1000  # Convert to milliseconds
        except Exception as e:
            print(f"Error computing inference time: {e}")
            return float('inf')
    
    def evaluate_architecture(self, arch_config: Dict[str, Any], 
                            compute_flops=True, compute_inference_time=True,
                            device='cpu'):
        """
        Evaluate an architecture configuration
        
        Args:
            arch_config: Architecture configuration dictionary
            compute_flops: Whether to compute FLOPs
            compute_inference_time: Whether to compute inference time
            device: Device to run evaluation on ('cpu' or 'cuda')
        
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Create model
            model = self.search_space.create_model(arch_config)
            
            results = {
                'arch_config': arch_config,
                'success': True
            }
            
            # Compute FLOPs and parameters
            if compute_flops:
                macs, params = self.compute_flops(model)
                results['flops'] = macs
                results['params'] = params
            
            # Compute inference time
            if compute_inference_time:
                inf_time = self.compute_inference_time(model, device)
                results['inference_time'] = inf_time
            
            return results
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return {
                'arch_config': arch_config,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_architecture_simple(self, arch_config: Dict[str, Any]):
        """
        Simple evaluation function that returns a score based on model complexity
        This is a placeholder for actual training/evaluation
        """
        try:
            # Create model
            model = self.search_space.create_model(arch_config)
            
            # Compute parameters as a simple metric
            total_params = sum(p.numel() for p in model.parameters())
            
            # Normalize by a reference value (e.g., parameters of a standard model)
            # Smaller models get higher scores (assuming we want to minimize parameters)
            reference_params = 5000000  # 5M parameters as reference
            score = max(0, 1 - (total_params / reference_params))
            
            return score
        except Exception as e:
            print(f"Error in simple evaluation: {e}")
            return 0.0
    
    def evaluate_architecture_with_training_proxy(self, arch_config: Dict[str, Any]):
        """
        Evaluation function that uses a proxy for training performance
        Based on model characteristics like parameter count, FLOPs, etc.
        """
        try:
            # Create model
            model = self.search_space.create_model(arch_config)
            
            # Compute various metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Compute FLOPs
            input_tensor = torch.randn(self.input_shape)
            macs, _ = profile(model, inputs=(input_tensor,))
            
            # Create a composite score
            # Lower parameters and FLOPs are better (efficiency)
            # But we also want a reasonable model, so we balance these factors
            
            # Normalize parameters (0-1 scale, lower is better)
            param_score = max(0, 1 - (total_params / 10000000))  # 10M as reference
            
            # Normalize FLOPs (0-1 scale, lower is better)
            flops_score = max(0, 1 - (macs / 2000000000))  # 2G FLOPs as reference
            
            # Combine scores (can be adjusted based on priorities)
            composite_score = 0.5 * param_score + 0.5 * flops_score
            
            return composite_score
        except Exception as e:
            print(f"Error in training proxy evaluation: {e}")
            return 0.0


# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Test with a sample architecture
    search_space = StarNetSearchSpace()
    arch_config = search_space.sample_architecture()
    
    print("Evaluating architecture:", arch_config)
    
    # Simple evaluation
    score = evaluator.evaluate_architecture_simple(arch_config)
    print(f"Simple evaluation score: {score:.4f}")
    
    # Training proxy evaluation
    proxy_score = evaluator.evaluate_architecture_with_training_proxy(arch_config)
    print(f"Training proxy score: {proxy_score:.4f}")
    
    # Full evaluation
    results = evaluator.evaluate_architecture(arch_config)
    print("Full evaluation results:", results)
