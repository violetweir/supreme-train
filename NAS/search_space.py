"""
Search space definition for StarNet_NEW_CONV model NAS
"""
import torch
import torch.nn as nn
import random


class StarNetSearchSpace:
    """
    Define search space for StarNet_NEW_CONV model
    """
    
    def __init__(self):
        # Define search space for model dimensions
        self.dim_choices = [
            [24, 48, 96, 192],    # Small model
            [32, 64, 128, 256],   # Medium model
            [40, 80, 160, 320],   # Large model (default)
            [48, 96, 192, 384],   # Extra large model
        ]
        
        # Define search space for model depths
        self.depth_choices = [
            [1, 1, 2, 2],    # Very shallow
            [1, 2, 4, 5],    # Default (medium)
            [2, 3, 6, 7],    # Deep
            [2, 4, 8, 10],   # Very deep
        ]
        
        # Define search space for MLP ratios
        self.mlp_ratio_choices = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Define search space for wavelet types
        self.wt_type_choices = ['db1', 'db2', 'db3', 'haar']
        
        # Define search space for learnable wavelet
        self.learnable_wavelet_choices = [True, False]
    
    def sample_architecture(self):
        """
        Sample a random architecture from the search space
        """
        dims = random.choice(self.dim_choices)
        depths = random.choice(self.depth_choices)
        mlp_ratio = random.choice(self.mlp_ratio_choices)
        wt_type = random.choice(self.wt_type_choices)
        learnable_wavelet = random.choice(self.learnable_wavelet_choices)
        
        return {
            'dims': dims,
            'depths': depths,
            'mlp_ratio': mlp_ratio,
            'wt_type': wt_type,
            'learnable_wavelet': learnable_wavelet
        }
    
    def create_model(self, arch_config):
        """
        Create a model based on the given architecture configuration
        Note: This is a simplified version that doesn't actually create the model
        to avoid dependency issues. In a real implementation, this would create
        the actual StarNet_NEW_CONV model.
        """
        # In a real implementation, this would create the actual model
        # For now, we'll just return a placeholder
        class PlaceholderModel(nn.Module):
            def __init__(self, arch_config):
                super().__init__()
                self.arch_config = arch_config
                
            def forward(self, x):
                # Simple placeholder forward pass
                return torch.mean(x, dim=[2, 3])  # Global average pooling
        
        return PlaceholderModel(arch_config)
    
    def get_architecture_vector(self, arch_config):
        """
        Convert architecture configuration to a vector representation
        """
        # Convert dims to indices
        dim_idx = self.dim_choices.index(arch_config['dims'])
        
        # Convert depths to indices
        depth_idx = self.depth_choices.index(arch_config['depths'])
        
        # Convert mlp_ratio to index
        mlp_idx = self.mlp_ratio_choices.index(arch_config['mlp_ratio'])
        
        # Convert wt_type to index
        wt_idx = self.wt_type_choices.index(arch_config['wt_type'])
        
        # Convert learnable_wavelet to index
        lw_idx = int(arch_config['learnable_wavelet'])
        
        return [dim_idx, depth_idx, mlp_idx, wt_idx, lw_idx]
    
    def get_architecture_from_vector(self, vector):
        """
        Convert vector representation back to architecture configuration
        """
        dim_idx, depth_idx, mlp_idx, wt_idx, lw_idx = vector
        
        return {
            'dims': self.dim_choices[dim_idx],
            'depths': self.depth_choices[depth_idx],
            'mlp_ratio': self.mlp_ratio_choices[mlp_idx],
            'wt_type': self.wt_type_choices[wt_idx],
            'learnable_wavelet': bool(lw_idx)
        }


# Example usage
if __name__ == "__main__":
    search_space = StarNetSearchSpace()
    
    # Sample a random architecture
    arch = search_space.sample_architecture()
    print("Sampled architecture:", arch)
    
    # Create model from architecture
    model = search_space.create_model(arch)
    print("Model created successfully")
    
    # Convert to vector and back
    vector = search_space.get_architecture_vector(arch)
    print("Vector representation:", vector)
    
    arch_back = search_space.get_architecture_from_vector(vector)
    print("Converted back:", arch_back)
    print("Match:", arch == arch_back)
