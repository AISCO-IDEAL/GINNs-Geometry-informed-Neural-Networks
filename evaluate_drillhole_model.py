#!/usr/bin/env python3
"""Evaluate a trained GINN drillhole model on a grid and save SDF values.

Usage:
    python evaluate_drillhole_model.py <model_dir> <domain_width> <domain_height> <grid_resolution> <output_path>

Where:
    <model_dir> = Directory containing the trained model checkpoint and config
    <domain_width> = Domain width in mm
    <domain_height> = Domain height in mm  
    <grid_resolution> = Resolution for the evaluation grid (default: 200)
    <output_path> = Path where to save the SDF grid as .npy file
"""

import os
import sys
import glob
import numpy as np
import torch
import yaml

# Add GINN to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.wire import ConditionalWIRE
from util.misc import get_problem


def main():
    if len(sys.argv) < 6:
        print("Usage: python evaluate_drillhole_model.py <model_dir> <domain_width> <domain_height> <grid_resolution> <output_path>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    domain_width = float(sys.argv[2])
    domain_height = float(sys.argv[3])
    grid_resolution = int(sys.argv[4])
    output_path = sys.argv[5]
    
    # Find config file
    config_path = os.path.join(model_dir, "config.yml")
    if not os.path.exists(config_path):
        config_files = glob.glob(os.path.join(model_dir, "*config.yml"))
        if config_files:
            config_path = config_files[0]
        else:
            print(f"ERROR: Config file not found in {model_dir}", file=sys.stderr)
            sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract problem config
    problem_config = config.get('problem', {})
    nx = config.get('nx', 2)
    nz = config.get('nz', 1)
    
    problem_config.update({
        'nx': nx,
        'n_points_domain': config.get('problem_sampling', {}).get('n_points_domain', 2048),
        'n_points_envelope': config.get('problem_sampling', {}).get('n_points_envelope', 4096),
        'n_points_interfaces': config.get('problem_sampling', {}).get('n_points_interfaces', 2048),
    })
    
    # Create problem instance
    problem = get_problem(problem_config)
    
    # Find checkpoint file
    checkpoint_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_files = glob.glob(os.path.join(model_dir, "*model.pt"))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
        else:
            print(f"ERROR: Checkpoint file not found in {model_dir}", file=sys.stderr)
            sys.exit(1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    init_params = checkpoint['init_params']
    state_dict = checkpoint['state_dict']
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Init params: {init_params}")
    
    # Create model from init_params
    model = ConditionalWIRE(**init_params)
    
    # Try to load state_dict - this might fail if architecture doesn't match
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded model state")
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("Attempting partial load...")
        # Try loading with strict=False
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Create grid in domain coordinates (mm)
    x = np.linspace(0, domain_width, grid_resolution)
    y = np.linspace(0, domain_height, grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # [N, 2]
    
    # Normalize coordinates to GINN's coordinate system
    f_scale = max(domain_width, domain_height) / 2.0
    center = np.array([domain_width / 2.0, domain_height / 2.0])
    normalized_points = (grid_points - center) / f_scale
    
    # Convert to tensor - x should be 2D coordinates
    x_tensor = torch.tensor(normalized_points, dtype=torch.float32)
    
    # For 2D problems with nz=1, z should be a 1D latent vector
    # We use z=0 for unconditioned evaluation
    z_tensor = torch.zeros((x_tensor.shape[0], 1), dtype=torch.float32)
    
    print(f"x tensor shape: {x_tensor.shape}")
    print(f"z tensor shape: {z_tensor.shape}")
    with torch.no_grad():
        sdf_values = model(x_tensor, z_tensor).squeeze().numpy()
    
    # Reshape to 2D grid
    sdf_grid = sdf_values.reshape(grid_resolution, grid_resolution)
    
    # Save to numpy file
    np.save(output_path, sdf_grid)
    print(f"SDF grid saved to {output_path} with shape {sdf_grid.shape}")
    print(f"SDF range: [{sdf_grid.min():.6f}, {sdf_grid.max():.6f}]")


if __name__ == "__main__":
    main()