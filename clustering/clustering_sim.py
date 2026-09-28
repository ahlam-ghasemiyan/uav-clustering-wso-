# clustering/clustering_sim.py
import argparse
import json
import numpy as np
import os
from clustering.utils import generate_uavs
from clustering.wso import WhiteSharkOptimizer

def main():
    parser = argparse.ArgumentParser(description="WSOODL-UAVCSC Clustering Simulation")
    
    # Simulation Setup Parameters [cite: 396-403]
    parser.add_argument('--n', type=int, default=40, help='Number of UAV nodes (Paper suggests small scale, e.g., 10-50)')
    parser.add_argument('--area_size', type=int, default=1000, help='Simulation area size in meters')
    parser.add_argument('--bs_x', type=float, default=0.0, help='Base Station X coord ')
    parser.add_argument('--bs_y', type=float, default=0.0, help='Base Station Y coord ')
    parser.add_argument('--comm_range', type=float, default=200.0, help='Communication Range (Rc) ')
    
    # WSO Algorithm Parameters 
    parser.add_argument('--pop', type=int, default=50, help='Population size')
    parser.add_argument('--gen', type=int, default=100, help='Max iterations (MaxGen)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # 1. Generate UAVs (Nodes) with Energy and Position
    # UAVs have initial energy and random positions
    print(f"Generating {args.n} UAVs in {args.area_size}x{args.area_size} area...")
    nodes = generate_uavs(n=args.n, area_size=args.area_size, seed=args.seed)
    
    # Define Base Station Position
    bs_position = np.array([args.bs_x, args.bs_y])
    print(f"Base Station located at: {bs_position}")

    # 2. Initialize White Shark Optimizer (WSO)
    # WSO optimizes the objective function F (Eq. 14 in paper)
    print("Initializing White Shark Optimizer (WSO)...")
    wso = WhiteSharkOptimizer(
        nodes=nodes,
        bs_position=bs_position,
        comm_range=args.comm_range,
        population_size=args.pop,
        max_gen=args.gen,
        seed=args.seed
    )

    # 3. Run Optimization
    print("Running Clustering Optimization...")
    res = wso.run(verbose=True)

    # 4. Display Results
    print("\n" + "="*30)
    print(f"Best Fitness Value: {res['gbest_val']:.4f}")
    print(f"Optimal Cluster Heads (Indices): {res['ch_indices']}")
    print("="*30 + "\n")

    # 5. Save Results for Visualization/Analysis
    # Preparing data structure
    out = {
        'parameters': vars(args),
        'best_fitness': res['gbest_val'],
        'ch_indices': [int(x) for x in res['ch_indices']], # Convert numpy ints to python ints
        'cluster_map': {str(k): [int(x) for x in v] for k, v in res['cluster_map'].items()},
        'nodes': [
            {'id': i, 'x': n.x, 'y': n.y, 'energy': n.energy} 
            for i, n in enumerate(nodes)
        ]
    }

    # Ensure output directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
        
    out_path = 'results/clustering_result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    
    print(f"Result saved to '{out_path}'")

if __name__ == '__main__':
    main()