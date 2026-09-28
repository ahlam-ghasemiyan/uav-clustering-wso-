# clustering/wso.py
import numpy as np
from clustering.fitness import objective_F
from clustering.utils import euclidean

class WhiteSharkOptimizer:
    def __init__(self, nodes, bs_position, comm_range, population_size=50, max_gen=100, seed=42):
        """
        White Shark Optimizer for UAV Clustering.
        Based on WSOODL-UAVCSC paper.
        """
        self.nodes = nodes
        self.bs_position = bs_position
        self.comm_range = comm_range
        self.n_nodes = len(nodes)
        
        # WSO Parameters
        self.pop_size = population_size
        self.max_gen = max_gen
        self.rng = np.random.RandomState(seed)
        
        # Search Space: A continuous value [0, 1] for each UAV.
        self.dim = self.n_nodes 
        self.lb = 0.0
        self.ub = 1.0
        
        # Initialize White Sharks (Positions and Velocities)
        self.pos = self.rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.vel = np.zeros_like(self.pos)
        
        # Best Solutions
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_val = -np.inf # We want to MAXIMIZE fitness
        
        # Personal bests
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.full(self.pop_size, -np.inf)

        # WSO Constants (Paper Eq constants)
        self.f_min = 0.07
        self.f_max = 0.75
        self.tau = 4.125
        self.p_min = 0.5
        self.p_max = 1.5
        self.a0 = 6.250
        self.a1 = 100

    def _form_clusters(self, position_vector):
        """
        Decodes the WSO position vector into a Cluster Topology.
        1. Select top K nodes as CHs based on position values.
        2. Assign other nodes to the nearest CH within range.
        """
        # CHopt is roughly 5% to 10% of n
        n_ch = max(1, int(0.10 * self.n_nodes)) 
        
        # Select indices of nodes with highest scores in position_vector
        ch_indices = np.argsort(position_vector)[-n_ch:]
        ch_indices = [int(i) for i in ch_indices]
        
        cluster_map = {ch: [] for ch in ch_indices}
        
        # Assign members to nearest CH
        for i in range(self.n_nodes):
            if i in ch_indices:
                continue
                
            my_pos = np.array([self.nodes[i].x, self.nodes[i].y])
            best_ch = -1
            min_dist = float('inf')
            
            for ch in ch_indices:
                ch_pos = np.array([self.nodes[ch].x, self.nodes[ch].y])
                dist = euclidean(my_pos, ch_pos)
                
                # Check range and distance
                if dist <= self.comm_range and dist < min_dist:
                    min_dist = dist
                    best_ch = ch
            
            if best_ch != -1:
                cluster_map[best_ch].append(i)
                
        return ch_indices, cluster_map

    def run(self, verbose=True):
        
        for t in range(self.max_gen):
            # 1. Update Fitness and Global Best
            for i in range(self.pop_size):
                # Clamp position to bounds
                self.pos[i] = np.clip(self.pos[i], self.lb, self.ub)
                
                # Decode and Evaluate
                ch_idx, cl_map = self._form_clusters(self.pos[i])
                fitness = objective_F(self.nodes, ch_idx, cl_map, self.bs_position)
                
                # Update Pbest
                if fitness > self.pbest_val[i]:
                    self.pbest_val[i] = fitness
                    self.pbest_pos[i] = self.pos[i].copy()
                
                # Update Gbest
                if fitness > self.gbest_val:
                    self.gbest_val = fitness
                    self.gbest_pos = self.pos[i].copy()
            
            if verbose:
                print(f"WSO Gen {t+1}/{self.max_gen} | Best Fitness: {self.gbest_val:.4f}")

            # 
            # 2. Update WSO Parameters
            # Eq (2) & (3)
            term = (4 * t / self.max_gen) ** 2
            p1 = self.p_max + (self.p_max - self.p_min) * np.exp(-term)
            p2 = self.p_min + (self.p_max - self.p_min) * np.exp(-term)
            
            # Eq (4) - mu calculation
            try:
                mu = 2.0 / np.abs(2.0 - self.tau - np.sqrt(self.tau**2 - 4*self.tau))
            except:
                mu = 0.5 

            # 3. Update Position of Sharks
            for i in range(self.pop_size):
                # --- Eq (1): Update Velocity ---
                c1 = self.rng.rand(self.dim)
                c2 = self.rng.rand(self.dim)
                
                # v(t+1) logic: based on previous velocity (speed)
                v_new = mu * (self.vel[i] + p1 * (self.gbest_pos - self.pos[i]) * c1 + p2 * (self.pbest_pos[i] - self.pos[i]) * c2)
                self.vel[i] = v_new
                
                # --- Eq (5): Movement towards prey ---
                f = self.f_min + (self.f_max - self.f_min) / (self.f_max + self.f_min)
                
                # mv calculation
                mv = 1.0 / (self.a0 + np.exp((self.max_gen/2.0 - t) / self.a1))
                
                if self.rng.rand() < mv:
                    # Move towards prey
                    self.pos[i] = self.pos[i] + self.vel[i] / f
                else:
                    # Random search behavior
                    self.pos[i] = self.pos[i] + self.rng.uniform(-1, 1, self.dim)
                    
                # --- Eq (9): Fish School Behavior ---
                if self.rng.rand() < 0.5: 
                    r = self.rng.rand()
                    self.pos[i] = (self.pos[i] + self.gbest_pos) / (2 * r)

        # Final Extraction
        best_ch_indices, best_cluster_map = self._form_clusters(self.gbest_pos)
        
        return {
            'gbest_val': self.gbest_val,
            'gbest_pos': self.gbest_pos,
            'ch_indices': best_ch_indices,
            'cluster_map': best_cluster_map
        }