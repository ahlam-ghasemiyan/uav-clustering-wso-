# clustering/fitness.py
import numpy as np

def euclidean(pos1, pos2):
    """Calculates Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def f1_energy_efficiency(nodes, ch_indices):
    """
    Eq (10): Energy Efficiency 
    Ratio of CHs' residual energy to the network's average energy.
    Higher is better.
    """
    if len(ch_indices) == 0:
        return 0.0
    
    # میانگین انرژی کل شبکه
    avg_network_energy = np.mean([n.energy for n in nodes])
    if avg_network_energy == 0:
        return 0.0
        
    # مجموع انرژی کلاستر هد‌های انتخاب شده
    total_ch_energy = sum([nodes[i].energy for i in ch_indices])
    
    #(Sum of CH Energy) / Average Network Energy
    f1 = total_ch_energy / avg_network_energy
    return f1

def f2_cluster_node_density(cluster_map):
    """
    Eq (11): Cluster Node Density [cite: 273-276]
    The algorithm prioritizes forming dense clusters.
    Returns the maximum size among all clusters.
    Higher is better (based on Eq 14 structure).
    """
    if not cluster_map:
        return 0.0
    
    # تعداد اعضای هر کلاستر
    counts = [len(members) for members in cluster_map.values()]
    
    # طبق فرمول (11): max(n(CH_j))
    f2 = max(counts) if len(counts) > 0 else 0.0
    return float(f2)

def f3_intra_cluster_distance(nodes, ch_indices, cluster_map):
    """
    Eq (12): Average distance of UAVs to their CHs [cite: 279-284]
    Lower is better.
    """
    if not cluster_map:
        return float(1e6) # پنالتی بزرگ برای حالت بدون کلاستر

    total_dist = 0.0
    total_nodes = 0
    
    for ch_idx in ch_indices:
        members = cluster_map.get(ch_idx, [])
        ch_pos = np.array([nodes[ch_idx].x, nodes[ch_idx].y])
        
        for m_idx in members:
            m_pos = np.array([nodes[m_idx].x, nodes[m_idx].y])
            total_dist += euclidean(ch_pos, m_pos)
            total_nodes += 1
            
    if total_nodes == 0:
        return float(1e6)
        
    f3 = total_dist / total_nodes
    return f3

def f4_distance_ch_to_bs(nodes, ch_indices, bs_position):
    """
    Eq (13): Distance from CH to Base Station [cite: 287-289]
    Lower is better.
    """
    if len(ch_indices) == 0:
        return float(1e6)

    total_dist = 0.0
    for ch_idx in ch_indices:
        ch_pos = np.array([nodes[ch_idx].x, nodes[ch_idx].y])
        total_dist += euclidean(ch_pos, bs_position)
        
    f4 = total_dist / len(ch_indices)
    return f4

def objective_F(nodes, ch_indices, cluster_map, bs_position, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Eq (14): Maximize Fitness Function 
    F = alpha*f1 + beta*f2 + gamma*(1/f3) + delta*(1/f4)
    """
    alpha, beta, gamma, delta = weights
    
    f1 = f1_energy_efficiency(nodes, ch_indices)
    f2 = f2_cluster_node_density(cluster_map)
    f3 = f3_intra_cluster_distance(nodes, ch_indices, cluster_map)
    f4 = f4_distance_ch_to_bs(nodes, ch_indices, bs_position)
    
    # مخرج‌ها نباید صفر باشند (1e-9 برای پایداری)
    inv_f3 = 1.0 / (f3 + 1e-9)
    inv_f4 = 1.0 / (f4 + 1e-9)
    
    # محاسبه نهایی فیتنس
    fitness = (alpha * f1) + (beta * f2) + (gamma * inv_f3) + (delta * inv_f4)
    
    return float(fitness)