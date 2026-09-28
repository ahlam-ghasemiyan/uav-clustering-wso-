🚁 UAV Clustering using White Shark Optimizer (WSO) An energy-aware clustering framework for UAV networks powered by the White Shark Optimizer — a bio-inspired metaheuristic algorithm.

📖 Overview This project implements an energy-efficient clustering scheme for Unmanned Aerial Vehicle (UAV) networks using the White Shark Optimizer (WSO) — a nature-inspired metaheuristic algorithm.

The algorithm selects an optimal subset of UAVs to act as Cluster Heads (CHs), minimizing a multi-objective fitness function that balances energy consumption, cluster density, and communication distance.

🎯 Goal: Prolong network lifetime and reduce communication overhead in UAV-assisted wireless networks.

🧠 Algorithms Used 🦈 White Shark Optimizer (WSO) A bio-inspired metaheuristic that mimics the hunting behavior of great white sharks, combining:

Exploration — global search for optimal prey positions

Exploitation — refined movement near best solutions

Adaptive switching — balancing exploration vs. exploitation

Each shark represents a candidate set of Cluster Heads. The optimizer iteratively refines this set to minimize the fitness function.

🎯 Multi-Objective Fitness Function F
α ⋅ f 1 + β ⋅ f 2 + γ ⋅ 1 f 3 + δ ⋅ 1 f 4 F=α⋅f 1​+β⋅f 2​+γ⋅ f 3​

1​+δ⋅ f 4​

1​

Symbol Criterion Goal f₁ Energy efficiency Maximize f₂ Cluster node density Minimize f₃ Avg. distance (member → CH) Minimize f₄ Avg. distance (CH → BS) Minimize Default weights: α = β = γ = δ = 0.25

✨ Features 🧠 Full WSO implementation from scratch

🎯 Multi-objective fitness with 4 weighted criteria

⚙️ Configurable via command-line arguments

🔁 Reproducible experiments with random seeds

💾 JSON output for downstream analysis

📦 Modular architecture — clean and extensible

🛠 Tech Stack Component Purpose Python 3.9+ Core implementation NumPy Numerical computation argparse CLI parsing JSON Result serialization

📁 Project Structure text uav-clustering-wso/ ├── clustering/ │ ├── init.py │ ├── clustering_sim.py # Main entry point │ ├── fitness.py # Objective function │ ├── utils.py # Helpers (UAV generation, distance) │ └── wso.py # White Shark Optimizer ├── requirements.txt ├── README.md ├── LICENSE └── .gitignore 📜 License All Rights Reserved.

Copyright (c) 2025 Eng. Ghasemian

This project is intended solely for personal and educational purposes. Unauthorized copying, distribution, or commercial use is strictly prohibited.

👨‍🏫 Author Engineer Ahlam Ghasemiyan Developed as part of an academic research project on UAV network optimization.
