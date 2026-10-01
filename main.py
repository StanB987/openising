import logging
import sys
from pathlib import Path
import os

import itertools
import numpy as np

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

from ising import api
from ising.postprocessing.run_summary import summarize_runs


# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "Maxcut"  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, MPPI]
config_path = "ising/inputs/config/example.yaml"

# Run the Ising model simulation
ans, debug_info = api.get_hamiltonian_energy(
    problem_type=problem_type,
    config_path=config_path,
    logging_level=logging_level,
)

# Exact optimum: test all 2^N possible spin configurations
model = ans.ising_model
N = model.num_variables

best_energy = float("inf")
best_state = None

for state in itertools.product([-1, 1], repeat=N):
    state = np.array(state, dtype=np.float32)
    energy = model.evaluate(state)

    if energy < best_energy:
        best_energy = energy
        best_state = state.copy()

print("\n=== EXACT OPTIMUM ===")
print("Exact best energy:", best_energy)
print("Exact best state:", best_state)


energies = ans.energies["bSB"]
optimal_runs = sum(energy == best_energy for energy in energies)

print("\n=== bSB RESULTS ===")
print("Number of runs:", len(energies))
print("Best bSB energy:", min(energies))
print("Optimal runs:", optimal_runs)
print("Success rate:", optimal_runs / len(energies) * 100, "%")

#print("\n=== bSB RESULTS ===")
#print("Energies:", ans.energies["bSB"])
#print("Best bSB energy:", min(ans.energies["bSB"]))

# Output summary file
#print("Benchmark:", ans.benchmark)
#print("Energies:", ans.energies)
#print("States:", ans.states)


output_file = Path(f"./simulation_summary_{ans.benchmark}.pkl")
summarize_runs(output_file, ans, problem_type, config_path)
