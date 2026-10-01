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
from ising.stages.simulation_stage import SimulationStage
from ising.stages.model.ising import IsingModel
from ising.utils.flow import parse_hyperparameters

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "Maxcut"  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, MPPI]
config_path = "ising/inputs/config/example.yaml"

# Set to an integer for reproducible random partitioning.
PARTITION_SEED = 42

# If True, use absolute Ising coupling strength as the graph edge weight.
# This is useful for weighted-cut partitioning diagnostics; the initial
# partition itself below is random and balanced.
USE_ABS_COUPLING_WEIGHTS = True


# ---------------------------------------------------------------------------
# Helpers for Partitioning
# ---------------------------------------------------------------------------

def solve_once(stage, model, hyperparameters, seed, initial_state=None):
    """Run the configured bSB solver once, using OpenIsing's own dispatcher."""
    if initial_state is None:
        rng = np.random.default_rng(seed)
        initial_state = rng.uniform(-1.0, 1.0, model.num_variables)

    # Give each trial its own solver seed as well as its own initialization.
    # bSB uses its seed for its internal random momentum initialization.
    trial_hyperparameters = dict(hyperparameters)
    trial_hyperparameters["seed"] = int(seed)

    stop_crit = (
                        stage.config.stop_criterion_iterations
                        if hasattr(stage.config, "stop_criterion_iterations")
                        else False
                    )

    # run_solver dispatches the solver and passes the SB iteration count,
    # bSB parameters, stop criterion, and other supported options.
    return stage.run_solver(
        solver="bSB",
        s_init=np.asarray(initial_state),
        model=model,
        logfile=None,
        stop_criterion_it=stop_crit,
        **trial_hyperparameters,
    )


#Function that will be used for estimating how much information is lost when partitioning
def build_weighted_graph(model):
    """
    Build an undirected NetworkX graph from the Ising coupling matrix.
    The IsingModel used by this project stores the relevant couplings in
    the upper triangle; use each i<j pair once.
    """
    import networkx as nx

    J = np.asarray(model.J)
    graph = nx.Graph()
    graph.add_nodes_from(range(model.num_variables))

    for i in range(model.num_variables):
        for j in range(i + 1, model.num_variables):
            coupling = float(J[i, j])
            if coupling != 0.0:
                weight = abs(coupling) if USE_ABS_COUPLING_WEIGHTS else 1.0
                graph.add_edge(i, j, weight=weight, coupling=coupling)

    return graph


def random_balanced_partition(num_variables, seed):
    """Return two nonempty, nearly equal sorted arrays of zero-based matrix indices."""
    if num_variables < 2:
        raise ValueError("Partitioning requires at least two variables.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_variables)
    rng.shuffle(indices)

    split = num_variables // 2
    part_A = np.sort(indices[:split])
    part_B = np.sort(indices[split:])
    return part_A, part_B


def make_induced_ising_model(model, indices, name):
    """
    Construct an IsingModel for one induced subproblem.

    Preserve the project's energy convention:
        E(s) = -s.T @ J @ s - h.T @ s + c
    For the current dummy Max-Cut generator, J is upper triangular and
    c is the sum of its upper-triangular entries.
    """
    indices = np.asarray(indices, dtype=int)
    J_sub = np.asarray(model.J)[np.ix_(indices, indices)].copy()
    h_sub = np.asarray(model.h)[indices].copy()

    # Match the dummy Max-Cut convention, while also supporting a general
    # matrix representation by summing the upper triangle only.
    c_sub = float(np.sum(np.triu(J_sub, k=1)))

    return IsingModel(
        J_sub,
        h_sub,
        c_sub,
        name=name,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("Creating the original Ising model using the existing OpenIsing API...")
    ans, debug_info = api.get_hamiltonian_energy(
        problem_type=problem_type,
        config_path=config_path,
        logging_level=logging_level,
    )
    print("Unieke waarden in bSB state:", np.unique(ans.states["bSB"][0]))
    # 1) Exact optimum: test all 2^N possible spin configurations 
    model = ans.ising_model
    N = model.num_variables

    # 1. Controleer of J inderdaad strikt bovendriehoekig is (onderdriehoek k < 0 moet 0 zijn)
    is_upper_triangular = bool(np.all(np.tril(model.J, k=-1) == 0))

    # 2. Controleer of model.c overeenkomt met de som van de bovendriehoek
    expected_c = float(np.sum(np.triu(model.J, k=1)))
    c_matches = bool(np.isclose(model.c, expected_c))

    print(f"Controle model.J is bovendriehoekig: {is_upper_triangular}")
    print(f"Controle model.c komt overeen ({model.c} vs {expected_c}): {c_matches}")

    assert is_upper_triangular, "Fout: model.J is niet strikt bovendriehoekig!"
    assert c_matches, f"Fout: model.c ({model.c}) wijkt af van de som van J ({expected_c})!"

    best_energy = float("inf")
    best_state = None

    for state in itertools.product([-1, 1], repeat=N):
        state = np.array(state, dtype=np.float32)
        energy = model.evaluate(state)

        if energy < best_energy:
            best_energy = energy
            best_state = state.copy()

    """
    print("\n=== EXACT OPTIMUM ===")
    print("Exact best energy:", best_energy)
    print("Exact best state:", best_state)
    """

    # 2) Full bSB results directly from API simulation 
    energies = ans.energies["bSB"]
    optimal_runs = sum(energy == best_energy for energy in energies)

    """
    print("\n=== bSB RESULTS ===")
    print("Number of runs:", len(energies))
    print("Best bSB energy:", min(energies))
    print("Optimal runs:", optimal_runs)
    print("Success rate:", (optimal_runs / len(energies)) * 100, "%")
    
    """
    # Output summary file (From Original Code)
    output_file = Path(f"./simulation_summary_{ans.benchmark}.pkl")
    summarize_runs(output_file, ans, problem_type, config_path)


    # -----------------------------------------------------------------------
    # 3) Partitioned Code 
    # -----------------------------------------------------------------------
    config = ans.config
    n = model.num_variables

    """
    if "bSB" not in config.solvers:
        raise ValueError(
            "This experiment currently requires 'bSB' in the config's solvers list."
        )
    """

    # Gebruik project eigen config logic
    hyperparameters = parse_hyperparameters(config)
    print(f"Hyperparameters for bSB: {hyperparameters}")
    nb_runs = len(energies)  # Synchroon gehouden met originele run lengte

    # Graaf genereren en partitioneren
    graph = build_weighted_graph(model)
    print(
        f"\nDerived graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} nonzero-coupling edges"
    )

    part_A, part_B = random_balanced_partition(n, PARTITION_SEED)
    print(f"\nPartition A ({len(part_A)} variables): {part_A.tolist()}")
    print(f"Partition B ({len(part_B)} variables): {part_B.tolist()}")

    set_A, set_B = set(part_A.tolist()), set(part_B.tolist())
    cut_edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if (u in set_A and v in set_B) or (u in set_B and v in set_A)
    ]
    cut_weight = sum(float(data["weight"]) for _, _, data in cut_edges)
    print(
        f"Cut edges: {len(cut_edges)}; "
        f"sum of absolute coupling weights across cut: {cut_weight:.6g}"
    )

    model_A = make_induced_ising_model(model, part_A, "Partition_A")
    model_B = make_induced_ising_model(model, part_B, "Partition_B")

    # Instantiate stages that reuse SimulationStage.run_solver().
    stage_A = SimulationStage(
        list_of_callables=[],
        config=config,
        ising_model=model_A,
    )
    stage_B = SimulationStage(
        list_of_callables=[],
        config=config,
        ising_model=model_B,
    )

    print(
        f"\n=== Partitioned bSB: {nb_runs} paired trials "
        f"on {len(part_A)} + {len(part_B)} variables ==="
    )
    partitioned_states = []
    partitioned_energies = []
    partitioned_times = []

    for run_idx in range(nb_runs):
        # To make sure that they have their own unique initialization (which is also different from the original full bSB))
        seed_A = 20_000 + 2 * run_idx
        seed_B = seed_A + 1

        state_A, energy_A, time_A, _, iter_A = solve_once(
            stage_A, model_A, hyperparameters, seed_A
        )
        state_B, energy_B, time_B, _, iter_B = solve_once(
            stage_B, model_B, hyperparameters, seed_B
        )

        merged_state = np.zeros(n, dtype=np.float32)
        merged_state[part_A] = np.asarray(state_A, dtype=np.float32)
        merged_state[part_B] = np.asarray(state_B, dtype=np.float32)

        # Evalueer tegen de ORIGINELE full model (inclusief cross-partition couplings)
        merged_energy = float(model.evaluate(merged_state))

        partitioned_states.append(merged_state)
        partitioned_energies.append(merged_energy)
        partitioned_times.append(float(time_A) + float(time_B))

        print(
            f"  paired trial {run_idx + 1:>3}/{nb_runs}: "
            f"E_A={float(energy_A):.6g}, E_B={float(energy_B):.6g}, "
            f"E_merged(full J)={merged_energy:.6g}, "
            f"iterations={iter_A}+{iter_B}"
        )

    best_partition_idx = int(np.argmin(partitioned_energies))
    best_partition_state = partitioned_states[best_partition_idx]
    best_partition_energy = partitioned_energies[best_partition_idx]

    # Voor de overzichtelijkheid pakken we de states op uit de al gedraaide run
    full_states = ans.states["bSB"]
    full_energies = ans.energies["bSB"]
    best_full_idx = int(np.argmin(full_energies))
    best_full_energy = full_energies[best_full_idx]
    best_full_state = full_states[best_full_idx]


    # Summary print & logging.
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Variables: {n}")
    print(f"Runs: {nb_runs}")
    print(f"Partition sizes: {len(part_A)} + {len(part_B)}")
    print(f"Cut edges: {len(cut_edges)}")
    print(f"Cut absolute weight: {cut_weight:.6g}")
    print(f"Best full bSB energy: {best_full_energy:.6g}")
    print(f"Mean full bSB energy: {np.mean(full_energies):.6g}")
    print(f"Best partitioned energy (evaluated on full J): {best_partition_energy:.6g}")
    print(f"Mean partitioned energy: {np.mean(partitioned_energies):.6g}")

    if best_energy is not None:
        print(f"Exact optimum: {best_energy:.6g}")
        print(
            "Full bSB gap to optimum: "
            f"{best_full_energy - best_energy:.6g}"
        )
        print(
            "Partitioned gap to optimum: "
            f"{best_partition_energy - best_energy:.6g}"
        )
        print(
            "Full bSB optimal hits: "
            f"{sum(np.isclose(e, best_energy) for e in full_energies)}"
            f"/{nb_runs}"
        )
        print(
            "Partitioned optimal hits: "
            f"{sum(np.isclose(e, best_energy) for e in partitioned_energies)}"
            f"/{nb_runs}"
        )

    print("\nBest full bSB state:")
    print(best_full_state)
    print("\nBest partitioned merged state:")
    print(best_partition_state)
    print("\nExact optimal state:")
    print(best_state)

    # Save a compact, reproducible record of this experiment.
    output_path = "./partitioning_comparison_1.npz"
    np.savez(
        output_path,
        part_A=part_A,
        part_B=part_B,
        full_bsb_states=np.asarray(full_states),
        full_bsb_energies=np.asarray(full_energies),
        partitioned_states=np.asarray(partitioned_states),
        partitioned_energies=np.asarray(partitioned_energies),
        exact_state=np.asarray(best_state) if best_state is not None else np.asarray([]),
        exact_energy=best_energy if best_energy is not None else np.nan,
        cut_edge_count=len(cut_edges),
        cut_abs_weight=cut_weight,
    )
    print(f"\nSaved experiment results to: {output_path}")


if __name__ == "__main__":
    main()