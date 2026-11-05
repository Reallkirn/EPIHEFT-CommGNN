# EpiHEFT-CommGNN

This repository accompanies the EpiHEFT-CommGNN research paper. It provides the pretrained main-model checkpoint and evaluation code for the Pegasus-S main experiment, together with the HEFT_Comm, PEFT, CPOP, and IPPTS-Dyn baselines.

The current release supports inference and evaluation. Additional code is being prepared for release.

## Repository structure

```text
EPIHEFT-CommGNN/
├── README.md
├── requirements.txt               # Pinned Python dependencies
├── run.py                         # Evaluation CLI, parallel workers, and CSV output
├── model.py                       # Actor network and observation-to-graph wrapper
├── settings.py                    # Fixed main-experiment inference settings
├── main_actor.pth                 # Pretrained main-model checkpoint (seed 0)
└── env/
    └── workflow_scheduling_v3/
        ├── simulator_wf.py        # Scheduling environment and baseline interfaces
        ├── dax/                   # Bundled Pegasus-S workflow definitions
        │   ├── CyberShake_30.xml
        │   ├── Montage_25.xml
        │   ├── Inspiral_30.xml
        │   └── Sipht_30.xml
        └── lib/
            ├── cloud_env_maxPktNum.py  # Workflow execution simulator
            ├── communication.py       # Network and communication model
            ├── ippts_dynamic.py       # IPPTS-Dyn baseline
            ├── dataset.py             # Workflow loading and VM specifications
            ├── processDAG.py          # DAG parsing and scheduling utilities
            ├── poissonSampling.py     # Workflow arrival sampling
            ├── simsetting.py          # Simulation setup
            ├── simqueue.py            # Simulation queue utilities
            ├── stats.py               # Simulation statistics
            ├── vm.py                  # Virtual machine model
            └── workflow.py            # Workflow model
```

Evaluation results are generated at runtime and are not part of the source release.

## Installation

Linux and Python 3.12 are recommended. The reference environment uses Python 3.12.3, PyTorch 2.8.0, and PyTorch Geometric (PyG) 2.5.3.

Run the following commands from the repository root to install the CPU inference environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For a CUDA-capable machine, replace the PyTorch installation command above with the CUDA 12.8 wheel installation:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Quick start

Run one complete scheduling instance, comparing the pretrained model with HEFT_Comm:

```bash
python run.py --methods main,HEFT_Comm --H 0 --n 1 --workers 2
```

Each instance contains 30 workflows and runs until all workflows finish. This single-instance command is a smoke test; it does not reproduce the full main experiment.

The default checkpoint is the bundled `main_actor.pth`. To select a checkpoint and run inference on a GPU:

```bash
python run.py --weights main_actor.pth --device cuda:0 --methods main --H 0 --n 1
```

## Main-experiment evaluation

Evaluate the main model and HEFT_Comm on all 30 held-out instances for each heartbeat interval:

```bash
python run.py --methods main,HEFT_Comm --H 0,3,5,10 --n 30 --workers 8
```

Include all supported baselines:

```bash
python run.py --methods main,HEFT_Comm,PEFT,CPOP,IPPTS_Dyn --H 0,3,5,10 --n 30 --workers 8
```

The command-line identifier for IPPTS-Dyn is `IPPTS_Dyn`. Each worker uses one CPU thread; adjust `--workers` to the available CPU and memory resources. CPU inference is the default.

### Fixed evaluation protocol

| Setting | Value |
| --- | --- |
| Main-model checkpoint | Seed 0 |
| Workflow dataset | Pegasus-S: CyberShake, Montage, Inspiral, and Sipht |
| Workflows per instance | 30 |
| Virtual machines, N | 10 |
| Contention parameter, α | 1.5 |
| Runtime heuristic prior weight, p_w | 0.10 for the main model |
| Workflow arrival rate | 5.4 workflows/hour |
| Held-out instance indices | 30–59 |
| Heartbeat intervals, H | 0, 3, 5, 10 |

`--n` selects the first n held-out instances, with valid values from 1 to 30. Workflow selections and arrival times are reconstructed in memory using fixed seeds. The workflow XML files are bundled, so no external dataset download, imitation-learning data, or precomputed cache is required.

### Outputs

Each run writes to a new `results/<timestamp>/` directory by default. Use `--output <directory>` to select a new or empty output directory.

| File | Contents |
| --- | --- |
| `run.json` | Run arguments, fixed experiment settings, test indices, and checkpoint checksum |
| `per_instance.csv` | Per-instance mean flow completion time (FCT), completed workflow count, decision count, runtime, and reproducibility hashes |
| `summary.csv` | Mean FCT across instances, population standard deviation (`ddof=0`), and percentage gap relative to HEFT_Comm |

The terminal also prints per-instance progress and summary metrics. The gap is computed as `100 × (method mean FCT / HEFT_Comm mean FCT − 1)`; a negative value indicates lower FCT than HEFT_Comm. If HEFT_Comm is not included in the run, the gap is reported as unavailable.
