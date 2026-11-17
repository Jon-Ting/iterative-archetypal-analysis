# IAA Complexity Scaling

Benchmarking scripts that produce the computational-complexity and parallel
scaling results reported in `iaa-manuscripts/IAA_manuscript.tex`.

The benchmarks cover the parallel Iterative Archetypal Analysis (IAA)
algorithm only, in three experiments: time complexity, strong scaling, and
weak scaling.

---

## Files

| File | Purpose |
|---|---|
| `runTC_IAA.py` | Time-complexity benchmark for **Iterative AA** |
| `runSS.py` | **Strong-scaling** benchmark (fixed problem, vary CPUs/nodes) |
| `runWS.py` | **Weak-scaling** benchmark (scale both problem and CPUs) |
| `plotTC.py` | Reads TC output files and generates the TC figures |
| `plotSS_WS.py` | Reads SS/WS output files and generates the scaling figures |
| `submitTC.sh` | Submits **all** PBS jobs to Gadi (TC + SS + WS) |
| `complexityScaling.ipynb` | Full interactive notebook (reads same output files) |

---

## Experiment Design

### Time Complexity (TC)

| Parameter | Value |
|---|---|
| Number of features (m) | 1000 |
| Number of samples (n) swept | 100, 200, 400, 800, 1600, 3200 |
| Number of archetypes (p) | n / 10 |
| Number of IAA subsets (k) | 10 |
| Repetitions | 30 independent runs |
| Random seed | 42 |

### Strong Scaling (SS)

| Parameter | Value |
|---|---|
| Fixed dataset | 6400 samples, 1000 features |
| Number of archetypes | 640 (= 6400 / 10) |
| Node counts swept | 1, 2, 4, 8, 16, 32 |
| Total CPUs | 48, 96, 192, 384, 768, 1536 |
| CPUs per node | 48 (Gadi normal queue) |
| Subsets per run | 1 subset per node, 48 threads per subset |
| Wall-clock model | T_wall ~= T_one_subset + T_final_AA |

### Weak Scaling (WS)

| Ncpus | Nodes | Subsets | Samples (n) |
|---|---|---|---|
| 1  | 1 | 1 | 43   |
| 2  | 1 | 1 | 87   |
| 4  | 1 | 1 | 177  |
| 8  | 1 | 1 | 321  |
| 16 | 1 | 1 | 683  |
| 24 | 1 | 1 | 1055 |
| 48 | 1 | 2 | 1961 |
| 96 | 2 | 4 | 4189 |

Sample counts are sized so that samples per core stays roughly constant
(≈ 43) at every configuration, which is what keeps the per-core workload
fixed for a weak-scaling experiment.

---

## Output File Format

All benchmark scripts write one line per run:

```
Run 0: 1.234 s
Run 1: 1.198 s
...
```

The notebook and plot scripts parse lines matching `Run ` and extract the
duration as the third whitespace-delimited token.

### Expected directory layout after all runs

```
iaa-complexity-scaling/
├── data/runTimes/computationalComplexity/
│   └── TC_PIAA_{100,200,400,800,1600,3200}.out
├── data/runTimes/strongScaling/
│   └── SS_IAA_{48,96,192,384,768,1536}.out
└── data/runTimes/weakScaling/
    └── WS_IAA_{1,2,4,8,16,24,48,96}.out
```

---

## How to Run

### Local / interactive

```bash
cd iaa-complexity-scaling
source .venv/bin/activate

python runTC_IAA.py        # TC for IAA
python runSS.py            # Strong scaling (simulates parallel wall-clock locally)
python runWS.py            # Weak scaling

python plotTC.py           # -> TC_PIAA.png, TC_PIAA_loglog.png, TC_PIAA_BoxPlots.png
python plotSS_WS.py        # -> SS_IAA_*.png, WS_IAA_Efficiency.png
```

### On Gadi (HPC)

One-time environment setup (the venv must be built with the same Python module
the jobs load, here `python3/3.11.7`):

```bash
cd iaa-complexity-scaling
module load python3/3.11.7
python3 -m venv .venv
.venv/bin/pip install -e ../iteraa      # local iteraa, not the PyPI copy
.venv/bin/pip install -e .              # numpy / scipy / sklearn / spams
```

Do **not** rely on `source .venv/bin/activate` for batch work: PBS starts a
clean non-interactive shell, so an activated venv does not carry over and
`python3` in the job resolves to the bare module Python (which has no
`iteraa`). `submitTC.sh` therefore calls `${SCRIPT_DIR}/.venv/bin/python`
directly in every PBS script, and checks that this interpreter exists and can
import `numpy, scipy, sklearn, iteraa` before queueing anything. Use
`PYTHON=/path/to/python ./submitTC.sh` to point the jobs at a different
interpreter.

```bash
cd iaa-complexity-scaling

./submitTC.sh          # submit all jobs (TC + SS + WS)
./submitTC.sh iaa      # IAA time-complexity only
./submitTC.sh ss       # strong scaling only
./submitTC.sh ws       # weak scaling only

# After all jobs finish (activate the venv and run from this directory):
source .venv/bin/activate
python plotTC.py
python plotSS_WS.py
```

Or open `complexityScaling.ipynb` for the full interactive analysis.

---

## Manuscript Connections

| Script / Output | Manuscript Location |
|---|---|
| `plotTC.py` -> `TC_PIAA.png`, `TC_PIAA_loglog.png` | Supporting evidence for section "Computational scaling" |
| `runSS.py` + `plotSS_WS.py` -> `SS_IAA_*.png` | Strong-scaling speedup and efficiency figures |
| `runWS.py` + `plotSS_WS.py` -> `WS_IAA_Efficiency.png` | Weak-scaling efficiency figure |
