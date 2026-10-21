# MNP Timing Benchmark on NCI Gadi — Runbook

Instructions for running the IAA-vs-AA timing benchmark on the monometallic nanoparticle
(MNP) dataset, for someone who has not seen this project before.

## What this measures

The manuscript's complexity-scaling claim rests on Eq. (4): an iterative archetypal
analysis (IAA) pipeline should cost less than standard archetypal analysis (AA) on the
same data once the number of final archetypes `p2` is large enough. This benchmark
measures both sides on the real MNP dataset and reports them in the two units the
manuscript needs:

* **computational cost** — core-seconds (one core per subset), the quantity in Eq. (4);
* **wall-clock time** — what you actually wait for on `P` parallel workers.

The pipeline that is timed (all stages use the same solver call as the production runs
in `runIAA_MNP.py`):

| Stage | Work |
|---|---|
| 1. split | `k` disjoint subsets of the data (KFold) |
| 2. round 1 | `k` independent AA calls, one per subset, each with `p1` archetypes |
| 3. round 2 | one AA call on the `k * p1` intermediate archetypes, with `p2` archetypes |
| 4. compose | matrix products that fold the codes back together (no optimisation) |
| 5. baseline | standard AA on the **full** dataset with `p2` archetypes |

Dataset: `reduced-np-regression/data/MNP_E_DF.pkl` → **n = 9299** conformations,
**m = 134** structural features (after dropping one row with NaN features and the two
energy columns).

Production configuration: `k = 10`, `p1 = 300`, `p2 = 250`.

## Files

| File | Role |
|---|---|
| `benchmarkTimingMNP.py` | the measurement script (does the work, writes the results) |
| `runBenchmarkTimingMNP.sh` | the PBS job script (the thing `qsub` runs) |
| `submitBenchmarkTimingMNP.sh` | the driver you run by hand: sets every resource and knob, validates, submits |
| `timingBenchmarkMNP_results/` | outputs (created on first run) |
| `timingBenchmarkMNP_results/aaCalibrationMNP.py` / `.log` | standalone measurements of the standard-AA baseline, used to size the run |

## Step 0 — what you need

* An NCI Gadi account with access to the project you will charge (the scripts default
  to project `p00`, queue `normal`).
* The code and the dataset (Step 1).
* A Python ≥ 3.11 with `spams` (Step 2) — Gadi's default `python3/3.10.4` module is
  **too old** for this project.

## Step 1 — copy the code to Gadi

The working tree contains two git submodules (`iteraa`, `reduced-np-regression`); the
dataset pickle is tracked inside `reduced-np-regression`, so a copy of the tree carries
the data with it.

From your laptop:

```bash
rsync -av --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  /path/to/iterative-archetypal-analysis/ \
  <user>@gadi.nci.org.au:/scratch/p00/<user>/iterative-archetypal-analysis/
```

`--exclude '.venv'` matters: the virtual environment on your laptop is built for its own
Python and will not run on Gadi. Step 2 rebuilds it there.

<details>
<summary>Alternative: clone the two public submodule repositories instead of rsyncing</summary>

```bash
git clone https://github.com/Jon-Ting/iteraa.git
git clone https://github.com/Jon-Ting/metal-nanoparticle-energy-regression.git reduced-np-regression
```

The benchmark scripts themselves live in the top-level working tree, which has no remote,
so copy `iaa-np-reduction/runIAA_MNP/` across by hand in this case.
</details>

## Step 2 — build the Python environment on Gadi

```bash
ssh gadi.nci.org.au
module avail python3          # pick a >= 3.11 build, e.g. python3/3.11.x
module load python3/3.11.x    # if the newest module is older (3.10.x), skip this and
                              # let uv supply its own 3.11 below

cd /scratch/p00/<user>/iterative-archetypal-analysis/iteraa
curl -LsSf https://astral.sh/uv/install.sh | sh     # if `uv` is not already there
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.11        # standalone CPython 3.11, if the module has none
uv sync --no-dev              # creates .venv from uv.lock (numpy, pandas, scikit-learn, spams-bin, ...)

.venv/bin/python -c "import numpy, pandas, sklearn, spams; print('environment ok')"
```

`spams-bin` ships a prebuilt `manylinux` wheel for CPython 3.11, so no compiler is
needed on the cluster. The benchmark script does not import the `iteraa` package itself,
but syncing `iteraa`'s lockfile is the easiest way to get the exact dependency versions
the production runs used.

If you would rather not use `uv`:

```bash
python3 -m venv .venv
.venv/bin/pip install "numpy>=1.26.4" "pandas>=2.2.2" "scikit-learn>=1.5.1" \
                        "scipy>=1.14.0" "spams-bin>=2.6.10"
```

## Step 3 — smoke test (about a minute, on the login node)

```bash
cd /scratch/p00/<user>/iterative-archetypal-analysis/iaa-np-reduction/runIAA_MNP
../../iteraa/.venv/bin/python -u benchmarkTimingMNP.py \
    --nSamples 930 --p1 10 --p2Sweep 5,10 --workers 4 --repeat 1 --no-raa \
    --outdir /tmp/smoke_mnp
```

You are looking for `Dataset: n = 930, m = 134`, a series of `RESULT summary ...` lines,
and four files in `/tmp/smoke_mnp`. If that works, the cluster run will work.

## Step 4 — dry run, then submit

```bash
cd /scratch/p00/<user>/iterative-archetypal-analysis/iaa-np-reduction/runIAA_MNP

./submitBenchmarkTimingMNP.sh --dry-run \
    -c 12 -w 04:00:00 -m 16GB \
    --raa-threads 10 --raa-repeats 3 \
    --sweep 100:250:500:1000 \
    -p /scratch/p00/<user>/iterative-archetypal-analysis/iteraa/.venv/bin/python
```

The dry run prints the exact `qsub` command and submits nothing. Check it, then drop
`--dry-run` to submit for real.

Recommended production command:

```bash
./submitBenchmarkTimingMNP.sh \
    -c 12 -w 04:00:00 -m 16GB \
    --raa-threads 10 --raa-repeats 3 \
    --sweep 100:250:500:1000 \
    -p /scratch/p00/<user>/iterative-archetypal-analysis/iteraa/.venv/bin/python \
    -M <your.email@anu.edu.au>
```

`-p` is **not optional in practice**: without it the job runs the cluster's default
`python3`, which is older than 3.11 and has no `spams`, so the job dies in its
fail-fast import check within seconds. The driver checks the interpreter before
submitting, so a wrong path is reported at submission time rather than in the queue.

The driver refuses to submit a configuration that cannot work (non-numeric arguments,
`--workers` larger than `--ncpus`, a sweep value of 3000 or more when `k * p1` is only
3000, a missing interpreter, an unreadable data file) and prints the reason instead.

### Driver options

| Option | Default | Meaning |
|---|---|---|
| `-n`, `--dry-run` | off | print the `qsub` command, submit nothing |
| `-q`, `--queue` | `normal` | PBS queue |
| `-P`, `--project` | `p00` | PBS project |
| `-N`, `--name` | `iaaTimingMNP` | job name |
| `-c`, `--ncpus` | 48 | cores to request (10 are actually used; 12 is enough) |
| `-w`, `--walltime` | `12:00:00` | walltime to request |
| `-m`, `--mem` | `32GB` | memory to request (16GB is enough) |
| `--jobfs` | `1GB` | job filesystem |
| `-M`, `--email` / `--no-email` | set | notification address |
| `-k`, `--subsets` | 10 | number of subsets `k` |
| `--p1` | 300 | intermediate archetypes per subset |
| `--p2` | 250 | headline `p2` (the one printed as `Run i: x s`) |
| `--sweep` | `100:250:500:1000` | `p2` values to time (colon separated — PBS splits `-v` on commas) |
| `-r`, `--repeat` | 3 | IAA repeats |
| `-W`, `--workers` | 10 | cores used for round 1 |
| `-T`, `--threads` | 1 | spams threads per IAA AA call |
| `--raa-threads` | 1 | spams threads for the AA baseline |
| `--raa-repeats` | 1 | AA baseline repeats |
| `--raa-p2` | same as `--sweep` | `p2` values for the AA baseline |
| `--no-raa` | off | skip the AA baseline entirely |
| `-l`, `--label` | `MNP` | output file prefix |
| `-d`, `--data` | repo copy | feature pickle |
| `-o`, `--outdir` | `timingBenchmarkMNP_results/` | output directory |
| `-p`, `--python` | `python3` | interpreter with the dependencies |
| `--module` | none | environment module to load in the job |

### If you use a different project

`-P` sets the project, but `runBenchmarkTimingMNP.sh` still asks for
`#PBS -l storage=scratch/p00`. If you are not in `p00`, edit that line in the job
script as well, otherwise the job requests scratch it cannot use.

## Step 5 — monitor

```bash
qstat -u $USER                                   # queue status
tail -f timingBenchmarkMNP_results/MNP_benchmark.out
```

The live log is the file above (the job `tee`s into it). Progress lines look like:

```
RESULT repeat=0 round1_serial cost=269.309 core-s per_subset_mean=26.927 s
RESULT repeat=0 IAA p2=250 compute_cost=354.963 core-s wall_clock=150.539 s ...
RESULT raa n=9299 p2=250 threads=10 repeat=0 time=... s EV=...
```

PBS Pro also writes its own `<jobname>.o<jobid>` file in the directory you submitted
from; the benchmark's own log is the more useful one.

If the job is killed at the walltime limit, the repeats that finished are still in
`timingBenchmarkMNP_results/timingBenchmark_MNP_partial.json` (checkpointed after every
repeat).

## Step 6 — collect the results

| File | Content |
|---|---|
| `timingBenchmarkMNP_results/timingBenchmark_MNP_table.tex` | the manuscript table, ready to paste |
| `timingBenchmarkMNP_results/timingBenchmark_MNP_summary.csv` | mean / std / min / max per quantity and `p2` |
| `timingBenchmarkMNP_results/timingBenchmark_MNP_raw.json` | per-repeat records plus hostname, CPU count, package versions and git commit |
| `timingBenchmarkMNP_results/submission_<jobid>.txt` | the exact `qsub` command that was used |

Reading the table: for each `p2` there is an **AA** row (standard archetypal analysis on
the full dataset) and an **IAA** row. `Core-seconds` is the Eq. (4) cost; `Wall-clock` is
the measured elapsed time; `Speedup (cost)` and `Speedup (wall)` are AA/IAA ratios, so
values above 1 mean IAA is cheaper.

`EV` is the end-to-end explained variance of the composed reconstruction. The raw JSON
records two explained variances per round: the final round's own fit (`finalRound.
explainedVariance`, 0.965 at `p2 = 250` when measured locally) and the end-to-end value
of the composed reconstruction (0.948). The `0.961` printed by the production job
(`runIAA_MNP_250As.out`) comes from the `iteraa` package's own composition convention and
is close to, but not identical with, either of these — expect agreement to about one
percent, not to the digit.

## How long it takes

Measured on a 14-core workstation (n = 9299, m = 134, k = 10, p1 = 300):

| Stage | Time |
|---|---|
| round 1, serial (the cost measurement) | 270 core-s (27 s per subset) |
| round 1, 10 workers (the wall-clock pass) | 65 s |
| final rounds at p2 = 100 / 250 / 500 / 1000 | 33 / 86 / 189 / 458 s |
| AA baseline, one thread, same four `p2` | 132 / 316 / 737 / 1877 s (51 min in total) |
| AA baseline, 10 threads, same four `p2` | ~20 min in total (2.4–2.6× faster) |

So one IAA repeat is ~18 min, three repeats ~55 min, and the AA baseline adds ~20 min
(threaded) to ~51 min (single-threaded). **Budget about 1.3 h with `--raa-threads 10`
and about 2 h with `--raa-repeats 3`.** The `04:00:00` walltime above is comfortable.

**Do not reduce the dataset to save time.** Halving `n` cuts the AA baseline and round 1
by ~0.38×, but the final round runs on `k * p1 = 3000` intermediate archetypes and does
not shrink at all, so you would save only 20–40 minutes of a ~1.3–2 h run while losing
comparability with the published production timings (138.76 s at `p2 = 250` and 3023.5 s
at `p2 = 5000`, both measured on the full n = 9299 dataset) and changing the `p2`/n
regime that decides where IAA overtakes AA.

**Request 12 cores, not 48.** Only 10 cores are ever used at once. Gadi bills service
units as roughly `ncpus * wall-hours / 30`, so 48 cores for 1.8 h is ~2.9 SU while 12
cores for the same work is ~0.7 SU.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ERROR: ... cannot import numpy/pandas/scikit-learn/spams` | the job's interpreter lacks the dependencies | pass `-p /scratch/p00/<user>/.../iteraa/.venv/bin/python` (the driver checks this before submitting) |
| `ERROR: --sweep value 3000 must satisfy 0 < p2 < k*p1 = 3000; see comment R1.2` | more archetypes than the intermediate samples | lower `--p2`/`--sweep`, or raise `--p1`/`-k` (the `p2 > k*p1` regime is a separate manuscript comment) |
| `ERROR: --workers (50) exceeds --ncpus (12)` | more workers than cores | raise `-c` or lower `-W` |
| `qsub not found` | you are on a compute node, not a login node | run it from the login node |
| `Warning: WORKERS=... exceeds the ... CPUs requested` | cores will be queued for | raise `-c` |
| job killed with no output | exceeded the walltime | raise `-w`; partial results are in `timingBenchmark_MNP_partial.json` |

## Other scripts in this repository

`runIAA_MNP.py`, `iaa-np-reduction/runIAA_BNP/runBNP.sh` and `iaa-scripts/runAAs.sh` are
the **original production scripts** that produced the `.out` records in this repository.
They hard-code the original author's scratch paths (`/scratch/p00/jt5911/...`), load the
pre-`iteraa` package name `iaa`, and expect the per-subset AA outputs to be submitted as
one PBS job per subset. They are not needed for the timing benchmark above, and they need
their paths and interpreter updated before they will run on a different account.
