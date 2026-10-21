#!/usr/bin/env python3
"""Timing benchmark for iterative archetypal analysis (IAA) on the MNP MD dataset.

Motivation
----------
This script measures both
quantities on the monometallic nanoparticle (MNP) feature dataset (n = 9300 conformations, m = 134 structural features).

Measured stages (identical solver calls to ``iteraa.piaa.runAA`` and
``iteraa.iaa.ArchetypalAnalysis._combine_subset_results``):

  1. subset splitting                       (k disjoint subsets, KFold)
  2. round 1  : k x AA(n/k, m, p1)          (the k subsets, run on P workers)
  3. round 2  : AA(k*p1, m, p2)             (on the k*p1 intermediate archetypes)
  4. compose  : coefficient matrices        (matrix products, no QPs)
  5. baseline : AA(n, m, p2)                (standard AA on the full dataset)

Reported quantities, following Eq. (4) of the manuscript:

  * computational cost  C_IAA = sum_i T(AA(n/k, m, p1))_i + T(AA(k*p1, m, p2))
                        i.e. core-seconds with one thread per subset, which is
                        the quantity the equation describes. The subsets are run
                        one at a time for this number: running them concurrently
                        inflates each T(AA(n/k, m, p1)) by up to ~3x at k = 10
                        (memory-bandwidth contention), which Eq. (4) does not
                        model.
  * wall-clock time     T_IAA(P) = measured wall-clock of the subset phase on P
                        workers + T(final round) + T(compose). The ideal value
                        ceil(k/P) * T(AA(n/k, m, p1)) + T(...) is reported too.
  * C_RAA = T(AA(n, m, p2)) and its wall-clock, for the same p2.

Round 1 is therefore executed twice per repeat: once sequentially (for the cost)
and once on P workers (for the wall-clock). Use --no-wallPass to skip the second
pass and report the ideal wall-clock instead.

Why a p2 sweep
--------------
Round 1 costs k * T(AA(n/k, m, p1)) and does not depend on p2, whereas the
standard-AA baseline at the same p2 does. A single p2 therefore only shows one
point of the trade-off; the sweep reuses the round-1 archetypes for every p2 and
runs just the cheap final round each time. The production MNP run (p2 = 250) is
included by default, but note that at small p2 standard AA is cheaper than IAA on
this dataset: the k round-1 calls with p1 = 300 dominate. The crossover p2 - the
point past which IAA is cheaper - is what the manuscript's scaling argument
predicts and what this table measures.

The single-thread-per-subset layout (``--threads 1``) is deliberate: on the
small subset calls (n = 930, p1 = 300) spams gains almost nothing from extra
threads (see the threaded control run), so one core per subset both reproduces
the production layout (k concurrent PBS jobs) and makes the core-seconds an
honest measure.

The full-dataset AA baseline behaves differently: on n = 9299 it scales 2.4-2.6x
on 10 threads (measured: 737 s -> 305 s at p2 = 500, 1877 s -> 732 s at
p2 = 1000). Time it with ``--raaThreads`` set to the same value as
``--workers`` when its wall-clock is compared with IAA's, since the IAA row
spends WORKERS cores and a one-core AA row would not be a like-for-like
comparison.

Usage
-----
On the cluster, submit through the driver, which fills in the PBS resources and
every knob (dry run first to see the qsub command):

    ./submitBenchmarkTimingMNP.sh --dry-run
    ./submitBenchmarkTimingMNP.sh -k 10 -p1 300 -r 3 -W 10 -c 48 -w 12:00:00

When submitting the job script directly, remember that PBS splits `-v` on
commas, so multi-valued knobs are colon-separated on the command line:

    qsub -v P2_SWEEP=100:250:500:1000,REPEAT=3 runBenchmarkTimingMNP.sh

Full run without the driver (production configuration, p2 sweep):

    python benchmarkTimingMNP.py --workers 10 --repeat 3 --p2Sweep 100,250,500,1000

Quick local smoke test (about a minute):

    python benchmarkTimingMNP.py --nSamples 930 --p1 10 --p2Sweep 5,10 --workers 4 \
        --repeat 1 --no-raa

Outputs (under ``--outdir``):
    timingBenchmark_<label>_raw.json     per-repeat, per-stage records
    timingBenchmark_<label>_summary.csv  mean/std/min/max per quantity and p2
    timingBenchmark_<label>_table.tex    summary table ready to paste in the manuscript

Note: the spams solver prints its residual history ("RSS FISTA = ...") to stdout,
as in the original PBS ``.out`` records. Summary lines are prefixed with
``RESULT`` so they can be filtered, e.g. ``... | grep RESULT``.
"""

import os

# Pin the BLAS/OpenMP pools to one thread *before* numpy is imported. The design of
# this benchmark is one core per subset (spams itself only gets numThreads, below),
# and letting every worker spawn a full BLAS pool would oversubscribe the node and
# inflate the round-1 timings. Set these to another value in the environment if you
# deliberately want threaded BLAS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import pickle
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from spams import archetypalAnalysis
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit(
        "spams is not importable. Run this script with the iteraa virtual environment, "
        "e.g. `iteraa/.venv/bin/python benchmarkTimingMNP.py ...`, or install spams-bin."
    )

REPO_ROOT = Path(__file__).resolve().parents[2]

# Feature tables the MNP run was made from, in order of preference. The first is
# the copy kept in this repository; the second is the archive copy on Gadi.
DATA_CANDIDATES = (
    REPO_ROOT / "reduced-np-regression" / "data" / "MNP_E_DF.pkl",
    Path("data/MNPFeatsDF.pkl"),
)
# Non-feature columns of those tables (energies, elemental label).
LABEL_COLUMNS = ("Total_E", "Formation_E", "Ele", "eleComb")


def loadMNPFeatures(path: Path) -> pd.DataFrame:
    """Load the MNP feature table and strip the non-feature columns.

    ``MNP_E_DF.pkl`` holds 9300 rows x 134 features plus Total_E and Formation_E;
    ``MNPFeatsDF.pkl`` holds the same 134 features plus an ``Ele`` label. Both
    therefore reduce to the n = 9300, m = 134 matrix analysed in the manuscript.
    """
    with open(path, "rb") as f:
        df = pickle.load(f)
    missing = df.isna().any(axis=1)
    if missing.any():
        print(f"  dropping {int(missing.sum())} row(s) containing NaN features")
        df = df.loc[~missing]
    drop = [c for c in LABEL_COLUMNS if c in df.columns]
    feats = df.drop(columns=drop).astype(np.float64)
    return feats


def resolveDataPath(userPath: str | None) -> Path:
    if userPath:
        path = Path(userPath)
        if not path.exists():
            sys.exit(f"--data path does not exist: {path}")
        return path
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit(
        "Could not find the MNP feature table. Pass one explicitly with "
        f"--data, e.g. --data {DATA_CANDIDATES[1]}"
    )


def runAA(X: np.ndarray, nArchetypes: int, cfg: dict) -> dict:
    """Run one spams AA call on X of shape (nSamples, nFeatures).

    Mirrors the solver call in ``iteraa.piaa.runAA`` (Z0=None, epsilon=tolerance,
    stepsFISTA/stepsAS as configured) so the timings are comparable with the
    production runs whose ``.out`` records are in this directory.
    """
    start = time.perf_counter()
    Z, A, B = archetypalAnalysis(
        np.asfortranarray(X.T),
        Z0=None,
        p=nArchetypes,
        returnAB=True,
        robust=cfg["robust"],
        epsilon=cfg["tolerance"],
        computeXtX=cfg["computeXtX"],
        stepsFISTA=cfg["stepsFISTA"],
        stepsAS=cfg["stepsAS"],
        randominit=cfg["randominit"],
        numThreads=cfg["threads"],
    )
    elapsed = time.perf_counter() - start
    return {"time_s": elapsed, "Z": Z, "A": A.toarray(), "B": B.toarray()}


def explainedVariance(X: np.ndarray, Z: np.ndarray, A: np.ndarray) -> float:
    """Explained variance of X (nSamples, nFeatures) given archetypes Z and codes A.

    Same convention as ``fitPIAA``: the score is evaluated over samples, i.e.
    against X of shape (nSamples, nFeatures) and its reconstruction (Z @ A).T.
    """
    return float(explained_variance_score(X, (Z @ A).T))


def subsetTask(payload: tuple) -> dict:
    """Round-1 AA on one subset; returns the archetypes and codes."""
    subsetID, Xsub, nArchetypes, cfg = payload
    out = runAA(Xsub, nArchetypes, cfg)
    return {
        "subsetID": subsetID,
        "nSamples": int(Xsub.shape[0]),
        "time_s": out["time_s"],
        "explainedVariance": explainedVariance(Xsub, out["Z"], out["A"]),
        "Z": out["Z"],
        "A": out["A"],
    }


def subsetTimingTask(payload: tuple) -> dict:
    """Round-1 AA on one subset for the wall-clock pass; timings only."""
    subsetID, Xsub, nArchetypes, cfg = payload
    out = runAA(Xsub, nArchetypes, cfg)
    return {"subsetID": subsetID, "nSamples": int(Xsub.shape[0]), "time_s": out["time_s"]}


def summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def benchmarkIAA(X: np.ndarray, args, cfg: dict, p2List: list[int]) -> list[dict]:
    """One repeat of the full IAA pipeline, timed once per p2 in ``p2List``.

    Round 1 is independent of p2, so it runs once and its archetypes are reused
    by every final round. The returned list holds one record per p2.
    """
    nSamples = X.shape[0]
    k, p1 = args.k, args.p1

    # ---- stage 1: split -----------------------------------------------------
    start = time.perf_counter()
    kFold = KFold(n_splits=k, shuffle=True, random_state=args.seed + args.repeatIndex)
    sampleIdxs = [testIdx for (_, testIdx) in kFold.split(X)]
    subsets = [X[idx, :] for idx in sampleIdxs]
    splitTime = time.perf_counter() - start
    print(f"RESULT repeat={args.repeatIndex} split_time={splitTime:.3f} s")

    # ---- stage 2a: round 1, one subset at a time -> computational cost ------
    # Run strictly sequentially so each T(AA(n/k, m, p1)) is measured without
    # contending for memory bandwidth with the other subsets. Running the subsets
    # concurrently inflates the per-subset time by up to ~3x at k = 10 (measured),
    # which would misrepresent the cost term in Eq. (4).
    payloads = [(i, subsets[i], p1, cfg) for i in range(k)]
    subsetOuts = []
    start = time.perf_counter()
    for payload in payloads:
        out = subsetTask(payload)
        subsetOuts.append(out)
        print(
            f"RESULT repeat={args.repeatIndex} subset={out['subsetID']} "
            f"n={out['nSamples']} p1={p1} time={out['time_s']:.3f} s "
            f"EV={out['explainedVariance']:.4f}"
        )
    subsetSerial = time.perf_counter() - start
    subsetTimes = [out["time_s"] for out in subsetOuts]
    print(
        f"RESULT repeat={args.repeatIndex} round1_serial cost={subsetSerial:.3f} core-s "
        f"per_subset_mean={np.mean(subsetTimes):.3f} s"
    )

    # ---- stage 2b: round 1 on P workers -> wall-clock time ------------------
    # Results are discarded (the serial pass above already produced them); this
    # pass measures how long the first round actually takes on P cores.
    workers = max(1, min(args.workers, k))
    subsetWallPredicted = float(np.ceil(k / workers) * np.mean(subsetTimes))
    if args.wallPass:
        start = time.perf_counter()
        # With more than one spams thread per call, the parent has already started
        # an OpenMP pool (the warmup call), and forked children inherit that state
        # and deadlock on it. "spawn" gives each child a clean interpreter; it
        # costs a second or two of start-up per worker, which is why fork is kept
        # for the default single-threaded layout.
        ctx = mp.get_context("spawn" if cfg["threads"] > 1 else "fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            wallOuts = list(pool.map(subsetTimingTask, payloads))
        subsetWall = time.perf_counter() - start
        wallTimes = sorted(out["time_s"] for out in wallOuts)
        print(
            f"RESULT repeat={args.repeatIndex} round1_parallel wall={subsetWall:.3f} s "
            f"slowest_subset={wallTimes[-1]:.3f} s workers={workers} "
            f"predicted={subsetWallPredicted:.3f} s"
        )
    else:
        subsetWall = subsetWallPredicted
        print(
            f"RESULT repeat={args.repeatIndex} round1_parallel wall={subsetWall:.3f} s "
            "(predicted, --no-wallPass)"
        )

    # ---- stages 3-4 over the p2 sweep --------------------------------------
    # Intermediate samples are fixed by round 1, so each p2 only pays the final
    # round plus the coefficient composition.
    allSubsetZs = np.concatenate([out["Z"] for out in subsetOuts], axis=1)  # (m, k*p1)
    nIntermediate = allSubsetZs.shape[1]

    records = []
    for p2 in p2List:
        finalOut = runAA(allSubsetZs.T, p2, cfg)
        print(
            f"RESULT repeat={args.repeatIndex} final_round p2={p2} n={nIntermediate} "
            f"time={finalOut['time_s']:.3f} s "
            f"EV={explainedVariance(allSubsetZs.T, finalOut['Z'], finalOut['A']):.4f}"
        )

        start = time.perf_counter()
        prev = 0
        blocks = []
        for out in subsetOuts:
            nSubsetArch = out["Z"].shape[1]
            blocks.append(finalOut["A"][:, prev : prev + nSubsetArch] @ out["A"])
            prev += nSubsetArch
        alfaOverall = np.empty((p2, nSamples), dtype=np.float64)
        for block, idx in zip(blocks, sampleIdxs):
            alfaOverall[:, idx] = block
        composeTime = time.perf_counter() - start
        # Scored after timing: the score itself is not part of the pipeline.
        endToEndEV = explainedVariance(X, finalOut["Z"], alfaOverall)
        print(
            f"RESULT repeat={args.repeatIndex} compose p2={p2} time={composeTime:.3f} s "
            f"end_to_end_EV={endToEndEV:.4f}"
        )

        computeCost = subsetSerial + finalOut["time_s"] + composeTime
        wallClock = subsetWall + finalOut["time_s"] + composeTime
        idealWall = subsetWallPredicted + finalOut["time_s"] + composeTime
        parallelEfficiency = (float(np.mean(subsetTimes)) * k / workers) / subsetWall
        print(
            f"RESULT repeat={args.repeatIndex} IAA p2={p2} "
            f"compute_cost={computeCost:.3f} core-s wall_clock={wallClock:.3f} s "
            f"ideal_wall={idealWall:.3f} s parallel_efficiency={parallelEfficiency:.3f}"
        )
        # "Run i: x s" keeps the format parsed by getAllDurations() in
        # iaa-complexity-scaling/complexityScaling.ipynb. Only the headline p2 is
        # emitted in that format so the notebook is not fed several times per run.
        if p2 == args.p2:
            print(f"Run {args.repeatIndex}: {wallClock:.3f} s")

        records.append(
            {
                "repeat": args.repeatIndex,
                "seed": args.seed + args.repeatIndex,
                "p2": p2,
                "splitTime_s": splitTime,
                "subsets": [
                    {key: val for key, val in out.items() if key not in ("Z", "A")}
                    for out in subsetOuts
                ],
                "subsetPhaseWall_s": subsetWall,
                "subsetPhaseWallPredicted_s": subsetWallPredicted,
                "subsetPhaseSerial_s": subsetSerial,
                "wallPassMeasured": bool(args.wallPass),
                "finalRound": {
                    "nSamples": int(nIntermediate),
                    "nArchetypes": p2,
                    "time_s": finalOut["time_s"],
                    "explainedVariance": explainedVariance(
                        allSubsetZs.T, finalOut["Z"], finalOut["A"]
                    ),
                    "endToEndExplainedVariance": endToEndEV,
                },
                "composeTime_s": composeTime,
                "iaaComputeCost_coreS": computeCost,
                "iaaWallClock_s": wallClock,
                "iaaIdealWallClock_s": idealWall,
                "subsetParallelEfficiency": parallelEfficiency,
                "workers": workers,
            }
        )
    return records


def benchmarkRAA(X: np.ndarray, args, cfg: dict, p2List: list[int]) -> list[dict]:
    """Standard (non-iterative) AA on the full dataset, once per p2."""
    records = []
    for p2 in p2List:
        for repeat in range(args.raaRepeats):
            out = runAA(X, p2, cfg)
            ev = explainedVariance(X, out["Z"], out["A"])
            print(
                f"RESULT raa n={X.shape[0]} p2={p2} threads={cfg['threads']} "
                f"repeat={repeat} time={out['time_s']:.3f} s EV={ev:.4f}"
            )
            records.append(
                {
                    "nSamples": int(X.shape[0]),
                    "nArchetypes": p2,
                    "threads": cfg["threads"],
                    "repeat": repeat,
                    "time_s": out["time_s"],
                    "explainedVariance": ev,
                }
            )
    return records


def perP2Summary(records: list[dict], raaRecords: list[dict], args) -> dict:
    """Group the per-repeat records by p2 and add the AA/IAA comparisons."""
    byP2: dict[int, dict] = {}
    for p2 in args.p2Sweep:
        iaaRecs = [rec for rec in records if rec["p2"] == p2]
        if not iaaRecs:
            continue
        iaaCompute = [rec["iaaComputeCost_coreS"] for rec in iaaRecs]
        iaaWall = [rec["iaaWallClock_s"] for rec in iaaRecs]
        block = {
            "round2_s": summarise([rec["finalRound"]["time_s"] for rec in iaaRecs]),
            "compose_s": summarise([rec["composeTime_s"] for rec in iaaRecs]),
            "iaa_computational_cost_core_s": summarise(iaaCompute),
            "iaa_wall_clock_s": summarise(iaaWall),
            "iaa_end_to_end_explained_variance": summarise(
                [rec["finalRound"]["endToEndExplainedVariance"] for rec in iaaRecs]
            ),
        }
        raaRecs = [rec for rec in raaRecords if rec["nArchetypes"] == p2]
        if raaRecs:
            raaWall = [rec["time_s"] for rec in raaRecs]
            raaCost = [rec["time_s"] * rec["threads"] for rec in raaRecs]
            block["raa_wall_clock_s"] = summarise(raaWall)
            block["raa_computational_cost_core_s"] = summarise(raaCost)
            block["raa_explained_variance"] = summarise(
                [rec["explainedVariance"] for rec in raaRecs]
            )
            # Ratio of AA to IAA for the same p2: >1 means IAA is cheaper.
            # Pair on repeat index; falls back to the mean when the counts differ.
            if len(raaRecs) >= len(iaaRecs):
                pairs = list(zip(raaCost[: len(iaaCompute)], iaaCompute))
                wallPairs = list(zip(raaWall[: len(iaaWall)], iaaWall))
            else:
                pairs = [(float(np.mean(raaCost)), float(np.mean(iaaCompute)))]
                wallPairs = [(float(np.mean(raaWall)), float(np.mean(iaaWall)))]
            block["speedup_computational_cost"] = summarise([a / b for (a, b) in pairs])
            block["speedup_wall_clock"] = summarise([a / b for (a, b) in wallPairs])
        byP2[str(p2)] = block
    return byP2


def writeOutputs(records: list[dict], raaRecords: list[dict], args, cfg: dict) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / f"timingBenchmark_{args.label}"

    # Round-1 quantities are shared by every p2, so summarise them once. With
    # several p2 values each repeat contributes several identical rows; keep one
    # record per repeat to avoid weighting them by the sweep length.
    uniqueRepeats = {}
    for rec in records:
        uniqueRepeats.setdefault(rec["repeat"], rec)
    roundOne = list(uniqueRepeats.values())

    summary = {
        "split_s": summarise([rec["splitTime_s"] for rec in roundOne]),
        "round1_total_core_s": summarise([rec["subsetPhaseSerial_s"] for rec in roundOne]),
        "round1_wall_s": summarise([rec["subsetPhaseWall_s"] for rec in roundOne]),
        "round1_wall_ideal_s": summarise(
            [rec["subsetPhaseWallPredicted_s"] for rec in roundOne]
        ),
        "byP2": perP2Summary(records, raaRecords, args),
    }

    raw = {
        "dataset": {
            "label": args.label,
            "path": str(args.dataPath),
            "nSamples": int(args.nSamplesUsed),
            "nFeatures": int(args.nFeatures),
        },
        "configuration": {
            "k": args.k,
            "p1": args.p1,
            "p2Sweep": list(args.p2Sweep),
            "p2Headline": args.p2,
            "workers": args.workers,
            "threads": args.threads,
            "repeat": args.repeat,
            "raaThreads": args.raaThreads,
            "raaRepeats": args.raaRepeats,
            "seed": args.seed,
            "solver": {
                "robust": cfg["robust"],
                "tolerance": cfg["tolerance"],
                "computeXtX": cfg["computeXtX"],
                "stepsFISTA": cfg["stepsFISTA"],
                "stepsAS": cfg["stepsAS"],
                "randominit": cfg["randominit"],
            },
        },
        "environment": environmentInfo(),
        "iaa": records,
        "raa": raaRecords,
        "summary": summary,
    }

    with open(f"{prefix}_raw.json", "w") as f:
        json.dump(raw, f, indent=2, default=float)

    rows = []
    for quantity, stats in summary.items():
        if quantity == "byP2":
            for p2, block in stats.items():
                for subQuantity, subStats in block.items():
                    rows.append({"quantity": subQuantity, "p2": p2, **subStats})
        else:
            rows.append({"quantity": quantity, "p2": "", **stats})
    pd.DataFrame(rows).to_csv(f"{prefix}_summary.csv", index=False)

    with open(f"{prefix}_table.tex", "w") as f:
        f.write(latexTable(summary, args))

    print(f"\nWrote {prefix}_raw.json, _summary.csv and _table.tex")
    return summary


def latexTable(summary: dict, args) -> str:
    """Manuscript-ready timing table (booktabs-free, Wiley class friendly)."""
    round1Cost = summary["round1_total_core_s"]
    round1Wall = summary["round1_wall_s"]
    lines = [
        "% Generated by iaa-np-reduction/runIAA_MNP/benchmarkTimingMNP.py",
        f"% n = {args.nSamplesUsed}, m = {args.nFeatures}, k = {args.k}, "
        f"p1 = {args.p1}, p2 = {list(args.p2Sweep)}, P = {args.workers}",
        "\\begin{table}[t] \\centering",
        "\\small",
        "\\caption{Measured timing on the monometallic nanoparticle dataset "
        f"({args.nSamplesUsed} conformations, {args.nFeatures} features) as a "
        f"function of the number of final archetypes $p_2$, with $k$ = {args.k} "
        f"subsets and $p_1$ = {args.p1} intermediate archetypes. Core-seconds are "
        "with one thread per subset; wall-clock times for IAA use "
        f"{args.workers} workers for the first round, measured in a separate pass "
        f"over the same subsets. Values are the mean $\\pm$ standard deviation over "
        f"{args.repeat} IAA run(s) and {args.raaRepeats} AA run(s). Round 1 "
        f"contributes {round1Cost['mean']:.0f} core-s "
        f"({round1Wall['mean']:.0f} s wall) to every IAA row and does not depend on "
        "$p_2$, whereas the cost of standard AA grows with $p_2$; the sweep "
        "therefore locates the $p_2$ at which the two cross. Speedups are AA/IAA "
        "for the same $p_2$, so values above one favour IAA. EV is the end-to-end "
        "explained variance of the reconstruction.}",
        "\\label{tab:timingMNP}",
        "\\begin{tabular}{rrrrrrr}",
        "\\hline",
        "$p_2$ & Method & Core-seconds & Wall-clock (s) & EV & Speedup (cost) & "
        "Speedup (wall) \\\\",
        "\\hline",
    ]
    for p2, block in summary["byP2"].items():
        ev = block["iaa_end_to_end_explained_variance"]
        cost = block["iaa_computational_cost_core_s"]
        wall = block["iaa_wall_clock_s"]
        if "raa_computational_cost_core_s" in block:
            raaCost = block["raa_computational_cost_core_s"]
            raaWall = block["raa_wall_clock_s"]
            raaEV = block["raa_explained_variance"]
            lines.append(
                f"{p2} & AA & {raaCost['mean']:.1f} $\\pm$ {raaCost['std']:.1f} & "
                f"{raaWall['mean']:.1f} $\\pm$ {raaWall['std']:.1f} & "
                f"{raaEV['mean']:.3f} & 1.00$\\times$ & 1.00$\\times$ \\\\"
            )
            speedupCost = block["speedup_computational_cost"]
            speedupWall = block["speedup_wall_clock"]
            speedupText = (
                f"{speedupCost['mean']:.2f}$\\times$ & {speedupWall['mean']:.2f}$\\times$"
            )
        else:
            speedupText = "-- & --"
        lines.append(
            f"{p2} & IAA & {cost['mean']:.1f} $\\pm$ {cost['std']:.1f} & "
            f"{wall['mean']:.1f} $\\pm$ {wall['std']:.1f} & {ev['mean']:.3f} & "
            f"{speedupText} \\\\"
        )
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def environmentInfo() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpuCount": len(os_cpuAffinity()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        info["scikit-learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import spams

        info["spams"] = getattr(spams, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        pass
    try:
        info["gitCommit"] = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        info["gitCommit"] = "unavailable"
    return info


def os_cpuAffinity() -> list:
    try:
        return list(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return list(range(os.cpu_count() or 1))


def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", help="MNP feature pickle (defaults to the repo copy)")
    parser.add_argument("--outdir", default="timingBenchmarkMNP_results")
    parser.add_argument("--label", default="MNP", help="prefix for output file names")

    # IAA configuration (production MNP run: k=10, p1=300, p2=250)
    parser.add_argument("--k", type=int, default=10, help="number of subsets")
    parser.add_argument("--p1", type=int, default=300, help="archetypes per subset")
    parser.add_argument(
        "--p2",
        type=int,
        default=None,
        help="headline p2, used for the 'Run i: x s' line (default: 250 if in "
        "--p2Sweep, else the largest sweep value)",
    )
    parser.add_argument(
        "--p2Sweep",
        default="100,250,500,1000",
        help="comma-separated p2 values; round 1 is reused for all of them",
    )
    parser.add_argument("--workers", type=int, default=10, help="cores for round 1")
    parser.add_argument("--threads", type=int, default=1, help="spams threads per AA call")
    parser.add_argument("--repeat", type=int, default=3, help="IAA repeats")
    parser.add_argument("--wallPass", dest="wallPass", action="store_true", default=True,
                        help="also time round 1 on P workers (real wall-clock)")
    parser.add_argument("--no-wallPass", dest="wallPass", action="store_false",
                        help="skip the parallel pass and predict wall-clock instead")
    parser.add_argument("--seed", type=int, default=42)

    # RAA baseline
    parser.add_argument("--raa", dest="raa", action="store_true", default=True)
    parser.add_argument("--no-raa", dest="raa", action="store_false",
                        help="skip standard AA on the full dataset")
    parser.add_argument("--raaThreads", type=int, default=1)
    parser.add_argument("--raaRepeats", type=int, default=1)
    parser.add_argument(
        "--raaP2",
        default="",
        help="comma-separated p2 values to time standard AA at (default: all of --p2Sweep)",
    )

    # Solver settings (must match the production runs in runIAA_MNP.py)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--stepsFISTA", type=int, default=3)
    parser.add_argument("--stepsAS", type=int, default=50)
    parser.add_argument("--robust", action="store_true", default=False)
    parser.add_argument("--computeXtX", action="store_true", default=False)
    parser.add_argument("--randominit", action="store_true", default=False)

    # Smoke-testing
    parser.add_argument("--nSamples", type=int, default=0,
                        help="use only the first N samples (0 = full dataset)")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")

    args = parser.parse_args(argv)
    args.p2Sweep = sorted({int(x) for x in args.p2Sweep.split(",") if x.strip()})
    args.raaP2 = sorted({int(x) for x in args.raaP2.split(",") if x.strip()}) or args.p2Sweep
    if args.p2 is None:
        # Prefer the production configuration (p2 = 250) for the headline line.
        args.p2 = 250 if 250 in args.p2Sweep else max(args.p2Sweep)
    elif args.p2 not in args.p2Sweep:
        args.p2Sweep = sorted([*args.p2Sweep, args.p2])
    return args


def main(argv: list[str] | None = None) -> int:
    # Line-buffer stdout: progress is otherwise invisible when the job's output is
    # redirected to a file or piped through tee, and would be lost if the job is
    # killed at the walltime limit.
    sys.stdout.reconfigure(line_buffering=True)
    args = parseArgs(argv)
    args.dataPath = resolveDataPath(args.data)

    if args.p2 >= args.k * args.p1:
        sys.exit(
            f"p2 = {args.p2} must be smaller than k*p1 = {args.k * args.p1}; "
            "otherwise the final round has fewer intermediate samples than archetypes."
        )
    tooLarge = [p for p in args.p2Sweep if p >= args.k * args.p1]
    if tooLarge:
        sys.exit(
            f"--p2Sweep values {tooLarge} are not smaller than k*p1 = {args.k * args.p1}. "
            "spams cannot return more archetypes than samples; lower p2, or raise "
            "p1/k (this is the p2 > k*p1 regime discussed in comment R1.2)."
        )

    print(f"Loading MNP features from {args.dataPath}")
    feats = loadMNPFeatures(args.dataPath)
    if args.nSamples:
        feats = feats.iloc[: args.nSamples, :]
    X = StandardScaler().fit_transform(feats.to_numpy())
    args.nSamplesUsed, args.nFeatures = X.shape
    print(
        f"Dataset: n = {args.nSamplesUsed}, m = {args.nFeatures} (standardised)\n"
        f"IAA: k = {args.k}, p1 = {args.p1}, p2 = {args.p2Sweep}, "
        f"workers = {args.workers}, threads = {args.threads}, repeats = {args.repeat}"
    )

    available = len(os_cpuAffinity())
    if args.workers > available:
        print(
            f"Warning: --workers {args.workers} exceeds the {available} CPUs visible to "
            "this process; round-1 timings will include queueing."
        )
    if args.raaP2 != args.p2Sweep:
        print(f"Standard AA will be timed at p2 = {args.raaP2} only")

    cfg = {
        "robust": args.robust,
        "tolerance": args.tolerance,
        "computeXtX": args.computeXtX,
        "stepsFISTA": args.stepsFISTA,
        "stepsAS": args.stepsAS,
        "randominit": args.randominit,
        "threads": args.threads,
    }

    if args.warmup:
        # Absorb first-call overhead (allocation, spams initialisation) so it is
        # not attributed to the first timed stage.
        runAA(X[: min(50, args.nSamplesUsed), :], min(2, 49), cfg)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(args.repeat):
        args.repeatIndex = repeat
        print(f"\n=== IAA repeat {repeat + 1}/{args.repeat} ===")
        records.extend(benchmarkIAA(X, args, cfg, args.p2Sweep))
        # Checkpoint after each repeat: a long job can be killed at the walltime
        # limit, and the stages already measured are still useful.
        partial = Path(args.outdir) / f"timingBenchmark_{args.label}_partial.json"
        with open(partial, "w") as f:
            json.dump({"configuration": vars(args), "iaa": records}, f,
                      indent=2, default=str)
        print(f"RESULT checkpoint repeat={repeat + 1} -> {partial}")

    raaRecords = []
    if args.raa:
        print(
            f"\n=== Standard AA baseline (p2 = {args.raaP2}, "
            f"threads = {args.raaThreads}) ==="
        )
        raaCfg = dict(cfg, threads=args.raaThreads)
        raaRecords = benchmarkRAA(X, args, raaCfg, args.raaP2)

    summary = writeOutputs(records, raaRecords, args, cfg)

    print("\n=== Summary (mean +- std) ===")
    for quantity, stats in summary.items():
        if quantity == "byP2":
            continue
        print(f"RESULT summary {quantity} = {stats['mean']:.3f} +- {stats['std']:.3f}")
    for p2, block in summary["byP2"].items():
        for quantity, stats in block.items():
            print(
                f"RESULT summary p2={p2} {quantity} = "
                f"{stats['mean']:.3f} +- {stats['std']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
