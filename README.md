# PACT: Contract-Aware xApp Authority and Control Value Arbitration in Multi-Tenant Open RAN

**Authors:**

Md. Kamrul Hossain, Walid Aljoby, Ahmed M. Abdelmoniem, Daniel B. da Costa

Md. Kamrul Hossain is with Information and Computer Science Department, King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia.

Walid Aljoby is with Information and Computer Science Department, and IRC for Intelligent Secure Systems, King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia.

Ahmed M. Abdelmoniem is with School of Electronic Engineering and Computer Science, Queen Mary University of London, UK.

Daniel B. da Costa is with IRC for Communication Systems and Sensing, Department of Electrical Engineering, KFUPM, Saudi Arabia.

Emails: g202215400@kfupm.edu.sa, waleed.gobi@kfupm.edu.sa, ahmed.sayed@qmul.ac.uk, danielbcosta@ieee.org

**This work has been submitted for review in IEEE WCNC.**

#

This repository contains the code and the evaluation results of **PACT**
(**P**aired-twin **A**uthority and **C**ontract-aware arbi**T**ration). PACT is a
two-loop arbiter for the near-real-time RAN Intelligent Controller (near-RT RIC):

- **Outer loop (every 1 s).** It grants *authority leases* (temporary, exclusive
  write permissions) to a set of xApp claims that satisfies the single-writer (C1)
  and resource-envelope (C2) constraints. It maximizes the contract-weighted
  probability that each intent is met.
- **Its effect model.** The drift, authority and pair-interaction effects
  (α, β, γ) are identified offline from paired simulator twins and learned with
  extremely randomized trees as functions of the network state and the pending
  xApp requests.
- **Inner loop (every 0.5 s).** It synthesizes the value of each authorized write
  by minimizing the weighted predicted shortfall of intent margins below their
  targets, then applies a safety mediator.
- **Weights.** Intent priorities, tenant priorities, MILD failure-risk urgency and a
  long-run tenant-floor deficit (C3) set the contract weights.

The code reproduces every number, table and figure of the evaluation section of the
paper.

---------------------------------------------------------------------------

## 1. Method names in the code

The code uses internal method keys. The proposed method is **`W3R-V`**.

| Code key | Name in the paper |
|---|---|
| `W3R-V` | **PACT (proposed)** |
| `QACM-style` | QACM-style value synthesis |
| `QACM-contract` | Value synthesis + PACT weights (equals PACT without outer selection) |
| `B3-contract` | Greedy value arbitration + PACT weights |
| `B3-contract-raw` | Greedy + PACT weights, without the safety mediator |
| `B3-contract-reactive` | Greedy + PACT weights, reactive risk instead of MILD |
| `B3` | Greedy value arbitration |
| `inner-contract` | Inner arbitration only (all claims, C1 resolved by weight) |
| `all-reject` | No arbitration |
| `W3R` | PACT with a projection inner loop (design ablation) |
| `W` | Earlier objective (reference) |
| `W3R-V-noalpha`, `-nobeta`, `-nogamma`, `-scalar`, `-noMILD`, `-nourgency`, `-nopi`, `-noomega`, `-noC3`, `-lin` | Component ablations (Table V of the paper) |

---------------------------------------------------------------------------

## 2. Repository layout

```
intact/                     simulator, xApps, effect models, arbiters (Python package)
  ran/analytic.py           flow-level analytical RAN model (single cell)
  w_revised/                scenario world, paired-twin calibration, outer/inner loops
  w3/core3.py               PACT: request-conditioned effects, probability objective,
                            value-synthesis inner loop, evaluation loop
  mild/, estimation/, inner/, outer/, ...
configs/base.yaml           base configuration (read by every scenario)
scripts/
  run_intact_w_revised.py   scenario generator (seed -> scenario)
  run_w3.py                 runner: calibration + all methods, parallel and resumable
  report_w3.py              statistics and automatic verdicts (REPORT.md)
  make_paper_artifacts.py   every paper table, figure and result paragraph
  verify_code_hash.py       checks the code against the code that produced the results
w3_docs/REGISTRATION_W3V.json   evaluation protocol: seeds, methods, tests, decision rule
runs_w3v_confirmation/      the reported results: per-run metrics (see Section 6)
requirements.txt
```

---------------------------------------------------------------------------

## 3. Installation

Requirements: Linux, macOS or Windows Subsystem for Linux, and Python 3.10 or newer.
No GPU is needed.

```bash
git clone <this repository URL> pact
cd pact
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---------------------------------------------------------------------------

## 4. Reproduce the paper results

All commands are run from the repository root with the virtual environment active.
Every run must pass `--registration w3_docs/REGISTRATION_W3V.json`.

### Step 1 — run the evaluation (automated, long)

```bash
python scripts/run_w3.py --registration w3_docs/REGISTRATION_W3V.json \
       --out runs_reproduction --workers 14
```

**What happens for each of the 20 evaluation scenarios** (seeds 5101–5120):

1. Phase A: controlled sensitivity sweeps.
2. MILD risk-model training.
3. Phase B: paired-twin rollouts, after which the ExtraTrees effect models are fitted.
4. Each registered method is evaluated for 300 control epochs (150 s of network
   time) on identical random streams.

**Cost.** About 9 CPU-minutes per scenario for calibration plus about 1.5 s per
method. With 14 worker processes the full run takes roughly 20–30 minutes. Set
`--workers` to the number of CPU cores you want to use.

**Expected console output.** One line per finished method (for example,
`5101 W3R-V WIF=0.9...`), and `COMPLETE` at the end.

**Resuming.** If the run is interrupted, run the identical command again.
Calibration episodes and evaluation checkpoints are cached. A changed code hash
refuses to reuse the folder, so use a new `--out` in that case.

### Step 2 — statistics and automatic verdicts (automated, seconds)

```bash
python scripts/report_w3.py --registration w3_docs/REGISTRATION_W3V.json \
       --root runs_reproduction
```

**Output.** `runs_reproduction/report_w3/REPORT.md`, containing:

- integrity checks (all methods complete; PACT has zero C1/C2 violations);
- method means;
- registered paired comparisons (sign-flip test, Holm correction, bootstrap CIs);
- the component ablations.

### Step 3 — all paper tables, figures and text (automated, about 1–3 minutes)

```bash
python scripts/make_paper_artifacts.py --root runs_reproduction \
       --registration w3_docs/REGISTRATION_W3V.json --out paper_artifacts --name PACT
```

**Output** (`paper_artifacts/`):

| Artefact | Paper element |
|---|---|
| `tables/tab_main` | Table III (main comparison) |
| `tables/tab_paired`, `tables/tab_secondary`, `tables/tab_service_tests` | Paired tests reported in Sec. V-D and Table IV |
| `tables/tab_service` | Service-protection rates (Table IV, Fig. 2) |
| `tables/tab_ablation` | Table V (ablation) |
| `tables/tab_tenants_fulfilment` | Per-tenant fulfillment (Sec. V-D) |
| `tables/tab_floor_attainability` | C3 attainable-floor analysis |
| `tables/tab_model_evidence` | Held-out effect-model and MILD evidence |
| `tables/tab_environment`, `tab_tenants`, `tab_intents`, `tab_claims`, `tab_hyperparameters` | Setup (Sec. V-A) |
| `figures/fig_service_protection` | Fig. 2 |
| `figures/fig_main_wif`, `fig_paired`, `fig_ablation`, `fig_wif_vs_writes`, `fig_tenants`, `fig_telemetry` | Supplementary figures |
| `text/*.md`, `text/*.tex` | Result paragraphs generated from the numbers |
| `ARTIFACT_INDEX.md` | Index of all artefacts |

Each table is written as `.tex`, `.md` and `.csv`, and each figure as `.pdf` and
`.png`.

---------------------------------------------------------------------------

## 5. Regenerate the artefacts from the shipped results (no simulation)

`runs_w3v_confirmation/` contains the per-run metrics behind the paper (Section 6).
The following command rebuilds the tables and Fig. 2 directly from them in about a
minute:

```bash
python scripts/report_w3.py --registration w3_docs/REGISTRATION_W3V.json \
       --root runs_w3v_confirmation
python scripts/make_paper_artifacts.py --root runs_w3v_confirmation \
       --registration w3_docs/REGISTRATION_W3V.json --out paper_artifacts_shipped --name PACT
```

The per-epoch telemetry files are not stored in the repository, because of their
size. The cached `service_metrics.json` files replace them for the
service-protection metrics; only the optional `fig_telemetry` needs telemetry and is
skipped.

---------------------------------------------------------------------------

## 6. Verify the code and compare with the shipped results

**Code identity.** The confirmation run recorded a hash of all code it used, in
`runs_w3v_confirmation/manifest.json`. Check that this repository is byte-identical
to that code:

```bash
python scripts/verify_code_hash.py --run runs_w3v_confirmation \
       --registration w3_docs/REGISTRATION_W3V.json
```

The expected output is `MATCH`.

**Your reproduction.** After Section 4, compare `runs_reproduction/report_w3/REPORT.md`
with `runs_w3v_confirmation/report_w3/REPORT.md`. With the same library versions,
the numbers should agree. Different NumPy, SciPy or scikit-learn versions can cause
small numerical differences.

**What `runs_w3v_confirmation/` contains:**

- `manifest.json` (seeds, methods, epochs, code hash);
- per scenario `s<seed>/`:
  - `config.json` (the generated scenario);
  - `calibration/evidence.json` (MILD evidence);
  - `calibration/w3_reliability.json`;
  - `calibration_req/evidence_req.json` (held-out effect-model evidence);
- per method `<method>/`:
  - `result.json` (all run-level metrics);
  - `service_metrics.json`;
- `report_w3/` (the report and figures of the confirmation run).

Calibration models (`*.pkl`), checkpoints and per-epoch telemetry are not
included; Step 1 regenerates them.

---------------------------------------------------------------------------

## 7. Scope and limitations

The evaluation uses a single-cell, flow-level analytical RAN model without actuation
lag. Paired twins require an offline simulator that can be copied together with its
random state; they are not an experiment on live users. E2 transport, vendor
schedulers and testbed effects are not modeled. See the Limitations section of the
paper.

---------------------------------------------------------------------------

## 8. Citation

If you use this code, please cite the PACT paper. The BibTeX entry will be added
after publication.
