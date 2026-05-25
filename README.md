# LCSAR Code

Core code for the manuscript:

**LCSAR: A Local Community Enhanced Spatial Autoregressive Model for Analyzing Birth Outcomes**

This repository is organized as a compact methods-code release. Core algorithms and the minimal example are both kept under `src`.

## Repository Layout

```text
LCSAR_open_code/
  README.md
  pyproject.toml
  requirements.txt
  LICENSE.md
  src/
    lcsar/
      rws.py          # Random Walk with Sweeping core routines
      lcsar.py        # LCSAR least-squares estimation routines
      simulation.py   # Small simulation utilities used by the example
    examples/
      run_core_demo.py
```

## Quick Start

```bash
python -m pip install -e .
python src/examples/run_core_demo.py
```

The demo simulates an SBM network, constructs an RWS-based spatial weight matrix, generates a SAR response from the oracle community weights, and fits the LCSAR least-squares estimator.
