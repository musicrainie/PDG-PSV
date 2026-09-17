# PDG-PSV — Method Reference Implementation

This repository provides a compact, executable reference implementation of the **PDG-PSV method core** in the ICASSP manuscript *Stateful Geometry Regulation for Self-Supervised EEG Sleep Staging*.

Its purpose is deliberately narrow: **make the paper's Specify–Assess–Enforce / PDG-PSV mechanics easy to inspect in code.** It is not a full experimental-reproduction package.

## What is implemented

- the selected symmetric InfoNCE / NT-Xent base objective;
- **P**: normalized positive-pair squared distance;
- **S**: within-view standardized cross-feature dependence (`log1p` of mean squared off-diagonal dependence);
- **V**: marginal standard-deviation deficit on unnormalized projections;
- fixed or warm-up-derived acceptable boundaries;
- signed residuals `q - t` and residual scaling `sigma`;
- the signed primal geometry term `lambda * sigma * (q - t)`;
- interval aggregation, per-condition EMA, and active-update-count bias correction;
- projected next-interval multiplier updates;
- explicit temporal separation between current-interval primal regulation and next-interval dual adaptation.

The main implementation is intentionally organized in the same order as the manuscript Method. See [`PAPER_TO_CODE.md`](PAPER_TO_CODE.md) for a direct equation-to-function map.

## Minimal run

Requirements: **Python 3.10+** and **PyTorch 2.0+**.

```bash
pip install -r requirements.txt
python demo_minimal.py
```

The demo uses synthetic projected representations only. No EEG dataset is required.

## Core tests

```bash
python -m unittest discover -s tests -v
```

## Scope boundary

This reference implementation begins from the two projected representations `z1, z2`. The encoder/projector implementation, dataset download and preprocessing, dataset-specific runners, full training recipes, checkpoints, and reproduction of the reported EDF20/EDF78/HMC/ISRUC1 results are intentionally outside this release and belong to a later experimental-reproduction package.

This reference implementation contains the manuscript method core; exploratory alternatives outside the reported method are omitted.

## License

This method reference implementation is released under the **Apache License 2.0**. See [`LICENSE`](LICENSE).
