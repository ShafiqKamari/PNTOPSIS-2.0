# PNTOPSIS Streamlit App

Implements PNTOPSIS (Pythagorean Neutrosophic TOPSIS) with crisp input scores mapped to PNS triplets.

## Features
- Linguistic mappings: 5, 7, 9, 11-point (strict lookup).
- Criterion types: Benefit (B) and Cost (C) from file or editable in UI.
- Weights: equal or manual (no auto-normalization). Warning if sum of weights is not 1.
- Distance toggle: PN-Euclidean, PN-Hamming, FIQ (implemented exactly as Eq. 23–24).
- Sample dataset download button.
- Summary interpretation text for the best alternative.
- Export: Excel (all matrices) and PDF report.

## Input format (Excel/CSV)
Recommended: include a top row of criterion types.

| Alt | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
|   | B | B | C | C |
| A1 | 7 | 8 | 4 | 6 |

If the B/C row is missing, the app defaults all criteria to B and you can adjust before running.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
