import io
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# -----------------------------
# PNS Linguistic Tables
# -----------------------------
PNS_TABLES: Dict[int, Dict[int, Tuple[float, float, float]]] = {
    5: {1: (0.10, 0.85, 0.90), 2: (0.30, 0.65, 0.70), 3: (0.50, 0.45, 0.45), 4: (0.70, 0.25, 0.20), 5: (0.90, 0.10, 0.05)},
    7: {1: (0.10, 0.80, 0.90), 2: (0.20, 0.70, 0.80), 3: (0.35, 0.60, 0.60), 4: (0.50, 0.40, 0.45), 5: (0.65, 0.30, 0.25), 6: (0.80, 0.20, 0.15), 7: (0.90, 0.10, 0.10)},
    9: {1: (0.05, 0.90, 0.95), 2: (0.10, 0.85, 0.90), 3: (0.20, 0.80, 0.75), 4: (0.35, 0.65, 0.60), 5: (0.50, 0.50, 0.45), 6: (0.65, 0.35, 0.30), 7: (0.80, 0.25, 0.20), 8: (0.90, 0.15, 0.10), 9: (0.95, 0.05, 0.05)},
    11: {1: (0.05, 0.90, 0.95), 2: (0.10, 0.80, 0.85), 3: (0.20, 0.70, 0.75), 4: (0.30, 0.60, 0.65), 5: (0.40, 0.50, 0.55), 6: (0.50, 0.45, 0.45), 7: (0.60, 0.40, 0.35), 8: (0.70, 0.30, 0.25), 9: (0.80, 0.20, 0.15), 10: (0.90, 0.15, 0.10), 11: (0.95, 0.05, 0.05)},
}


# -----------------------------
# Helpers
# -----------------------------
def is_bc_row(values: List[str]) -> bool:
    cleaned = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        cleaned.append(s.upper())
    if not cleaned:
        return False
    return all(x in {"B", "C"} for x in cleaned)


def coerce_int_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        try:
            out[c] = out[c].apply(lambda x: int(str(x).strip()))
        except Exception as e:
            raise ValueError(f"Column '{c}' contains non-integer values. ({e})")
    return out


def validate_score_range(df_int: pd.DataFrame, scale: int) -> None:
    lo, hi = 1, scale
    bad = (df_int < lo) | (df_int > hi)
    if bad.values.any():
        idx = np.argwhere(bad.values)[0]
        r, c = idx[0], idx[1]
        raise ValueError(f"Invalid crisp score: {df_int.iloc[r, c]}. Allowed range for {scale}-point scale is {lo}..{hi}.")


def map_crisp_to_pns(df_int: pd.DataFrame, scale: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = PNS_TABLES[scale]
    m, n = df_int.shape
    tau = np.zeros((m, n), dtype=float)
    xi = np.zeros((m, n), dtype=float)
    eta = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            t, x, e = table[int(df_int.iat[i, j])]
            tau[i, j], xi[i, j], eta[i, j] = t, x, e
    return tau, xi, eta


def normalize_pns(tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, crit_types: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, n = tau.shape
    tau_n = np.zeros_like(tau)
    xi_n = np.zeros_like(xi)
    eta_n = np.zeros_like(eta)

    for j in range(n):
        ctype = crit_types[j].upper()
        if ctype not in {"B", "C"}:
            raise ValueError("Criterion types must be 'B' or 'C' for every criterion.")
        if ctype == "B":
            tmax, xmax, emax = float(np.max(tau[:, j])), float(np.max(xi[:, j])), float(np.max(eta[:, j]))
            if tmax == 0 or xmax == 0 or emax == 0:
                raise ValueError(f"Normalization failed: max component is 0 for criterion {j+1}.")
            tau_n[:, j] = tau[:, j] / tmax
            xi_n[:, j] = xi[:, j] / xmax
            eta_n[:, j] = eta[:, j] / emax
        else:
            tmin, xmin, emin = float(np.min(tau[:, j])), float(np.min(xi[:, j])), float(np.min(eta[:, j]))
            if np.any(tau[:, j] == 0) or np.any(xi[:, j] == 0) or np.any(eta[:, j] == 0):
                raise ValueError(f"Normalization failed: a component is 0 in criterion {j+1}.")
            tau_n[:, j] = tmin / tau[:, j]
            xi_n[:, j] = xmin / xi[:, j]
            eta_n[:, j] = emin / eta[:, j]
    return tau_n, xi_n, eta_n


def apply_weights(tau_n: np.ndarray, xi_n: np.ndarray, eta_n: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tau_n * w.reshape(1, -1), xi_n * w.reshape(1, -1), eta_n * w.reshape(1, -1)


def compute_ideals(tau_w: np.ndarray, xi_w: np.ndarray, eta_w: np.ndarray, crit_types: List[str]):
    n = tau_w.shape[1]
    tau_p = np.zeros(n); xi_p = np.zeros(n); eta_p = np.zeros(n)
    tau_n = np.zeros(n); xi_n = np.zeros(n); eta_n = np.zeros(n)
    for j in range(n):
        if crit_types[j].upper() == "B":
            tau_p[j], xi_p[j], eta_p[j] = np.max(tau_w[:, j]), np.min(xi_w[:, j]), np.min(eta_w[:, j])
            tau_n[j], xi_n[j], eta_n[j] = np.min(tau_w[:, j]), np.max(xi_w[:, j]), np.max(eta_w[:, j])
        else:
            tau_p[j], xi_p[j], eta_p[j] = np.min(tau_w[:, j]), np.max(xi_w[:, j]), np.max(eta_w[:, j])
            tau_n[j], xi_n[j], eta_n[j] = np.max(tau_w[:, j]), np.min(xi_w[:, j]), np.min(eta_w[:, j])
    return tau_p, xi_p, eta_p, tau_n, xi_n, eta_n


def dist_pn_euclidean(tau_row, xi_row, eta_row, tau_ideal, xi_ideal, eta_ideal) -> float:
    n = tau_row.shape[0]
    diff2 = (tau_row - tau_ideal) ** 2 + (xi_row - xi_ideal) ** 2 + (eta_row - eta_ideal) ** 2
    return float(math.sqrt((1.0 / (3.0 * n)) * float(np.sum(diff2))))


def dist_pn_hamming(tau_row, xi_row, eta_row, tau_ideal, xi_ideal, eta_ideal) -> float:
    n = tau_row.shape[0]
    s = np.abs(tau_row - tau_ideal) + np.abs(xi_row - xi_ideal) + np.abs(eta_row - eta_ideal)
    return float((1.0 / (3.0 * n)) * float(np.sum(s)))


def dist_fiq(tau_row, xi_row, eta_row, tau_ideal, xi_ideal, eta_ideal) -> float:
    """
    Correct FIQ distance implementation matching Eq. (23)-(24):

    S_i^{±} = (1/(3n)) * Σ_j  [ wτ*|Δτ|^{pτ} + wξ*|Δξ|^{pξ} + wη*|Δη|^{pη} ]^{1/p}

    IMPORTANT:
    The outer exponent 1/p is applied per-criterion j, using p computed from that pair (A_ij, Ideal_j),
    then summed across criteria. This matches the worked example in the paper.
    """
    n = tau_row.shape[0]

    d_tau = np.abs(tau_row - tau_ideal)
    d_xi = np.abs(xi_row - xi_ideal)
    d_eta = np.abs(eta_row - eta_ideal)

    # Adaptive weights
    w_tau = 1.0 - (xi_row * xi_ideal)
    w_eta = 1.0 - (xi_row * xi_ideal)
    w_xi = 1.0 + (np.abs(tau_row - eta_ideal) + np.abs(eta_row - tau_ideal)) / 2.0

    # Power parameters
    p_tau = 1.0 + (xi_row + xi_ideal) / 2.0
    p_eta = 1.0 + (xi_row + xi_ideal) / 2.0
    p_xi = 2.0 - np.abs(tau_row - eta_ideal)

    # Norm order (per criterion)
    p = 2.0 - (xi_row * xi_ideal)
    p = np.maximum(p, 1e-9)  # safety

    inner = (w_tau * (d_tau ** p_tau)) + (w_xi * (d_xi ** p_xi)) + (w_eta * (d_eta ** p_eta))

    contrib = inner ** (1.0 / p)  # per-criterion exponent
    return float((1.0 / (3.0 * n)) * float(np.sum(contrib)))


def format_triplets(tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, decimals: int = 2) -> pd.DataFrame:
    m, n = tau.shape
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = f"({tau[i,j]:.{decimals}f}, {xi[i,j]:.{decimals}f}, {eta[i,j]:.{decimals}f})"
    return pd.DataFrame(out)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    return buf.read()


def sample_dataset_bytes_excel(scale: int) -> bytes:
    crit_names = ["C1", "C2", "C3", "C4"]
    types = pd.DataFrame([["", "B", "B", "C", "C"]], columns=["Alt"] + crit_names)
    df = pd.DataFrame({"Alt": ["A1", "A2", "A3", "A4", "A5"], "C1": [7, 6, 8, 5, 9], "C2": [8, 7, 6, 9, 7], "C3": [4, 5, 6, 3, 4], "C4": [6, 7, 5, 8, 6]})
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        types.to_excel(writer, sheet_name="Input_Matrix", index=False, header=True)
        df.to_excel(writer, sheet_name="Input_Matrix", index=False, header=True, startrow=1)
    out.seek(0)
    return out.read()


def build_pdf_report(distance_name: str, scale: int, crit_meta: pd.DataFrame, result: pd.DataFrame, top_k: int = 10) -> bytes:
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    heading = styles["Heading2"]
    h1 = styles["Heading1"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    story = []
    story.append(Paragraph("PNTOPSIS Report", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Settings", heading))
    story.append(Paragraph(f"Linguistic scale: {scale}-point", normal))
    story.append(Paragraph(f"Distance: {distance_name}", normal))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Criteria metadata", heading))
    meta_tbl_data = [list(crit_meta.columns)] + crit_meta.values.tolist()
    meta_tbl = Table(meta_tbl_data, hAlign="LEFT")
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Ranking results", heading))
    res = result.copy().reset_index().rename(columns={"index": "Alternative"}).head(top_k)
    res_tbl_data = [list(res.columns)] + res.round(6).values.tolist()
    res_tbl = Table(res_tbl_data, hAlign="LEFT")
    res_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(res_tbl)
    story.append(Spacer(1, 0.4*cm))

    best_name = result.index[0]
    best = result.iloc[0]
    story.append(Paragraph("Interpretation", heading))
    story.append(Paragraph(f"{best_name} is ranked best because it has the highest relative closeness P_i = {best['P_i']:.6f}.", normal))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PNTOPSIS Ranking", layout="wide")
st.title("PNTOPSIS Ranking (Pythagorean Neutrosophic TOPSIS)")

st.sidebar.header("Settings")
scale = st.sidebar.selectbox("Select linguistic scale", options=[5, 7, 9, 11], index=3)
decimals = st.sidebar.slider("Triplet display decimals", 2, 6, 2)
distance_name = st.sidebar.selectbox("Distance measure", ["PN-Euclidean", "PN-Hamming", "FIQ"], index=0)

st.sidebar.subheader("Sample dataset")
st.sidebar.download_button(
    "Download sample Excel",
    data=sample_dataset_bytes_excel(scale=scale),
    file_name=f"pntopsis_sample_{scale}scale.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
weight_mode = st.sidebar.radio("Criteria weights", ["Equal Weights", "Manual Weights"], index=0)

table_df = pd.DataFrame([{"Score": k, "τ": v[0], "ξ": v[1], "η": v[2]} for k, v in PNS_TABLES[scale].items()]).sort_values("Score")

st.subheader("1) Data input")

def read_uploaded_file(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

raw_df: Optional[pd.DataFrame] = None
if uploaded is not None:
    try:
        raw_df = read_uploaded_file(uploaded)
        st.success(f"Loaded file: {uploaded.name}")
        st.dataframe(raw_df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.info("No file uploaded. Use the manual input grid below (optional).")
    m = st.number_input("Number of alternatives (m)", min_value=2, max_value=500, value=5, step=1)
    n = st.number_input("Number of criteria (n)", min_value=2, max_value=200, value=4, step=1)
    default_grid = pd.DataFrame(np.ones((m, n), dtype=int), columns=[f"C{j+1}" for j in range(n)])
    raw_df = st.data_editor(default_grid, use_container_width=True, key="manual_grid")

if raw_df is None or raw_df.shape[0] == 0:
    st.error("Empty input.")
    st.stop()

df = raw_df.copy()
first_col_values = df.iloc[:, 0].tolist()
first_col_is_alt = False
try:
    _ = [int(str(v).strip()) for v in first_col_values[: min(10, len(first_col_values))]]
except Exception:
    first_col_is_alt = True

if first_col_is_alt:
    alt_names = df.iloc[:, 0].astype(str).tolist()
    df_mat = df.iloc[:, 1:].copy()
    crit_names = [str(c) for c in df_mat.columns]
else:
    alt_names = [f"A{i+1}" for i in range(df.shape[0])]
    df_mat = df.copy()
    crit_names = [str(c) for c in df_mat.columns]

first_row = df_mat.iloc[0, :].tolist()
has_bc = is_bc_row(first_row)
if has_bc:
    crit_types = [str(x).strip().upper() for x in first_row]
    df_scores = df_mat.iloc[1:, :].copy()
    alt_names = alt_names[1:]
    st.info("Detected criterion types row (top row) in the uploaded file.")
else:
    crit_types = ["B"] * len(crit_names)
    df_scores = df_mat.copy()
    st.warning("No criterion types row found. Defaulting all criteria to Benefit (B). Please adjust below before computing.")

st.subheader("2) Criterion types (Benefit/Cost)")
type_df = pd.DataFrame([crit_types], columns=crit_names, index=["Type (B/C)"])
edited_type_df = st.data_editor(type_df, use_container_width=True, key="crit_types_editor")
crit_types = [str(edited_type_df.iloc[0, j]).strip().upper() for j in range(len(crit_names))]
if any(t not in {"B", "C"} for t in crit_types):
    st.error("Criterion types must be only 'B' or 'C' for every criterion.")
    st.stop()

try:
    crisp_df = df_scores.copy()
    crisp_df = crisp_df.loc[:, ~crisp_df.columns.astype(str).str.contains("^Unnamed")]
    crisp_df.columns = crit_names[: crisp_df.shape[1]]
    crisp_df = crisp_df.reset_index(drop=True)
    crisp_int = coerce_int_matrix(crisp_df)
    validate_score_range(crisp_int, scale)
except Exception as e:
    st.error(f"Input validation error: {e}")
    st.stop()

st.subheader("3) Crisp decision matrix (validated)")
crisp_show = crisp_int.copy()
crisp_show.index = alt_names
st.dataframe(crisp_show, use_container_width=True)

st.subheader("4) Criteria weights")
n_criteria = len(crit_names)
if weight_mode == "Equal Weights":
    w = np.array([1.0 / n_criteria] * n_criteria, dtype=float)
    st.info(f"Using equal weights: each w_j = 1/{n_criteria} = {1.0/n_criteria:.6f}")
else:
    w_default = pd.DataFrame([[round(1.0 / n_criteria, 6)] * n_criteria], columns=crit_names, index=["w"])
    w_edit = st.data_editor(w_default, use_container_width=True, key="weights_editor")
    try:
        w = np.array([float(w_edit.iloc[0, j]) for j in range(n_criteria)], dtype=float)
    except Exception as e:
        st.error(f"Manual weights must be numeric. ({e})")
        st.stop()
    w_sum = float(np.sum(w))
    if abs(w_sum - 1.0) > 1e-3:
        st.warning(f"Sum of weights = {w_sum:.6f} (should be 1.000000). Computation will proceed anyway.")

st.subheader("Reference: Selected PNS linguistic table")
st.dataframe(table_df, use_container_width=True)

st.subheader("5) Compute PNTOPSIS ranking")
run = st.button("Run PNTOPSIS", type="primary")
if not run:
    st.stop()

try:
    tau, xi, eta = map_crisp_to_pns(crisp_int, scale)
    tau_n, xi_n, eta_n = normalize_pns(tau, xi, eta, crit_types)
    tau_w, xi_w, eta_w = apply_weights(tau_n, xi_n, eta_n, w)
    tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg = compute_ideals(tau_w, xi_w, eta_w, crit_types)
except Exception as e:
    st.error(str(e))
    st.stop()

if distance_name == "PN-Euclidean":
    dist_fn = dist_pn_euclidean
elif distance_name == "PN-Hamming":
    dist_fn = dist_pn_hamming
else:
    dist_fn = dist_fiq

m_alt = tau_w.shape[0]
S_plus = np.zeros(m_alt); S_minus = np.zeros(m_alt)
for i in range(m_alt):
    S_plus[i] = dist_fn(tau_w[i, :], xi_w[i, :], eta_w[i, :], tau_p, xi_p, eta_p)
    S_minus[i] = dist_fn(tau_w[i, :], xi_w[i, :], eta_w[i, :], tau_neg, xi_neg, eta_neg)

Pi = S_minus / (S_plus + S_minus)
result = pd.DataFrame({"S_i_plus": S_plus, "S_i_minus": S_minus, "P_i": Pi}, index=alt_names)
result["Rank"] = (-result["P_i"]).rank(method="dense").astype(int)
result = result.sort_values(["Rank", "P_i"], ascending=[True, False])

st.subheader("Outputs")
pns_df = format_triplets(tau, xi, eta, decimals=decimals); pns_df.columns = crit_names; pns_df.index = alt_names
norm_df = format_triplets(tau_n, xi_n, eta_n, decimals=decimals); norm_df.columns = crit_names; norm_df.index = alt_names
w_df2 = format_triplets(tau_w, xi_w, eta_w, decimals=decimals); w_df2.columns = crit_names; w_df2.index = alt_names

colA, colB = st.columns(2)
with colA:
    st.markdown("### Converted PNS matrix (numeric triplets)")
    st.dataframe(pns_df, use_container_width=True)
with colB:
    st.markdown("### Normalized PNS matrix (numeric triplets)")
    st.dataframe(norm_df, use_container_width=True)

st.markdown("### Weighted normalized PNS matrix (numeric triplets)")
st.dataframe(w_df2, use_container_width=True)

pis = pd.DataFrame({"tau+": tau_p, "xi+": xi_p, "eta+": eta_p}, index=crit_names)
nis = pd.DataFrame({"tau-": tau_neg, "xi-": xi_neg, "eta-": eta_neg}, index=crit_names)
pisnis = pd.concat([pis, nis], axis=1)
st.markdown("### PIS (V+) and NIS (V-) per criterion")
st.dataframe(pisnis, use_container_width=True)

st.markdown("### Distances, closeness, and ranking")
st.dataframe(result, use_container_width=True)
st.markdown("### Closeness chart (P_i)")
st.bar_chart(result["P_i"])

best_alt = result.index[0]
best = result.iloc[0]
st.subheader("Summary interpretation")
st.write(
    f"{best_alt} is the best alternative because it has the highest relative closeness P_i = {best['P_i']:.6f}. "
    f"Under {distance_name}, it achieves a smaller distance to the positive ideal (S_i_plus = {best['S_i_plus']:.6f}) "
    f"and a larger distance from the negative ideal (S_i_minus = {best['S_i_minus']:.6f})."
)

st.subheader("Export")
meta = pd.DataFrame({"Criterion": crit_names, "Type (B/C)": crit_types, "Weight": w})
sheets = {
    "Crisp_Matrix": crisp_show,
    "PNS_Matrix": pns_df,
    "Normalized": norm_df,
    "Weighted_Normalized": w_df2,
    "PIS_NIS": pisnis,
    "Results": result,
    "Meta": meta.set_index("Criterion"),
    "Linguistic_Table": table_df.set_index("Score"),
}
xlsx_bytes = to_excel_bytes(sheets)
st.download_button(
    "Download results (Excel)",
    data=xlsx_bytes,
    file_name="pntopsis_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

pdf_bytes = build_pdf_report(distance_name=distance_name, scale=scale, crit_meta=meta, result=result, top_k=10)
st.download_button("Download report (PDF)", data=pdf_bytes, file_name="pntopsis_report.pdf", mime="application/pdf")
