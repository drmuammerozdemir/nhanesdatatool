# pages/2_Analysis.py
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NHANES Analysis", layout="wide")
st.title("NHANES Pre vs Post – Analiz (CSV)")

st.write(
    "Pre ve Post CSV dosyalarını yükle. Otomatik olarak indeksleri (NLR/PLR/SII) hesaplar; "
    "özet tabloya **etki büyüklüğü (Cliff’s δ)**, **% değişim**, **FDR düzeltilmiş p** ve "
    "**CRP için log10 dönüşümü** opsiyonlarını ekler. Grafikler: jitter + medyan çizgisi + bracket p-etiketi."
)

# ---------------------------
# Helpers
# ---------------------------
def robust_read_csv(uploaded_file) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1254", "latin1"]:
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            uploaded_file.seek(0)
            continue
    raise ValueError("CSV okunamadı. Encoding uyumsuz olabilir.")

def ensure_upper_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    return df

def find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer absolute neutrophil/lymphocyte counts if present.
    Fallback to percentages (LBXNEPCT / LBXLYPCT).
    Creates: NLR, PLR, SII, INDEX_MODE
    """
    out = ensure_upper_cols(df.copy())

    numeric_candidates = [
        "LBXNEPCT", "LBXLYPCT", "LBXPLTSI", "LBXCRP", "RIDAGEYR",
        "LBXNE", "LBXLY", "LBXNEUT", "LBXLYMPH", "LBXNEUTSI", "LBXLYMPSI"
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    neut_abs_col = find_first_col(out, ["LBXNE", "LBXNEUT", "LBXNEUTSI"])
    lymph_abs_col = find_first_col(out, ["LBXLY", "LBXLYMPH", "LBXLYMPSI"])

    neut_pct_col = "LBXNEPCT" if "LBXNEPCT" in out.columns else None
    lymph_pct_col = "LBXLYPCT" if "LBXLYPCT" in out.columns else None

    if neut_abs_col and lymph_abs_col:
        neut = out[neut_abs_col]
        lymph = out[lymph_abs_col].replace(0, np.nan)
        out["INDEX_MODE"] = "absolute"
    else:
        neut = out[neut_pct_col] if neut_pct_col else np.nan
        lymph = out[lymph_pct_col].replace(0, np.nan) if lymph_pct_col else np.nan
        out["INDEX_MODE"] = "percent"

    if isinstance(neut, pd.Series) and isinstance(lymph, pd.Series):
        out["NLR"] = neut / lymph
        if "LBXPLTSI" in out.columns:
            out["PLR"] = out["LBXPLTSI"] / lymph
            out["SII"] = (out["LBXPLTSI"] * neut) / lymph
        else:
            out["PLR"] = np.nan
            out["SII"] = np.nan
    else:
        out["NLR"] = np.nan
        out["PLR"] = np.nan
        out["SII"] = np.nan

    return out

def median_iqr(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    med = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    return med, q1, q3

def mann_whitney_u(x, y):
    """Works with pandas Series or numpy arrays; returns (U, p)."""
    try:
        from scipy.stats import mannwhitneyu

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if x.size < 3 or y.size < 3:
            return np.nan, np.nan

        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan

def cliffs_delta(x, y, max_pairs: int = 2_000_000, seed: int = 42) -> float:
    """
    Cliff's delta effect size.
    If nx*ny huge, subsample to keep it fast.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return np.nan

    if nx * ny > max_pairs:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=min(nx, 1500), replace=False)
        y = rng.choice(y, size=min(ny, 1500), replace=False)
        nx, ny = len(x), len(y)

    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return (gt - lt) / (nx * ny)

def p_label(p: float) -> str:
    """Clinical-style p reporting."""
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    if p < 0.05:
        return "<0.05"
    return "NS"

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.array(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out_vals = np.empty_like(ranked)
    out_vals[order] = np.clip(q, 0, 1)
    out[mask] = out_vals
    return out

def format_med_iqr(med, q1, q3) -> str:
    if np.isfinite(med):
        return f"{med:.3g} [{q1:.3g}–{q3:.3g}]"
    return "NA"

def safe_n(series) -> int:
    return int(pd.to_numeric(pd.Series(series), errors="coerce").dropna().shape[0])

def stripplot_with_p(ax, data_groups, labels, p_text, title="", ylabel="", point_size=9):
    """
    Jitter scatter + median line + bracket + p label.
    Horizontal thick line = MEDIAN.
    """
    x_positions = [1, 2]

    cleaned = []
    for vals in data_groups:
        v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
        cleaned.append(v)

    for i, vals in enumerate(cleaned):
        x = np.random.normal(x_positions[i], 0.045, size=len(vals))
        ax.scatter(x, vals, alpha=0.6, s=point_size)

        med = np.median(vals) if len(vals) else np.nan
        if np.isfinite(med):
            ax.hlines(med, x_positions[i]-0.22, x_positions[i]+0.22, linewidth=3)

    # bracket height based on data range
    y_max = max([v.max() if len(v) else 0 for v in cleaned])
    y_min = min([v.min() if len(v) else 0 for v in cleaned])
    yr = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    h = yr * 0.08

    ax.plot([1, 1, 2, 2], [y_max, y_max+h, y_max+h, y_max], lw=1.5)
    ax.text(1.5, y_max + h*1.05, f"p {p_text}", ha="center", va="bottom")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

# ---------------------------
# UI: Upload
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    pre_file = st.file_uploader("Pre CSV (nhanes_pre_pandemic.csv)", type=["csv"], key="pre_csv")
with col2:
    post_file = st.file_uploader("Post CSV (nhanes_post_pandemic.csv)", type=["csv"], key="post_csv")

st.divider()

if not pre_file or not post_file:
    st.info("Devam etmek için iki CSV’yi de yükle.")
    st.stop()

pre = ensure_upper_cols(robust_read_csv(pre_file))
post = ensure_upper_cols(robust_read_csv(post_file))

if "PERIOD" not in pre.columns:
    pre["PERIOD"] = "Pre"
if "PERIOD" not in post.columns:
    post["PERIOD"] = "Post"

pre = compute_indices(pre)
post = compute_indices(post)

df = pd.concat([pre, post], ignore_index=True, sort=False)

# ---------------------------
# Sidebar filters & plot options
# ---------------------------
st.sidebar.header("Filtreler / Grafik ayarları")

age_min = st.sidebar.number_input("Minimum yaş", value=18, min_value=0, max_value=120)
age_max = st.sidebar.number_input("Maksimum yaş", value=120, min_value=0, max_value=120)

exclude_crp_gt10 = st.sidebar.checkbox("CRP > 10 mg/L dışla (akut inflamasyon)", value=False)
log_transform_crp = st.sidebar.checkbox("CRP için log10 dönüşümü oluştur (LBXCRP_LOG10)", value=True)

point_size = st.sidebar.slider("Nokta boyutu (scatter)", min_value=3, max_value=25, value=9, step=1)
dpi_out = st.sidebar.selectbox("İndirme DPI", options=[150, 300, 600], index=1)

# Apply filters
df_f = df.copy()

if "RIDAGEYR" in df_f.columns:
    df_f = df_f[
        (df_f["RIDAGEYR"].notna()) &
        (df_f["RIDAGEYR"] >= age_min) &
        (df_f["RIDAGEYR"] <= age_max)
    ]

if exclude_crp_gt10 and "LBXCRP" in df_f.columns:
    df_f = df_f[~(pd.to_numeric(df_f["LBXCRP"], errors="coerce") > 10)]

if log_transform_crp and "LBXCRP" in df_f.columns:
    crp_num = pd.to_numeric(df_f["LBXCRP"], errors="coerce").replace(0, np.nan)
    df_f["LBXCRP_LOG10"] = np.log10(crp_num)

pre_f = df_f[df_f["PERIOD"].str.contains("pre", case=False, na=False)].copy()
post_f = df_f[df_f["PERIOD"].str.contains("post", case=False, na=False)].copy()

mode_pre = pre_f["INDEX_MODE"].mode().iloc[0] if "INDEX_MODE" in pre_f.columns and not pre_f.empty else "NA"
mode_post = post_f["INDEX_MODE"].mode().iloc[0] if "INDEX_MODE" in post_f.columns and not post_f.empty else "NA"
st.caption(f"İndeks hesaplama modu: Pre = **{mode_pre}**, Post = **{mode_post}**")

# ---------------------------
# Variable selection
# ---------------------------
st.subheader("1) Değişken seçimi")

default_vars = [c for c in [
    "LBXCRP", "LBXCRP_LOG10", "NLR", "PLR", "SII",
    "LBXWBCSI", "LBXNEPCT", "LBXLYPCT", "LBXPLTSI"
] if c in df_f.columns]

vars_to_analyze = st.multiselect(
    "Analiz edilecek değişkenler",
    options=sorted(df_f.columns),
    default=default_vars
)

if not vars_to_analyze:
    st.warning("En az bir değişken seç.")
    st.stop()

# ---------------------------
# Summary table
# ---------------------------
st.subheader("2) Özet tablo (Median [IQR]) + Mann–Whitney U + Etki büyüklüğü + FDR")

rows = []
pvals_numeric = []

for v in vars_to_analyze:
    if v not in pre_f.columns or v not in post_f.columns:
        rows.append({
            "Variable": v,
            "Pre Median [Q1–Q3]": "NA",
            "Post Median [Q1–Q3]": "NA",
            "%Δ (Post vs Pre)": "NA",
            "Mann–Whitney U": "NA",
            "p-value": "NA",
            "Cliff's δ": "NA",
            "n (Pre)": safe_n(pre_f[v]) if v in pre_f.columns else 0,
            "n (Post)": safe_n(post_f[v]) if v in post_f.columns else 0,
        })
        pvals_numeric.append(np.nan)
        continue

    pre_med, pre_q1, pre_q3 = median_iqr(pre_f[v])
    post_med, post_q1, post_q3 = median_iqr(post_f[v])

    pct_change = np.nan
    if np.isfinite(pre_med) and pre_med != 0 and np.isfinite(post_med):
        pct_change = 100.0 * (post_med - pre_med) / pre_med

    pre_vals_s = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
    post_vals_s = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()

    stat, p = mann_whitney_u(pre_vals_s, post_vals_s)
    delta = cliffs_delta(pre_vals_s, post_vals_s)

    rows.append({
        "Variable": v,
        "Pre Median [Q1–Q3]": format_med_iqr(pre_med, pre_q1, pre_q3),
        "Post Median [Q1–Q3]": format_med_iqr(post_med, post_q1, post_q3),
        "%Δ (Post vs Pre)": f"{pct_change:.1f}%" if np.isfinite(pct_change) else "NA",
        "Mann–Whitney U": f"{stat:.3g}" if np.isfinite(stat) else "NA",
        "p-value": p_label(p),
        "Cliff's δ": f"{delta:.3f}" if np.isfinite(delta) else "NA",
        "n (Pre)": safe_n(pre_vals_s),
        "n (Post)": safe_n(post_vals_s),
    })

    pvals_numeric.append(p if np.isfinite(p) else np.nan)

summary = pd.DataFrame(rows)

# FDR correction (q-values) + label
p_arr = pd.to_numeric(pd.Series(pvals_numeric), errors="coerce").values
q_arr = fdr_bh(p_arr)
summary["p_FDR"] = [p_label(q) for q in q_arr]

st.dataframe(summary, use_container_width=True)

st.download_button(
    "⬇️ Özet tabloyu CSV indir",
    data=summary.to_csv(index=False).encode("utf-8-sig"),
    file_name="nhanes_pre_post_summary_enhanced.csv",
    mime="text/csv"
)

# ---------------------------
# Plots (single + multi grid)
# ---------------------------
st.subheader("3) Grafikler")

# Single selector kept for convenience
plot_var_single = st.selectbox("Tek grafik için değişken (opsiyonel)", vars_to_analyze)

plot_vars = st.multiselect(
    "Çoklu grafik için değişken(ler) (4 → 2x2, 9 → 3x3 otomatik)",
    options=vars_to_analyze,
    default=[plot_var_single] if plot_var_single in vars_to_analyze else vars_to_analyze[:4]
)

if not plot_vars:
    st.info("Grafik için en az 1 değişken seç.")
    st.stop()

n = len(plot_vars)
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
axes = np.array(axes).reshape(-1)

for i, v in enumerate(plot_vars):
    ax = axes[i]

    pre_vals = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
    post_vals = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()

    _, p_raw = mann_whitney_u(pre_vals, post_vals)
    p_txt = p_label(p_raw)

    stripplot_with_p(
        ax=ax,
        data_groups=[pre_vals, post_vals],
        labels=["Pre", "Post"],
        p_text=p_txt,
        title=v,
        ylabel=v,
        point_size=point_size
    )

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

st.caption("Bilgi: Grafiklerdeki kalın yatay çizgi **medyanı** gösterir (ortalama değil).")

# Download 300 DPI (or selected)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=int(dpi_out), bbox_inches="tight")
buf.seek(0)

st.download_button(
    f"⬇️ Grafik(ler)i indir (PNG, {dpi_out} DPI)",
    data=buf,
    file_name=f"nhanes_plots_{dpi_out}dpi.png",
    mime="image/png"
)

# Optional histogram overlay for the single selected variable
with st.expander("Histogram (tek değişken, opsiyonel)", expanded=False):
    v = plot_var_single
    pre_vals_h = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
    post_vals_h = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(pre_vals_h, bins=40, alpha=0.6, label="Pre")
    ax2.hist(post_vals_h, bins=40, alpha=0.6, label="Post")
    ax2.set_title(f"{v} dağılımı (Pre vs Post)")
    ax2.set_xlabel(v)
    ax2.set_ylabel("Count")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

st.markdown("### Filtrelenmiş veri örnekleri")
cA, cB = st.columns(2)
with cA:
    st.write("Pre head")
    st.dataframe(pre_f.head(10), use_container_width=True)
with cB:
    st.write("Post head")
    st.dataframe(post_f.head(10), use_container_width=True)
