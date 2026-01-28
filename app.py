# pages/2_Analysis.py
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import statsmodels.formula.api as smf

st.set_page_config(page_title="NHANES Analysis", layout="wide")
st.title("NHANES Pre vs Post – Detaylı Analiz")

st.write(
    "Verileri yükleyin. Tablo; **Shapiro-Wilk normallik testini**, **Yaş Düzeltmeli ANCOVA** sonucunu "
    "ve **Etki Büyüklüğünü** gösterir. Yan menüden 'Parametrik Teste Zorla' seçeneği ile MWU yerine T-test yapabilirsiniz."
)

# ---------------------------
# 1. HARİTALAMA (DÜZELTİLMİŞ LİSTE)
# ---------------------------
RENAME_MAP = {
    # --- KİMLİK & AĞIRLIK ---
    "SEQN": "ID",
    "WTPH2YR": "WEIGHT_LAB",
    # --- DEMOGRAFİ ---
    "RIDAGEYR": "AGE",
    "RIAGENDR": "SEX",
    "RIDRETH1": "RACE",
    "INDFMPIR": "PIR",
    "PERIOD":   "PERIOD",
    # --- LAB DEĞERLERİ ---
    "LBXWBCSI": "WBC",       
    "LBXLYPCT": "LYMPH_PCT", 
    "LBXMOPCT": "MONO_PCT",  
    "LBXNEPCT": "NEUT_PCT",  
    "LBXEOPCT": "EOS_PCT",   
    "LBXBAPCT": "BASO_PCT",  
    # Mutlak Sayılar
    "LBDLYMNO": "LYMPH_ABS", 
    "LBDMONO":  "MONO_ABS",  
    "LBDNENO":  "NEUT_ABS",  
    "LBDEONO":  "EOS_ABS",   
    "LBDBANO":  "BASO_ABS",  
    # Eritrosit / Hemoglobin
    "LBXRBCSI": "RBC",       
    "LBXHGB":   "HGB",       
    "LBXHCT":   "HCT",       
    "LBXMCVSI": "MCV",       
    "LBXMC":    "MCHC",      
    "LBXMCHSI": "MCH",       
    "LBXRDW":   "RDW",       
    "LBXPLTSI": "PLT",       
    "LBXMPSI":  "MPV",       
    "LBXNRBC":  "NRBC",      
    # İnflamasyon
    "LBXCRP":   "CRP",
}

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

def p_label_detailed(p: float) -> str:
    """
    NS yerine gerçek değeri gösteren, anlamlıysa < sembolü kullanan format.
    """
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    # Anlamlı değilse (NS ise) gerçek değeri göster
    return f"{p:.3f}"

def check_normality(data):
    """
    Shapiro-Wilk testi.
    N > 5000 ise test çok hassaslaşır, p değeri güvenilmez olabilir ama yine de hesaplıyoruz.
    """
    try:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if len(data) < 3:
            return np.nan, "NA"
        # Scipy shapiro N>5000 uyarısı verebilir, biz sadece p'yi alalım
        stat, p = shapiro(data)
        return p, ("Normal" if p > 0.05 else "Not Normal")
    except:
        return np.nan, "NA"

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_upper_cols(df.copy())
    
    numeric_candidates = ["LBDNENO", "LBDLYMNO", "LBXPLTSI", "LBXNEPCT", "LBXLYPCT"]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    neut_abs = out["LBDNENO"] if "LBDNENO" in out.columns else None
    lymph_abs = out["LBDLYMNO"] if "LBDLYMNO" in out.columns else None
    plt_count = out["LBXPLTSI"] if "LBXPLTSI" in out.columns else None
    neut_pct = out["LBXNEPCT"] if "LBXNEPCT" in out.columns else None
    lymph_pct = out["LBXLYPCT"] if "LBXLYPCT" in out.columns else None

    if neut_abs is not None and lymph_abs is not None:
        out["NLR"] = neut_abs / lymph_abs.replace(0, np.nan)
        out["INDEX_MODE"] = "absolute (LBD)"
        if plt_count is not None:
            out["PLR"] = plt_count / lymph_abs.replace(0, np.nan)
            out["SII"] = (plt_count * neut_abs) / lymph_abs.replace(0, np.nan)
    elif neut_pct is not None and lymph_pct is not None:
        out["NLR"] = neut_pct / lymph_pct.replace(0, np.nan)
        out["INDEX_MODE"] = "percent (LBX)"
        out["PLR"] = np.nan
        out["SII"] = np.nan
    else:
        out["NLR"] = np.nan; out["PLR"] = np.nan; out["SII"] = np.nan; out["INDEX_MODE"] = "NA"
    return out

def median_iqr(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, np.nan, np.nan
    med = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    return med, q1, q3

def mean_sd(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, np.nan
    return float(s.mean()), float(s.std())

def cliffs_delta(x, y, max_pairs: int = 2_000_000, seed: int = 42) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3: return np.nan
    if nx * ny > max_pairs:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=min(nx, 1500), replace=False)
        y = rng.choice(y, size=min(ny, 1500), replace=False)
        nx, ny = len(x), len(y)
    gt = 0; lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return (gt - lt) / (nx * ny)

def format_val_disp(val, q1_sd, q3_sd, is_parametric):
    if np.isfinite(val):
        if is_parametric:
            return f"{val:.2f} ± {q1_sd:.2f}" # Mean ± SD
        else:
            return f"{val:.3g} [{q1_sd:.3g}–{q3_sd:.3g}]" # Median [IQR]
    return "NA"

def safe_n(series) -> int:
    return int(pd.to_numeric(pd.Series(series), errors="coerce").dropna().shape[0])

def stripplot_with_p(ax, data_groups, labels, p_text, title="", ylabel="", point_size=9, show_mean=False):
    x_positions = [1, 2]
    cleaned = []
    for vals in data_groups:
        v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
        cleaned.append(v)

    for i, vals in enumerate(cleaned):
        x = np.random.normal(x_positions[i], 0.045, size=len(vals))
        ax.scatter(x, vals, alpha=0.6, s=point_size)
        
        if show_mean:
            center_stat = np.mean(vals) if len(vals) else np.nan
            color_line = 'red' # Mean için kırmızı
        else:
            center_stat = np.median(vals) if len(vals) else np.nan
            color_line = 'black' # Median için siyah

        if np.isfinite(center_stat):
            ax.hlines(center_stat, x_positions[i]-0.22, x_positions[i]+0.22, linewidth=3, colors=color_line)

    y_max = max([v.max() if len(v) else 0 for v in cleaned])
    y_min = min([v.min() if len(v) else 0 for v in cleaned])
    yr = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    h = yr * 0.08

    ax.plot([1, 1, 2, 2], [y_max, y_max+h, y_max+h, y_max], lw=1.5, color='black')
    ax.text(1.5, y_max + h*1.05, f"p {p_text}", ha="center", va="bottom")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

# ---------------------------
# UI: Upload & Processing
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

if "PERIOD" not in pre.columns: pre["PERIOD"] = "Pre"
if "PERIOD" not in post.columns: post["PERIOD"] = "Post"

pre = compute_indices(pre)
post = compute_indices(post)

df = pd.concat([pre, post], ignore_index=True, sort=False)
df = df.rename(columns=RENAME_MAP)

# ---------------------------
# Sidebar: Filters & Settings
# ---------------------------
st.sidebar.header("Ayarlar")

# Test Seçimi (Kritik İstek)
force_parametric = st.sidebar.checkbox("⚠️ Parametrik Teste Zorla (Welch t-test)", value=False, 
                                       help="Normal dağılım olmasa bile Mann-Whitney U yerine T-test yapar.")

st.sidebar.subheader("Filtreler")
age_min = st.sidebar.number_input("Min Yaş", 18, 120, 18)
age_max = st.sidebar.number_input("Max Yaş", 18, 120, 120)

exclude_crp_gt10 = st.sidebar.checkbox("CRP > 10 dışla", False)
log_transform_crp = st.sidebar.checkbox("Log(CRP) dönüşümü", True)

point_size = st.sidebar.slider("Nokta boyutu", 3, 25, 9)
dpi_out = st.sidebar.selectbox("DPI", [150, 300, 600], 1)

# Filtre Uygula
df_f = df.copy()

if "AGE" in df_f.columns:
    df_f = df_f[(df_f["AGE"] >= age_min) & (df_f["AGE"] <= age_max)]

if exclude_crp_gt10 and "CRP" in df_f.columns:
    df_f = df_f[~(pd.to_numeric(df_f["CRP"], errors="coerce") > 10)]

if log_transform_crp and "CRP" in df_f.columns:
    crp_num = pd.to_numeric(df_f["CRP"], errors="coerce").replace(0, np.nan)
    df_f["CRP_LOG10"] = np.log10(crp_num)

pre_f = df_f[df_f["PERIOD"].str.contains("pre", case=False, na=False)].copy()
post_f = df_f[df_f["PERIOD"].str.contains("post", case=False, na=False)].copy()

# ---------------------------
# 1) Değişken Seçimi
# ---------------------------
st.subheader("1) Değişken Seçimi")
default_candidates = ["CRP", "WBC", "NLR", "PLR", "SII", "PLT", "MCHC", "RDW", "NEUT_ABS", "LYMPH_ABS", "AGE"]
default_vars = [c for c in default_candidates if c in df_f.columns]
vars_to_analyze = st.multiselect("Değişkenler", sorted(df_f.columns), default_vars)

if not vars_to_analyze:
    st.stop()

# ---------------------------
# 2) Özet Tablo
# ---------------------------
st.subheader("2) Özet Tablo (Shapiro-Wilk & Detaylı P)")

rows = []
pvals_numeric = []
progress_bar = st.progress(0)

for i, v in enumerate(vars_to_analyze):
    progress_bar.progress((i + 1) / len(vars_to_analyze))
    
    if v not in pre_f.columns or v not in post_f.columns:
        continue

    pre_vals = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
    post_vals = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()

    if len(pre_vals) < 3 or len(post_vals) < 3:
        continue

    # 1. Normallik Testi (Shapiro)
    p_sw_pre, status_pre = check_normality(pre_vals)
    p_sw_post, status_post = check_normality(post_vals)
    
    # Dağılım Normal mi? (Her iki grup da normal olmalı)
    is_normal_dist = (status_pre == "Normal") and (status_post == "Normal")
    
    # 2. Hipotez Testi Seçimi
    # Kullanıcı zorladıysa VEYA dağılım normalse -> T-test
    use_parametric = force_parametric or is_normal_dist
    
    if use_parametric:
        test_name = "Welch t-test"
        # Welch t-test (equal_var=False)
        stat, p_raw = ttest_ind(pre_vals, post_vals, equal_var=False)
        
        # Gösterim için Mean ± SD hesapla
        pre_m, pre_sd = mean_sd(pre_f[v])
        post_m, post_sd = mean_sd(post_f[v])
        pre_disp = format_val_disp(pre_m, pre_sd, 0, True) # q3 yerine 0 geçiyoruz, kullanılmıyor
        post_disp = format_val_disp(post_m, post_sd, 0, True)
        
    else:
        test_name = "Mann-Whitney U"
        stat, p_raw = mannwhitneyu(pre_vals, post_vals, alternative="two-sided")
        
        # Gösterim için Median [IQR]
        pre_med, pre_q1, pre_q3 = median_iqr(pre_f[v])
        post_med, post_q1, post_q3 = median_iqr(post_f[v])
        pre_disp = format_val_disp(pre_med, pre_q1, pre_q3, False)
        post_disp = format_val_disp(post_med, post_q1, post_q3, False)

    # Değişim Yüzdesi (Medyan veya Ortalamaya göre)
    center_pre = np.mean(pre_vals) if use_parametric else np.median(pre_vals)
    center_post = np.mean(post_vals) if use_parametric else np.median(post_vals)
    pct_change = 100.0 * (center_post - center_pre) / center_pre if center_pre != 0 else np.nan
    
    delta = cliffs_delta(pre_vals, post_vals)

    # 3. ANCOVA (Age Adjusted p)
    p_adj = np.nan
    if "AGE" in df_f.columns:
        try:
            temp_df = df_f[[v, "PERIOD", "AGE"]].dropna()
            temp_df.columns = ["Target", "Group", "Age"]
            if len(temp_df) > 20:
                model = smf.ols("Target ~ Group + Age", data=temp_df).fit()
                p_keys = [k for k in model.pvalues.index if "Group" in k]
                if p_keys:
                    p_adj = model.pvalues[p_keys[0]]
        except:
            p_adj = np.nan

    # Normallik Bilgisi Stringi
    norm_info = "Normal" if is_normal_dist else "Not Normal"
    if not is_normal_dist:
        # Hangi grup bozuyorsa onu belirtelim (Opsiyonel detay)
        if status_pre != "Normal": norm_info += " (Pre)"
        if status_post != "Normal": norm_info += " (Post)"

    rows.append({
        "Variable": v,
        "Pre (Ref)": pre_disp,
        "Post": post_disp,
        "% Change": f"{pct_change:.1f}%",
        "Test Used": test_name,
        "Normality": norm_info,
        "p (Raw)": p_label_detailed(p_raw),      # ARTIK NS YOK
        "p (Age Adj.)": p_label_detailed(p_adj), # ARTIK NS YOK
        "Cliff's δ": f"{delta:.3f}",
        "n (Pre/Post)": f"{safe_n(pre_vals)} / {safe_n(post_vals)}"
    })
    pvals_numeric.append(p_raw if np.isfinite(p_raw) else np.nan)

progress_bar.empty()
summary = pd.DataFrame(rows)

if not summary.empty:
    q_arr = pd.Series(pvals_numeric).dropna().values
    # Basit FDR (Benjamini-Hochberg) manuel implementasyon gerekebilir veya library
    # Burada basitlik için sadece ham veriyi gösteriyoruz, FDR istenirse eklenebilir.
    
    st.dataframe(summary, use_container_width=True)
    st.download_button("Tabloyu İndir (CSV)", summary.to_csv(index=False).encode("utf-8-sig"), "nhanes_detailed_stats.csv", "text/csv")

# ---------------------------
# 3) Grafikler
# ---------------------------
st.subheader("3) Grafikler")
plot_vars = st.multiselect("Grafik Seç", vars_to_analyze, vars_to_analyze[:min(4, len(vars_to_analyze))])

if plot_vars:
    n = len(plot_vars)
    cols_grid = int(np.ceil(np.sqrt(n)))
    rows_grid = int(np.ceil(n / cols_grid))
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 4.2, rows_grid * 4.2))
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(plot_vars):
        ax = axes[i]
        pre_vals = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
        post_vals = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()
        
        # Testi tekrar et (grafiğe basmak için)
        p_sw_pre, s1 = check_normality(pre_vals)
        p_sw_post, s2 = check_normality(post_vals)
        is_norm = (s1=="Normal" and s2=="Normal")
        
        use_para_plot = force_parametric or is_norm
        
        if use_para_plot:
            _, p_g = ttest_ind(pre_vals, post_vals, equal_var=False)
        else:
            _, p_g = mannwhitneyu(pre_vals, post_vals)

        # Başlık ve Çizgi Tipi
        title_txt = v + (" (T-test)" if use_para_plot else " (MWU)")
        stripplot_with_p(ax, [pre_vals, post_vals], ["Pre", "Post"], 
                         p_label_detailed(p_g), # Detaylı p etiketi
                         title=title_txt, ylabel=v, point_size=point_size,
                         show_mean=use_para_plot) # Parametrikse Ortalamayı çiz

    for j in range(i + 1, len(axes)): axes[j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi_out), bbox_inches="tight")
    buf.seek(0)
    st.download_button(f"Grafikleri İndir ({dpi_out} DPI)", buf, "nhanes_plots.png", "image/png")
