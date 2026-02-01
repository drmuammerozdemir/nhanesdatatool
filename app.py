# pages/2_Analysis.py
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.formula.api as smf
import seaborn as sns

st.set_page_config(page_title="NHANES Analysis", layout="wide")
st.title("NHANES Pre vs Post – Final Analiz")

# ---------------------------
# 1. HARİTALAMA
# ---------------------------
RENAME_MAP = {
    # --- KİMLİK & AĞIRLIK ---
    "SEQN": "ID", "WTPH2YR": "WEIGHT_LAB",
    # --- DEMOGRAFİ ---
    "RIDAGEYR": "AGE", "RIAGENDR": "SEX", "RIDRETH1": "RACE",
    "INDFMPIR": "PIR", "PERIOD": "PERIOD",
    # --- VÜCUT ÖLÇÜMLERİ ---
    "BMXWT": "WEIGHT_KG", "BMXHT": "HEIGHT_CM", "BMXBMI": "BMI",
    "BMXWAIST": "WAIST_CM", "BMXHIP": "HIP_CM",
    # --- SİGARA DEĞİŞKENLERİ ---
    "SMQ020": "SMOKE_LIFE_100",   
    "SMQ040": "SMOKE_NOW",        
    "SMD030": "AGE_STARTED",      
    "SMD650": "CIGS_PER_DAY_NOW", 
    "SMD057": "CIGS_PER_DAY_QUIT",
    "SMQ050Q": "TIME_SINCE_QUIT", 
    "SMQ050U": "UNIT_SINCE_QUIT", 
    "SMD630": "AGE_FIRST_CIG",    
    # --- LAB DEĞERLERİ ---
    "LBXWBCSI": "WBC", "LBXLYPCT": "LYMPH_PCT", "LBXMOPCT": "MONO_PCT",
    "LBXNEPCT": "NEUT_PCT", "LBXEOPCT": "EOS_PCT", "LBXBAPCT": "BASO_PCT",
    "LBDLYMNO": "LYMPH_ABS", "LBDMONO": "MONO_ABS", "LBDNENO": "NEUT_ABS",
    "LBDEONO": "EOS_ABS", "LBDBANO": "BASO_ABS",
    "LBXRBCSI": "RBC", "LBXHGB": "HGB", "LBXHCT": "HCT", "LBXMCVSI": "MCV",
    "LBXMC": "MCHC", "LBXMCHSI": "MCH", "LBXRDW": "RDW", "LBXPLTSI": "PLT",
    "LBXMPSI": "MPV", "LBXNRBC": "NRBC", "LBXCRP": "CRP",
}

# ---------------------------
# HELPERS
# ---------------------------
def robust_read_csv(uploaded_file):
    for enc in ["utf-8-sig", "utf-8", "cp1254", "latin1"]:
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            uploaded_file.seek(0)
            continue
    raise ValueError("CSV okunamadı.")

def ensure_upper_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def p_label_detailed(p):
    if not np.isfinite(p): return "NA"
    if p < 0.001: return "<0.001"
    if p < 0.01: return "<0.01"
    if p < 0.05: return "<0.05"
    return f"{p:.3f}"

def check_normality(data):
    try:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if len(data) < 3: return np.nan
        _, p = shapiro(data)
        return p
    except: return np.nan

def mean_sd(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, np.nan
    return float(s.mean()), float(s.std())

def median_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, np.nan, np.nan
    return float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))

def format_val_disp(val, q1_sd, q3_sd, is_parametric):
    if np.isfinite(val):
        if is_parametric: return f"{val:.2f} ± {q1_sd:.2f}" 
        else: return f"{val:.3g} [{q1_sd:.3g}–{q3_sd:.3g}]"
    return "NA"

# --- CLIFF'S DELTA ---
def cliffs_delta(x, y):
    """Effect size calculation for non-parametric data."""
    x = pd.to_numeric(x, errors='coerce').dropna().values
    y = pd.to_numeric(y, errors='coerce').dropna().values
    if len(x) == 0 or len(y) == 0: return np.nan
    
    if len(x) * len(y) > 1_000_000:
        np.random.seed(42)
        x = np.random.choice(x, min(len(x), 1000), replace=False)
        y = np.random.choice(y, min(len(y), 1000), replace=False)
        
    m, n = len(x), len(y)
    count = 0
    for i in x:
        count += np.sum(i > y) - np.sum(i < y)
    return count / (m * n)

# ---------------------------
# HESAPLAMA MOTORU
# ---------------------------
def compute_indices(df):
    out = df.copy()
    
    # 1. Sayısal Dönüşüm
    cols_to_numeric = [
        "NEUT_ABS", "LYMPH_ABS", "MONO_ABS", "PLT", "WBC", "CRP",
        "AGE", "AGE_STARTED", "AGE_FIRST_CIG", "CIGS_PER_DAY_NOW", "SMOKE_LIFE_100", "SMOKE_NOW"
    ]
    for c in cols_to_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 2. İndeksler
    if "NEUT_ABS" in out.columns and "LYMPH_ABS" in out.columns:
        out["NLR"] = out["NEUT_ABS"] / out["LYMPH_ABS"]
        if "PLT" in out.columns:
            out["SII"] = (out["PLT"] * out["NEUT_ABS"]) / out["LYMPH_ABS"]
            out["PLR"] = out["PLT"] / out["LYMPH_ABS"]
        if "MONO_ABS" in out.columns:
            out["SIRI"] = (out["NEUT_ABS"] * out["MONO_ABS"]) / out["LYMPH_ABS"]
            if "PLT" in out.columns:
                out["AISI"] = (out["NEUT_ABS"] * out["PLT"] * out["MONO_ABS"]) / out["LYMPH_ABS"]

    # 3. SİGARA (3 KATEGORİ)
    out["SMOKING_STATUS"] = np.nan
    if "SMOKE_LIFE_100" in out.columns and "SMOKE_NOW" in out.columns:
        s100 = out["SMOKE_LIFE_100"]
        snow = out["SMOKE_NOW"]
        out.loc[s100 == 2, "SMOKING_STATUS"] = "Never Smoker"
        out.loc[(s100 == 1) & (snow == 3), "SMOKING_STATUS"] = "Former Smoker"
        out.loc[(s100 == 1) & (snow.isin([1, 2])), "SMOKING_STATUS"] = "Current Smoker"

    # 4. PACK-YEARS
    out["PACK_YEARS"] = np.nan 
    out.loc[out["SMOKING_STATUS"] == "Never Smoker", "PACK_YEARS"] = 0.0
    
    if "AGE" in out.columns and "CIGS_PER_DAY_NOW" in out.columns:
        age = out["AGE"]
        cigs = out["CIGS_PER_DAY_NOW"].replace({999: np.nan, 777: np.nan})
        
        # Başlama Yaşı (Yedekli)
        start = pd.Series(np.nan, index=out.index)
        if "AGE_STARTED" in out.columns: start = start.fillna(out["AGE_STARTED"])
        if "AGE_FIRST_CIG" in out.columns: start = start.fillna(out["AGE_FIRST_CIG"])
        start = start.replace({999: np.nan, 777: np.nan})
        
        years = (age - start).clip(lower=0)
        py = (cigs / 20) * years
        
        mask = (out["SMOKING_STATUS"] == "Current Smoker")
        out.loc[mask, "PACK_YEARS"] = py[mask]

    return out

# ---------------------------
# UI: Yükleme
# ---------------------------
col1, col2 = st.columns(2)
pre_file = col1.file_uploader("Pre CSV", key="pre")
post_file = col2.file_uploader("Post CSV", key="post")

if not pre_file or not post_file:
    st.info("Dosyaları yükleyin.")
    st.stop()

pre = ensure_upper_cols(robust_read_csv(pre_file)).rename(columns=RENAME_MAP)
post = ensure_upper_cols(robust_read_csv(post_file)).rename(columns=RENAME_MAP)

pre["PERIOD"] = "Pre"
post["PERIOD"] = "Post"

pre = compute_indices(pre)
post = compute_indices(post)
df = pd.concat([pre, post], ignore_index=True)

# ---------------------------
# Sidebar: Filtreler ve Seçimler
# ---------------------------
st.sidebar.title("Ayarlar")
page = st.sidebar.radio("Sayfa:", ["1. Özet Tablo", "2. Grafikler", "3. Korelasyon", "4. Regresyon"])

st.sidebar.markdown("---")
if st.sidebar.checkbox("Sadece SII verisi olanlar", True):
    df_f = df.dropna(subset=["SII"]) if "SII" in df.columns else df
else:
    df_f = df

gender_filter = st.sidebar.radio("Cinsiyet:", ["Tümü", "Kadın (2)", "Erkek (1)"])
if gender_filter == "Kadın (2)": df_f = df_f[df_f["SEX"] == 2]
if gender_filter == "Erkek (1)": df_f = df_f[df_f["SEX"] == 1]

# Değişken Seçimi
st.sidebar.markdown("---")
st.sidebar.subheader("Değişkenler")
default_vars = ["SII", "NLR", "PLR", "CRP", "WBC", "AGE", "BMI", "SMOKING_STATUS", "PACK_YEARS"]
avail_vars = [c for c in default_vars if c in df_f.columns]
all_cols = sorted(list(df_f.columns))

# BURADA SEÇİLENLER 'vars_to_analyze' İÇİNE GİRER
vars_to_analyze = st.sidebar.multiselect("Seçiniz:", all_cols, default=avail_vars)

# --- DÜZELTİLEN KISIM BAŞLANGICI ---
st.sidebar.info("👇 Sayısal görünüp Kategorik olanları seç (Ki-Kare için)")
default_cats = ["SEX", "RACE", "SMOKING_STATUS"]
# Sadece şu an seçili olan değişkenlerin içinden varsayılanları belirle!
valid_cat_defaults = [c for c in default_cats if c in vars_to_analyze]

forced_cat_vars = st.sidebar.multiselect("Kategorik Zorlama", vars_to_analyze, default=valid_cat_defaults)
# --- DÜZELTİLEN KISIM BİTİŞİ ---

force_parametric = st.sidebar.checkbox("Parametrik Zorla (T-Test)", False)

pre_f = df_f[df_f["PERIOD"]=="Pre"]
post_f = df_f[df_f["PERIOD"]=="Post"]

# =========================================================
# SAYFA 1: ÖZET TABLO
# =========================================================
if page == "1. Özet Tablo":
    st.header("1. Özet İstatistikler")
    
    rows = []
    posthoc_results = {}
    
    for v in vars_to_analyze:
        pre_d = pre_f[v].dropna()
        post_d = post_f[v].dropna()
        
        if len(pre_d) < 2 or len(post_d) < 2: continue

        is_categorical = (v in forced_cat_vars) or (df_f[v].dtype == 'object')
        
        if is_categorical:
            ct = pd.crosstab(df_f[v], df_f["PERIOD"])
            if "Pre" in ct.columns and "Post" in ct.columns:
                chi2, p, _, _ = chi2_contingency(ct)
                
                def fmt(s):
                    tot = s.sum()
                    return " / ".join([f"**{i}** n:{n} (%{n/tot*100:.1f})" for i,n in s.items()])
                
                rows.append({
                    "Variable": v,
                    "Pre (Ref)": fmt(ct["Pre"]),
                    "Post": fmt(ct["Post"]),
                    "P-Value": p_label_detailed(p),
                    "Cliff's Delta": "—", 
                    "Test": "Chi-Square"
                })

                if ct.shape[0] > 2:
                    ph_rows = []
                    tot_pre = ct["Pre"].sum()
                    tot_post = ct["Post"].sum()
                    for cat in ct.index:
                        n1, n2 = ct.loc[cat, "Pre"], ct.loc[cat, "Post"]
                        r1, r2 = tot_pre - n1, tot_post - n2
                        _, p_sub, _, _ = chi2_contingency([[n1, n2], [r1, r2]])
                        
                        pc1, pc2 = (n1/tot_pre)*100, (n2/tot_post)*100
                        direction = "⬆ Artış" if pc2 > pc1 else "⬇ Azalış"
                        if abs(pc2 - pc1) < 0.1: direction = "↔ Sabit"
                        
                        ph_rows.append({
                            "Alt Grup": cat, "Pre %": f"%{pc1:.1f}", "Post %": f"%{pc2:.1f}",
                            "Yön": direction, "P-Değeri": p_label_detailed(p_sub),
                            "Anlamlı": "⭐" if p_sub < 0.05 else ""
                        })
                    posthoc_results[v] = pd.DataFrame(ph_rows)
        
        else:
            pre_vals = pd.to_numeric(pre_d, errors="coerce")
            post_vals = pd.to_numeric(post_d, errors="coerce")
            
            p_norm = check_normality(pre_vals)
            is_norm = p_norm > 0.05 if np.isfinite(p_norm) else False
            use_para = force_parametric or is_norm
            
            delta_val = cliffs_delta(pre_vals, post_vals)
            delta_str = f"{delta_val:.2f}" if np.isfinite(delta_val) else "NA"
            
            if use_para:
                stat, p = ttest_ind(pre_vals, post_vals, equal_var=False)
                d_pre = format_val_disp(*mean_sd(pre_vals), True)
                d_post = format_val_disp(*mean_sd(post_vals), True)
                test = "Welch T"
            else:
                stat, p = mannwhitneyu(pre_vals, post_vals)
                d_pre = format_val_disp(*median_iqr(pre_vals), False)
                d_post = format_val_disp(*median_iqr(post_vals), False)
                test = "MWU"
                
            rows.append({
                "Variable": v, "Pre (Ref)": d_pre, "Post": d_post,
                "P-Value": p_label_detailed(p), 
                "Cliff's Delta": delta_str,
                "Test": test
            })
            
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Veri yok.")

    if posthoc_results:
        st.markdown("---")
        st.subheader("🔍 Kategorik Alt Analiz (Post-Hoc)")
        for k, v in posthoc_results.items():
            st.markdown(f"**{k}**")
            st.table(v)

# =========================================================
# SAYFA 2: GRAFİKLER
# =========================================================
elif page == "2. Grafikler":
    st.header("2. Grafikler")
    plot_vars = st.multiselect("Çizilecekler:", vars_to_analyze, default=vars_to_analyze[:min(4, len(vars_to_analyze))])
    
    if plot_vars:
        cols = 2
        rows = int(np.ceil(len(plot_vars)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4*rows))
        axes = axes.flatten() if len(plot_vars) > 1 else [axes]
        
        for i, v in enumerate(plot_vars):
            is_categorical = (v in forced_cat_vars) or (df_f[v].dtype == 'object')
            if is_categorical:
                counts = df_f.groupby(["PERIOD", v]).size().reset_index(name="Count")
                sns.barplot(data=counts, x="PERIOD", y="Count", hue=v, ax=axes[i])
            else:
                sns.boxplot(data=df_f, x="PERIOD", y=v, ax=axes[i], palette="Set2")
            axes[i].set_title(v)
            
        plt.tight_layout()
        st.pyplot(fig)

# =========================================================
# SAYFA 3: KORELASYON
# =========================================================
elif page == "3. Korelasyon":
    st.header("3. Korelasyon")
    num_cols = [c for c in vars_to_analyze if df_f[c].dtype != 'object' and c not in forced_cat_vars]
    if len(num_cols) > 1:
        corr = df_f[num_cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# =========================================================
# SAYFA 4: REGRESYON
# =========================================================
elif page == "4. Regresyon":
    st.header("4. Regresyon")
    target = st.selectbox("Y", ["SII", "NLR", "PLR"], 0)
    covars = st.multiselect("X", ["PERIOD", "AGE", "SEX", "BMI", "SMOKING_STATUS"], default=["PERIOD", "AGE"])
    
    if st.button("Kur"):
        try:
            f = f"{target} ~ " + " + ".join(covars)
            model = smf.ols(f, data=df_f.dropna(subset=[target]+covars)).fit()
            st.code(model.summary().as_text())
        except Exception as e:
            st.error(str(e))
