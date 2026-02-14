# pages/2_Analysis.py
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency, pearsonr, spearmanr, kendalltau
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
    "RIDAGEYR": "AGE", "RIAGENDR": "SEX", "RIDRETH3": "RACE",
    "INDFMPIR": "PIR", "PERIOD": "PERIOD",
    "RIDEXMON": "SEASON_CODE",  # Mevsim
    "RIDEXPRG": "PREGNANCY",    # Hamilelik
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
    
    # 0. TANSİYON VE NABIZ ORTALAMALARI
    bp_cols = ['BPXOSY2', 'BPXOSY3', 'BPXODI2', 'BPXODI3', 'BPXOPLS2', 'BPXOPLS3']
    for col in bp_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    if 'BPXOSY2' in out.columns and 'BPXOSY3' in out.columns:
        out['SYSTOLICBP'] = out[['BPXOSY2', 'BPXOSY3']].mean(axis=1, skipna=True)
    
    if 'BPXODI2' in out.columns and 'BPXODI3' in out.columns:
        out['DIASTOLICBP'] = out[['BPXODI2', 'BPXODI3']].mean(axis=1, skipna=True)
        
    if 'BPXOPLS2' in out.columns and 'BPXOPLS3' in out.columns:
        out['PULSEAVG'] = out[['BPXOPLS2', 'BPXOPLS3']].mean(axis=1, skipna=True)
    
    # 1. Sayısal Dönüşüm
    cols_to_numeric = [
        "NEUT_ABS", "LYMPH_ABS", "MONO_ABS", "PLT", "WBC", "CRP",
        "AGE", "AGE_STARTED", "AGE_FIRST_CIG", "CIGS_PER_DAY_NOW", "SMOKE_LIFE_100", "SMOKE_NOW", "PREGNANCY"
    ]
    for c in cols_to_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 2. İndeksler
    if "NEUT_ABS" in out.columns and "LYMPH_ABS" in out.columns:
        out["NLR"] = out["NEUT_ABS"] / out["LYMPH_ABS"]
        
        if "WBC" in out.columns:
             denom = out["WBC"] - out["NEUT_ABS"]
             out["dNLR"] = out["NEUT_ABS"] / denom
             
             # --- YENİ EKLENEN: dPLR ---
             # Formül: PLT / (WBC - NEUT)
             # (WBC - NEUT), lenfosit yerine "derived" payda olarak kullanılır.
             if "PLT" in out.columns:
                 out["dPLR"] = out["PLT"] / denom    
        
        if "PLT" in out.columns:
            out["SII"] = (out["PLT"] * out["NEUT_ABS"]) / out["LYMPH_ABS"]
            out["PLR"] = out["PLT"] / out["LYMPH_ABS"]
        
        if "MONO_ABS" in out.columns:
            out["SIRI"] = (out["NEUT_ABS"] * out["MONO_ABS"]) / out["LYMPH_ABS"]
            out["MLR"] = out["MONO_ABS"] / out["LYMPH_ABS"]
            out["NMLR"] = (out["NEUT_ABS"] + out["MONO_ABS"]) / out["LYMPH_ABS"]

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
        
        start = pd.Series(np.nan, index=out.index)
        if "AGE_STARTED" in out.columns: start = start.fillna(out["AGE_STARTED"])
        if "AGE_FIRST_CIG" in out.columns: start = start.fillna(out["AGE_FIRST_CIG"])
        start = start.replace({999: np.nan, 777: np.nan})
        
        years = (age - start).clip(lower=0)
        py = (cigs / 20) * years
        
        mask = (out["SMOKING_STATUS"] == "Current Smoker")
        out.loc[mask, "PACK_YEARS"] = py[mask]
    
    # IRK ETİKETLEME
    if "RACE" in out.columns:
        race_mapping = {
            1: "Mexican American",
            2: "Other Hispanic",
            3: "Non-Hispanic White",
            4: "Non-Hispanic Black",
            6: "Non-Hispanic Asian",
            7: "Other/Multi-Racial"
        }
        if pd.api.types.is_numeric_dtype(out["RACE"]):
             out["RACE"] = out["RACE"].map(race_mapping)

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
# Sidebar: Filtreler ve Seçimler (ADVANCED ENGLISH FLOWCHART)
# ---------------------------
st.sidebar.title("Settings & Flow")
page = st.sidebar.radio("Page:", ["1. Summary Table", "2. Plots", "3. Correlation", "4. Regression"])

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Exclusion Criteria")

# --- FLOWCHART İÇİN YARDIMCI FONKSİYON ---
def get_stats_str(d):
    """Verilen dataframe için N, Pre/Post ve Male/Female sayılarını döndürür."""
    n = len(d)
    if n == 0: return "N=0"
    
    # Pre/Post
    n_pre = len(d[d["PERIOD"] == "Pre"])
    n_post = len(d[d["PERIOD"] == "Post"])
    
    # Male/Female (1:Male, 2:Female)
    n_male = len(d[d["SEX"] == 1])
    n_female = len(d[d["SEX"] == 2])
    
    return f"N = {n}\\n(Pre: {n_pre}, Post: {n_post})\\n(Male: {n_male}, Female: {n_female})"

# --- BAŞLANGIÇ VERİSİ ---
df_f = df.copy()
initial_stats = get_stats_str(df_f)

# Graphviz Başlangıcı
steps = []
node_counter = 0

# --- 0. HAMİLELİK FİLTRESİ ---
if "PREGNANCY" in df_f.columns:
    if st.sidebar.checkbox("Exclude Pregnancy", value=True):
        n_before = len(df_f)
        df_f = df_f[df_f["PREGNANCY"] != 1] # 1=Yes, dışla
        loss = n_before - len(df_f)
        
        if loss > 0:
            node_counter += 1
            steps.append({
                "label": "Excluded (Pregnancy)",
                "loss_n": loss,
                "stats": get_stats_str(df_f),
                "id": node_counter
            })

# --- 1. MEVSİM FİLTRESİ ---
season_loss = 0
n_before = len(df_f)
if "SEASON_CODE" in df_f.columns:
    st.sidebar.subheader("📅 Seasonality")
    season_choice = st.sidebar.radio("Select Season:", ["All", "Winter (1)", "Summer (2)"], index=0)
    
    if season_choice != "All":
        target_code = 1 if "Winter" in season_choice else 2
        df_f = df_f[df_f["SEASON_CODE"] == target_code]
        loss = n_before - len(df_f)
        
        if loss > 0:
            node_counter += 1
            steps.append({
                "label": f"Excluded (Season: {season_choice})",
                "loss_n": loss,
                "stats": get_stats_str(df_f),
                "id": node_counter
            })

# --- 2. YAŞ FİLTRESİ (18+) ---
st.sidebar.subheader("🔞 Age Group")
only_adults = st.sidebar.checkbox("Adults Only (18+)", value=True)

if only_adults and "AGE" in df_f.columns:
    n_before = len(df_f)
    df_f = df_f[df_f["AGE"] >= 18]
    loss = n_before - len(df_f)
    
    if loss > 0:
        node_counter += 1
        steps.append({
            "label": "Excluded (Age < 18)",
            "loss_n": loss,
            "stats": get_stats_str(df_f),
            "id": node_counter
        })

# --- 3. EKSİK VERİ FİLTRESİ ---
sii_filter = st.sidebar.checkbox("Exclude Missing CBC Data", value=True)

if sii_filter and "SII" in df_f.columns:
    n_before = len(df_f)
    df_f = df_f.dropna(subset=["SII"])
    loss = n_before - len(df_f)
    
    if loss > 0:
        node_counter += 1
        steps.append({
            "label": "Excluded (Missing Data: CBC)",
            "loss_n": loss,
            "stats": get_stats_str(df_f),
            "id": node_counter
        })
# =========================================================
# YENİ EKLENTİ: BMI ve SMOKING EKSİK VERİ FİLTRELERİ
# =========================================================

# --- 3.1. BMI EKSİK VERİ FİLTRESİ ---
if "BMI" in df_f.columns:
    # Default True yaptık ki regresyonla sayı tutsun
    if st.sidebar.checkbox("Exclude Missing Data: BMI", value=True):
        n_before = len(df_f)
        df_f = df_f.dropna(subset=["BMI"])
        loss = n_before - len(df_f)
        
        if loss > 0:
            node_counter += 1
            steps.append({
                "label": "Excluded (Missing Data: BMI)",
                "loss_n": loss,
                "stats": get_stats_str(df_f),
                "id": node_counter
            })

# --- 3.2. SMOKING EKSİK VERİ FİLTRESİ ---
if "SMOKING_STATUS" in df_f.columns:
    # Default True yaptık
    if st.sidebar.checkbox("Exclude Missing Data: Smoking", value=True):
        n_before = len(df_f)
        df_f = df_f.dropna(subset=["SMOKING_STATUS"])
        loss = n_before - len(df_f)
        
        if loss > 0:
            node_counter += 1
            steps.append({
                "label": "Excluded (Missing Data: Smoking)",
                "loss_n": loss,
                "stats": get_stats_str(df_f),
                "id": node_counter
            })
# --- 4. CRP FİLTRESİ ---
if "CRP" in df_f.columns:
    if st.sidebar.checkbox("Exclude Active Infection (CRP > 10)", value=False):
        n_before = len(df_f)
        df_f = df_f[df_f["CRP"] <= 10]
        loss = n_before - len(df_f)
        
        if loss > 0:
            node_counter += 1
            steps.append({
                "label": "Excluded (Infection/CRP>10)",
                "loss_n": loss,
                "stats": get_stats_str(df_f),
                "id": node_counter
            })

# --- 5. CİNSİYET FİLTRESİ ---
st.sidebar.subheader("Gender")
gender_filter = st.sidebar.radio("Select Gender:", ["All", "Female (2)", "Male (1)"])

if gender_filter != "All":
    n_before = len(df_f)
    target_sex = 2 if "Female" in gender_filter else 1
    df_f = df_f[df_f["SEX"] == target_sex]
    loss = n_before - len(df_f)
    
    if loss > 0:
        node_counter += 1
        steps.append({
            "label": f"Excluded ({'Male' if target_sex==2 else 'Female'})",
            "loss_n": loss,
            "stats": get_stats_str(df_f),
            "id": node_counter
        })

# --- GRAPHVIZ STRİNG OLUŞTURMA (İNGİLİZCE) ---
final_dot = f"""
digraph Flow {{
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=9];

    // Node 0: Total Data
    node0 [label="Total Data\\n{initial_stats}", fillcolor="#E1F5FE", color="#0277BD", penwidth=1.5];
"""

prev_node = "node0"

for step in steps:
    curr_id = step['id']
    loss_node = f"loss{curr_id}"
    main_node = f"node{curr_id}"
    
    # 1. Kayıp Düğümü (Kırmızı Sekizgen)
    final_dot += f'{loss_node} [label="{step["label"]}\\nn = {step["loss_n"]}", shape=octagon, fillcolor="#FFCDD2", color="#C62828"];\n'
    
    # 2. Kalan Düğümü (Mavi)
    final_dot += f'{main_node} [label="Remaining\\n{step["stats"]}", fillcolor="#E1F5FE", color="#0277BD"];\n'
    
    # 3. Bağlantılar
    final_dot += f'{prev_node} -> {loss_node} [style=dashed, color="red", arrowsize=0.8];\n'
    final_dot += f'{prev_node} -> {main_node} [color="black", arrowsize=0.8];\n'
    
    prev_node = main_node

# Son Düğümü Yeşile Boya (Final Data)
final_dot += f'{prev_node} [label="Final Analysis Data\\n{get_stats_str(df_f)}", fillcolor="#C8E6C9", color="#2E7D32", penwidth=2.5];\n'
final_dot += "}"

# --- FLOWCHART GÖSTERİMİ ---
st.sidebar.markdown("---")
with st.sidebar.expander("📊 Study Flowchart", expanded=True):
    st.graphviz_chart(final_dot)
    
    if len(df_f) == 0:
        st.error("⚠️ No data left! Please relax the filters.")

# =========================================================
# !!! KRİTİK KISIM !!!
# FLOWCHART'TAN SONRA, ANALİZ İÇİN GEREKLİ DEĞİŞKENLERİ 
# TEKRAR TANIMLAMAMIZ GEREKİYOR (SİLİNEN KISIM BURASIYDI)
# =========================================================

# Değişken Seçimi
st.sidebar.markdown("---")
st.sidebar.subheader("Variables")
default_vars = ["SII", "NLR", "dNLR", "PLR", "dPLR", "MLR", "NMLR", "SYSTOLICBP", "DIASTOLICBP", "PULSEAVG", "CRP", "WBC", "AGE", "BMI", "WAIST_CM", "SMOKING_STATUS"]
avail_vars = [c for c in default_vars if c in df_f.columns]
all_cols = sorted(list(df_f.columns))

vars_to_analyze = st.sidebar.multiselect("Select Variables to Analyze:", all_cols, default=avail_vars)

st.sidebar.info("👇 Select categorical vars (for Chi-Square)")
default_cats = ["SEX", "RACE", "SMOKING_STATUS"]
valid_cat_defaults = [c for c in default_cats if c in vars_to_analyze]

forced_cat_vars = st.sidebar.multiselect("Force Categorical", vars_to_analyze, default=valid_cat_defaults)

force_parametric = st.sidebar.checkbox("Force Parametric (T-Test)", False)

# --- YENİ EKLENEN: TABLO FORMAT AYARI (SIDEBAR) ---
st.sidebar.markdown("---")
non_param_style = st.sidebar.radio(
    "Non-Parametric Format:",
    ["Median [IQR] (Q1–Q3)", "Median [Min–Max]"],
    index=0,
    help="Tabloda parametrik olmayan verilerin (Medyan) yanındaki parantez içi değerin formatını seçin."
)

pre_f = df_f[df_f["PERIOD"]=="Pre"]
post_f = df_f[df_f["PERIOD"]=="Post"]

# =========================================================
# SAYFA 1: ÖZET TABLO (SİMGELİ ETKİ BÜYÜKLÜKLERİ: d, e)
# =========================================================
if page == "1. Summary Table":
    st.header("1. Summary Statistics")
    
    rows = []
    posthoc_results = {}
    
    # --- TEST VE ETKİ SİMGELERİ ---
    SYM_T = "ᵃ"       # T-Test
    SYM_MWU = "ᵇ"     # Mann-Whitney U
    SYM_CHI = "ᶜ"     # Chi-Square
    SYM_DELTA = "ᵈ"   # Cliff's Delta (Sayısal)
    SYM_V = "ᵉ"       # Cramer's V (Kategorik - User isteği 'e')

    # --- YARDIMCI FONKSİYONLAR ---
    def fmt_non_param(series, style_choice):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return "NA"
        med = s.median()
        
        if "Min" in style_choice:
            # Min-Max Formatı
            low = s.min()
            high = s.max()
        else:
            # IQR Formatı (Varsayılan)
            low = s.quantile(0.25)
            high = s.quantile(0.75)
        
        return f"{med:.3g} [{low:.3g}–{high:.3g}]"

    def fmt_mean_sd(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return "NA"
        m = s.mean()
        sd = s.std()
        return f"{m:.2f} ± {sd:.2f}"

    def calculate_cramers_v(confusion_matrix):
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        if min((kcorr-1), (rcorr-1)) == 0: return 0.0
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    for v in vars_to_analyze:
        pre_d = pre_f[v].dropna()
        post_d = post_f[v].dropna()
        
        if len(pre_d) < 2 or len(post_d) < 2: continue

        is_categorical = (v in forced_cat_vars) or (df_f[v].dtype == 'object')
        
        # ---------------------------------------------------------
        # A) KATEGORİK (Cramer's V -> 'e')
        # ---------------------------------------------------------
        if is_categorical:
            ct = pd.crosstab(df_f[v], df_f["PERIOD"])
            if "Pre" in ct.columns and "Post" in ct.columns:
                chi2, p_overall, _, _ = chi2_contingency(ct)
                cramer_v = calculate_cramers_v(ct)
                
                # Simge 'e' eklendi
                effect_str = f"{cramer_v:.2f} {SYM_V}" 

                rows.append({
                    "Variable": v, 
                    "Pre (Ref)": f"N={ct['Pre'].sum()}", 
                    "Post": f"N={ct['Post'].sum()}", 
                    "P-Value": f"{p_label_detailed(p_overall)} {SYM_CHI}", 
                    "Effect Size": effect_str 
                })

                # Post-Hoc
                if ct.shape[0] > 2:
                    ph_rows = []
                    tot_pre = ct["Pre"].sum()
                    tot_post = ct["Post"].sum()
                    for cat in ct.index:
                        n1, n2 = ct.loc[cat, "Pre"], ct.loc[cat, "Post"]
                        r1, r2 = tot_pre - n1, tot_post - n2
                        sub_ct = np.array([[n1, n2], [r1, r2]])
                        _, p_sub, _, _ = chi2_contingency(sub_ct)
                        sub_cramer = calculate_cramers_v(pd.DataFrame(sub_ct))
                        
                        pc1 = (n1/tot_pre)*100
                        pc2 = (n2/tot_post)*100
                        
                        ph_rows.append({
                            "Subgroup": cat,
                            "Pre (Ref)": f"{n1} ({pc1:.1f}%)",
                            "Post": f"{n2} ({pc2:.1f}%)",
                            "P-Value": f"{p_label_detailed(p_sub)} {SYM_CHI}",
                            "Cramer's V": f"{sub_cramer:.2f}"
                        })
                    posthoc_results[v] = pd.DataFrame(ph_rows)

        # ---------------------------------------------------------
        # B) SAYISAL (Cliff's Delta -> 'd')
        # ---------------------------------------------------------
        else:
            pre_vals = pd.to_numeric(pre_d, errors="coerce")
            post_vals = pd.to_numeric(post_d, errors="coerce")
            
            p_norm = check_normality(pre_vals)
            is_norm = p_norm > 0.05 if np.isfinite(p_norm) else False
            use_para = force_parametric or is_norm
            
            delta_val = cliffs_delta(pre_vals, post_vals)
            val_str = f"{delta_val:.2f}" if np.isfinite(delta_val) else "NA"
            
            # Simge 'd' eklendi
            delta_str = f"{val_str} {SYM_DELTA}"
            
            if use_para:
                _, p = ttest_ind(pre_vals, post_vals, equal_var=False)
                d_pre = fmt_mean_sd(pre_vals)
                d_post = fmt_mean_sd(post_vals)
                test_sym = SYM_T
            else:
                _, p = mannwhitneyu(pre_vals, post_vals)
                # Yeni fonksiyonu sidebar'daki seçimle çağırıyoruz
                d_pre = fmt_non_param(pre_vals, non_param_style)
                d_post = fmt_non_param(post_vals, non_param_style)
                test_sym = SYM_MWU
                
            rows.append({
                "Variable": v, 
                "Pre (Ref)": d_pre, 
                "Post": d_post,
                "P-Value": f"{p_label_detailed(p)} {test_sym}",
                "Effect Size": delta_str
            })
            
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.caption(f"""
        **Dipnotlar:**
        * **İstatistiksel Testler:** {SYM_T}: Welch T-Test, {SYM_MWU}: Mann-Whitney U, {SYM_CHI}: Ki-Kare.
        * **Etki Büyüklüğü (Effect Size):** {SYM_DELTA}: Cliff's Delta (Sayısal Veriler), {SYM_V}: Cramer's V (Kategorik Veriler).
        """)
    else:
        st.warning("Veri yok.")

    if posthoc_results:
        st.markdown("---")
        st.subheader("🔍 Kategorik Alt Analiz (Post-Hoc)")
        for k, v in posthoc_results.items():
            st.markdown(f"**{k} Kırılımı**")
            st.dataframe(v, use_container_width=True, hide_index=True)

    # ==============================================================================
    # ⬇️ SAYISAL ALT GRUP ANALİZİ (MEVCUT KODUNUZDA VARDI, KORUYORUZ) ⬇️
    # ==============================================================================
    st.markdown("---")
    st.subheader("🔬 Alt Grup Analizi (Subgroup Analysis)")
    st.markdown("Bir sayısal değişkenin (Örn: **SII**), belirli bir kategoriye (Örn: **RACE**) göre değişimini inceleyin.")

    numeric_opts = [c for c in vars_to_analyze if pd.api.types.is_numeric_dtype(df_f[c])]
    category_opts = [c for c in all_cols if (c in forced_cat_vars) or (df_f[c].dtype == 'object')]
    
    if numeric_opts and category_opts:
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            target_var = st.selectbox("1. Sayısal Değişkeni Seç (Target):", numeric_opts, index=0)
        with col_sub2:
            def_idx = category_opts.index("RACE") if "RACE" in category_opts else 0
            group_var = st.selectbox("2. Kime Göre Bakılsın? (Subgroup):", category_opts, index=def_idx)

        subgroup_rows = []
        # Mann-Whitney U simgesi
        SYM_MWU = "ᵇ"
        
        def fmt_median_iqr_sub(series):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty: return "NA"
            med = s.median()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            return f"{med:.3g} [{q1:.3g}–{q3:.3g}]"
        
        unique_groups = sorted(df_f[group_var].dropna().unique())
        
        for grp in unique_groups:
            sub_df = df_f[df_f[group_var] == grp]
            pre_d = sub_df[sub_df["PERIOD"] == "Pre"][target_var].dropna()
            post_d = sub_df[sub_df["PERIOD"] == "Post"][target_var].dropna()
            
            if len(pre_d) < 2 or len(post_d) < 2: continue
            
            delta_val = cliffs_delta(pre_d, post_d)
            delta_str = f"{delta_val:.2f}" if np.isfinite(delta_val) else "NA"
            _, p = mannwhitneyu(pre_d, post_d)
            
            subgroup_rows.append({
                "Subgroup": grp,
                "Pre (Ref)": fmt_median_iqr_sub(pre_d),
                "Post": fmt_median_iqr_sub(post_d),
                "P-Value": f"{p_label_detailed(p)} {SYM_MWU}",
                "Cliff's Delta": delta_str
            })
            
        if subgroup_rows:
            st.markdown(f"**Analiz:** `{target_var}` değişkeninin `{group_var}` alt kırılımları:")
            st.dataframe(pd.DataFrame(subgroup_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Veri yok.")
    else:
        st.info("Alt grup analizi için seçim yapınız.")

# =========================================================
# SAYFA 2: GRAFİKLER (OUTLIER YÖNETİMİ VE LOG SCALE EKLENDİ)
# =========================================================
elif page == "2. Plots":
    st.header("2. Graphs (Publication Ready)")

    # --- 1. DEĞİŞKEN SEÇİMİ ---
    plot_vars = st.multiselect(
        "Select Variables to Plot:", 
        vars_to_analyze, 
        default=vars_to_analyze[:min(4, len(vars_to_analyze))] if len(vars_to_analyze) > 0 else None
    )

    if plot_vars:
        # --- 2. GRAFİK AYARLARI (EXPANDER) ---
        with st.expander("⚙️ Graph Settings", expanded=True):
            tab1, tab2, tab3 = st.tabs(["📊 Data & Axis", "📏 Line Style", "🏷️ Labels"])
            
            with tab1:
                col_data1, col_data2 = st.columns(2)
                with col_data1:
                    st.markdown("##### 🔍 Data View")
                    # Outlier Filtresi
                    remove_outliers = st.checkbox("Hide Outliers (IQR Method)", value=False, help="Grafiği bastıran çok yüksek değerleri (Q3 + 1.5*IQR üzerini) gizler. Medyan farklarını görmek için idealdir.")
                    # --- YENİ EKLENEN: MANUEL ÜST KESİM (SLIDER) ---
                    upper_cut = st.slider(
                        "Upper Cut Limit (%)", 
                        min_value=75.0, 
                        max_value=100.0, 
                        value=100.0, 
                        step=0.5,
                        help="Verinin yüzde kaçını görmek istediğinizi seçin. Örneğin 95'e ayarlarsanız, en yüksek %5'lik uç değerler grafikten atılır (Alt değerlere dokunulmaz)."
                    )
                    # Log Scale
                    use_log = st.checkbox("Logarithmic Scale", value=False, help="Aşırı büyük ve küçük değerleri aynı grafikte dengeli gösterir.")
                    
                    # İstatistik Tipi
                    plot_type = st.radio(
                        "Statistical Summary:", 
                        ["Mean ± SD", "Median ± IQR"],
                        horizontal=True
                    )
                with col_data2:
                    st.markdown("##### 🎨 Appearance")
                    cols_num = st.slider("Columns", 1, 4, 2)
                    dot_size = st.slider("Dot Size", 1, 15, 4)
                    c1 = st.color_picker("Pre Color", "#4c72b0")
                    c2 = st.color_picker("Post Color", "#c44e52")

            with tab2:
                st.markdown("##### Error Bars")
                col_err1, col_err2, col_err3 = st.columns(3)
                with col_err1:
                    err_linewidth = st.slider("Line Width", 0.5, 5.0, 1.5)
                with col_err2:
                    err_capsize = st.slider("Cap Size", 0, 20, 8)
                with col_err3:
                    err_capthick = st.slider("Cap Thickness", 0.5, 5.0, 1.5)

            with tab3:
                st.info("Customize titles if single variable selected.")
                custom_title = st.text_input("Graph Title", value="")
                col_lbl1, col_lbl2 = st.columns(2)
                with col_lbl1: custom_xlabel = st.text_input("X Axis Label", value="Group")
                with col_lbl2: custom_ylabel = st.text_input("Y Axis Label", value=plot_vars[0] if len(plot_vars)==1 else "Value")

        is_parametric_plot = "Mean" in plot_type
        
        # --- 3. ÇİZİM MOTORU ---
        rows_num = int(np.ceil(len(plot_vars) / cols_num))
        fig_width = 5 * cols_num
        fig_height = 5 * rows_num
        fig, axes = plt.subplots(rows_num, cols_num, figsize=(fig_width, fig_height))
        
        if isinstance(axes, np.ndarray): axes = axes.flatten()
        else: axes = [axes]
        
        custom_palette = {"Pre": c1, "Post": c2}

        for i, v in enumerate(plot_vars):
            ax = axes[i]
            
            # --- GÖRSEL VERİ HAZIRLIĞI ---
            plot_data = df_f.copy()
            
            # 1. IQR Filtresi (Varsa)
            if remove_outliers and pd.api.types.is_numeric_dtype(plot_data[v]):
                Q1 = plot_data[v].quantile(0.25)
                Q3 = plot_data[v].quantile(0.75)
                IQR = Q3 - Q1
                upper_limit = Q3 + 1.5 * IQR
                lower_limit = Q1 - 1.5 * IQR
                plot_data = plot_data[(plot_data[v] <= upper_limit) & (plot_data[v] >= lower_limit)]

            # 2. Manuel Üst Kesim (Slider < 100 ise çalışır)
            if upper_cut < 100.0 and pd.api.types.is_numeric_dtype(plot_data[v]):
                # Kullanıcının seçtiği yüzdeliğe (örn: 0.95) denk gelen değeri bul
                limit_high = plot_data[v].quantile(upper_cut / 100.0)
                # Sadece üstten kes, alttan kesme
                plot_data = plot_data[plot_data[v] <= limit_high]
            
            # Kategorik kontrolü
            is_categorical = (v in forced_cat_vars) or (plot_data[v].dtype == 'object')
            
            if is_categorical:
                counts = plot_data.groupby(["PERIOD", v]).size().reset_index(name="Count")
                sns.barplot(data=counts, x="PERIOD", y="Count", hue=v, ax=ax, palette="Set2")
                ax.set_title(f"{v} Distribution")
            else:
                # --- DOT PLOT ---
                sns.stripplot(
                    data=plot_data, x="PERIOD", y=v, 
                    palette=custom_palette, 
                    alpha=0.6,    
                    size=dot_size, 
                    jitter=0.25,   
                    ax=ax,
                    zorder=0      
                )

                # --- İSTATİSTİK ÇİZGİLERİ (MEAN/MEDIAN) ---
                periods = ["Pre", "Post"]
                for j, period in enumerate(periods):
                    subset = plot_data[plot_data["PERIOD"] == period][v].dropna()
                    if len(subset) == 0: continue

                    if is_parametric_plot:
                        center = subset.mean()
                        # %95 Confidence Interval Hesabı (Mean için)
                        # Formül: 1.96 * (Standart Sapma / Karekök(N))
                        sd = subset.std()
                        n = len(subset)
                        sem = sd / np.sqrt(n) # Standart Hata
                        yerr = 1.96 * sem     # %95 Güven Aralığı
                    else:
                        center = subset.median()
                        q1 = subset.quantile(0.25)
                        q3 = subset.quantile(0.75)
                        yerr = [[center - q1], [q3 - center]] 
                    
                    # Error Bar
                    ax.errorbar(
                        x=j, y=center, yerr=yerr, 
                        fmt='none', ecolor='black', 
                        elinewidth=err_linewidth, capsize=err_capsize, capthick=err_capthick, 
                        zorder=5
                    )
                    # Mean/Median Line
                    ax.hlines(
                        y=center, xmin=j-0.2, xmax=j+0.2, 
                        colors='black', linewidth=err_linewidth + 0.5, zorder=6
                    )

                # ==================================================
                # ⬇️ P-DEĞERİ VE BRACKET (TABLO İLE EŞİTLENDİ) ⬇️
                # ==================================================
                vec_pre = plot_data[plot_data["PERIOD"] == "Pre"][v].dropna()
                vec_post = plot_data[plot_data["PERIOD"] == "Post"][v].dropna()

                if len(vec_pre) > 1 and len(vec_post) > 1:
                    # A) Tablodaki mantığın aynısı: Normallik Testi
                    p_norm_check = check_normality(vec_pre)
                    # Eğer test yapılamadıysa (nan) veya p > 0.05 ise Normal kabul edilebilir veya edilemez. 
                    # Burada güvenli yol: p > 0.05 ise Normaldir.
                    is_data_normal = p_norm_check > 0.05 if np.isfinite(p_norm_check) else False
                    
                    # B) Hangi Testi Kullanayım?
                    # force_parametric işaretliyse VEYA veri normalse -> T-Test
                    should_use_parametric = force_parametric or is_data_normal
                    
                    if should_use_parametric:
                        _, p_val_plot = ttest_ind(vec_pre, vec_post, equal_var=False)
                    else:
                        _, p_val_plot = mannwhitneyu(vec_pre, vec_post)

                    # Metin
                    if p_val_plot < 0.001: p_txt = "p < 0.001"
                    else: p_txt = f"p = {p_val_plot:.3f}"

                    # Koordinatlar
                    y_max_data = max(vec_pre.max(), vec_post.max())
                    y_min_data = min(vec_pre.min(), vec_post.min())
                    y_rng = y_max_data - y_min_data if y_max_data != y_min_data else y_max_data * 0.1
                    
                    bracket_h = y_max_data + (y_rng * 0.10) 
                    text_h = bracket_h + (y_rng * 0.02)     
                    tick_len = y_rng * 0.03                  

                    # Çizim
                    ax.plot([0, 0, 1, 1], [bracket_h - tick_len, bracket_h, bracket_h, bracket_h - tick_len], 
                            lw=1.5, c='black')
                    ax.text(0.5, text_h, p_txt, ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
                    
                    # Tavanı Yükselt
                    current_ylim = ax.get_ylim()
                    if text_h > current_ylim[1] or True: 
                         ax.set_ylim(current_ylim[0], text_h + (y_rng * 0.15))

            # --- LOG SCALE AYARI (DÜZELTİLMİŞ: Negatifleri Gizle) ---
            if use_log:
                # symlog kullanıyoruz (0 hatası vermesin diye)
                ax.set_yscale('symlog', linthresh=0.1)
                
                # Bilimsel gösterimi kapat
                from matplotlib.ticker import ScalarFormatter
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
                
                # EKSEN LİMİTLERİ (KRİTİK DÜZELTME)
                # Mevcut limitleri al
                bottom, top = ax.get_ylim()
                
                # Alt limiti kesinlikle 0 yap (Negatifleri at)
                # Üst limiti P değeri yazısı sığsın diye biraz daha açıyoruz (* 2.0)
                ax.set_ylim(0, top * 2.0)

            # --- ETİKETLER ---
            if len(plot_vars) == 1:
                ax.set_title(custom_title if custom_title else v, fontweight="bold", fontsize=14)
                ax.set_xlabel(custom_xlabel, fontsize=12, fontweight="bold")
                ax.set_ylabel(custom_ylabel, fontsize=12, fontweight="bold")
            else:
                ax.set_title(v, fontweight="bold")
                ax.set_xlabel("")

            # Temizlik
            ax.grid(axis='y', linestyle='--', alpha=0.3, which='both')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

        for j in range(i+1, len(axes)): fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)

        # İndirme
        st.markdown("---")
        col_d1, col_d2 = st.columns([3, 1])
        with col_d2:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', transparent=True)
            st.download_button("📥 Grafiği İndir (300 DPI)", buf.getvalue(), "grafik.png", "image/png", use_container_width=True)
    else:
        st.info("Lütfen soldan veya yukarıdan en az bir değişken seçin.")

# =========================================================
# SAYFA 3: KORELASYON (GELİŞMİŞ HEATMAP EDİTÖRÜ)
# =========================================================
elif page == "3. Correlation":
    st.header("3. Correlation Analysis (Heatmap)")

    # 1. Sadece Sayısal Değişkenleri Seç
    num_cols = [c for c in vars_to_analyze if pd.api.types.is_numeric_dtype(df_f[c]) and c not in forced_cat_vars]

    if len(num_cols) < 2:
        st.warning("⚠️ Korelasyon analizi yapabilmek için sol taraftan en az 2 adet sayısal değişken seçmelisiniz.")
    else:
        # --- AYARLAR MENÜSÜ ---
        with st.expander("⚙️ Heatmap Settings", expanded=True):
            tab1, tab2 = st.tabs(["📊 Analysis & Style", "📐 Dimensions"])
            
            with tab1:
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    corr_method = st.selectbox("Method", ["spearman", "pearson", "kendall"], help="Normal dağılım yoksa Spearman önerilir.")
                    mask_upper = st.checkbox("Mask Upper Triangle", value=True, help="Simetrik tekrarı önler, daha sade görünür.")
                with col_c2:
                    cmap_choice = st.selectbox("Color Palette", ["coolwarm", "RdBu_r", "viridis", "magma", "seismic", "icefire"], index=0)
                    show_annot = st.checkbox("Show Values", value=True)
                with col_c3:
                    annot_font_size = st.slider("Font Size", 6, 24, 10)
                    decimals = st.slider("Decimals", 1, 4, 2)

            with tab2:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_width = st.slider("Graph Width", 6, 30, 12)
                with col_d2:
                    fig_height = st.slider("Graph Height", 6, 30, 10)

        # --- HESAPLAMA ---
        corr_matrix = df_f[num_cols].corr(method=corr_method)

        # --- ÇİZİM ---
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Maskeleme (Üst üçgeni beyaz yap)
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix, 
            annot=show_annot,            # Değerleri yaz/yazma
            fmt=f".{decimals}f",        # Virgülden sonra kaç basamak
            cmap=cmap_choice,           # Renk paketi
            ax=ax, 
            mask=mask,                  # Üst üçgen maskesi
            annot_kws={"size": annot_font_size, "weight": "bold"}, # Yazı boyutu ve kalınlığı
            linewidths=1,               # Kutular arası beyaz çizgi kalınlığı
            linecolor='white',
            cbar_kws={"shrink": 0.8},   # Renk barının boyutu
            square=True,                 # Kutuları kare yap
            vmin=-1, vmax=1             # Renk skalasını -1 ile +1 arasına sabitle
        )
        
        # Eksen Yazılarını Düzelt
        plt.xticks(rotation=45, ha='right', fontsize=annot_font_size + 2)
        plt.yticks(rotation=0, fontsize=annot_font_size + 2)
        
        plt.title(f"{corr_method.capitalize()} Correlation Matrix", fontsize=annot_font_size + 4, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)

        # --- İNDİRME ---
        st.markdown("---")
        col_down1, col_down2 = st.columns([3, 1])
        with col_down2:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', transparent=True)
            st.download_button(
                label="📥 Heatmap İndir (300 DPI)",
                data=buf.getvalue(),
                file_name=f"correlation_heatmap_{corr_method}.png",
                mime="image/png",
                use_container_width=True
            )
            # ---------------------------------------------------------
    # 2. HEATMAP: R ve P DEĞERLERİ BİR ARADA
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader(f"Detailed Heatmap ({corr_method.capitalize()} + P-Values)")
    st.info("Visual representation where color indicates strength (r), and text shows significance (p).")

    # Yazı boyutu ayarı (Bu grafik daha yoğun olacağı için)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        p_font_size = st.slider("Text Size", 6, 20, 9, key="p_font")
    with col_h2:
        fig_h_height = st.slider("Height", 6, 30, 12, key="p_height")

    # --- HESAPLAMA (R ve P Matrislerini Oluştur) ---
    def get_annotated_matrix(df, method):
        cols = df.columns
        # 1. R Matrisi (Renkler için sayısal)
        r_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        # 2. Text Matrisi (Görünecek yazı için string)
        text_matrix = pd.DataFrame(index=cols, columns=cols, dtype=object)
        
        for c1 in cols:
            for c2 in cols:
                # Köşegen (Kendisiyle ilişkisi)
                if c1 == c2:
                    r_matrix.loc[c1, c2] = 1.0
                    text_matrix.loc[c1, c2] = "1.0\n(-)"
                    continue
                
                # Veri temizliği
                valid_data = df[[c1, c2]].dropna()
                if len(valid_data) < 2:
                    r_matrix.loc[c1, c2] = np.nan
                    text_matrix.loc[c1, c2] = "NA"
                    continue

                # Hesaplama
                if method == 'pearson':
                    r, p = pearsonr(valid_data[c1], valid_data[c2])
                elif method == 'spearman':
                    r, p = spearmanr(valid_data[c1], valid_data[c2])
                elif method == 'kendall':
                    r, p = kendalltau(valid_data[c1], valid_data[c2])
                else:
                    r, p = np.nan, np.nan
                
                # Matrislere işle
                r_matrix.loc[c1, c2] = r
                
                # P formatı
                p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
                star = "*" if p < 0.05 else ""
                
                # Kutu içinde görünecek yazı:
                # 0.54*
                # (p=0.02)
                text_matrix.loc[c1, c2] = f"{r:.2f}{star}\n(p={p_str})"
        
        return r_matrix, text_matrix

    # Hesaplamayı Başlat
    if len(num_cols) > 1:
        r_data, text_data = get_annotated_matrix(df_f[num_cols], corr_method)

        # Çizim
        fig2, ax2 = plt.subplots(figsize=(fig_width, fig_h_height))
        
        # Maskeleme (Yukarıdaki ayarı miras alır)
        mask_map = None
        if mask_upper:
            mask_map = np.triu(np.ones_like(r_data, dtype=bool))

        sns.heatmap(
            r_data.astype(float),       # Renkler buradan gelir
            annot=text_data.values,     # Yazılar buradan gelir
            fmt="",                     # String formatı (Hata almamak için boş bırakılır)
            cmap=cmap_choice,
            ax=ax2,
            mask=mask_map,
            annot_kws={"size": p_font_size, "weight": "normal"},
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1,
            center=0
        )

        plt.xticks(rotation=45, ha='right', fontsize=p_font_size+2)
        plt.yticks(rotation=0, fontsize=p_font_size+2)
        plt.title(f"Correlation Analysis with P-Values", fontsize=p_font_size+4, fontweight='bold')
        
        st.pyplot(fig2)

        # İndirme Butonu
        col_d1, col_d2 = st.columns([3, 1])
        with col_d2:
            import io
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight', transparent=True)
            st.download_button(
                label="📥 İndir (P-Value Heatmap)", 
                data=buf2.getvalue(), 
                file_name=f"heatmap_pvalues_{corr_method}.png", 
                mime="image/png",
                use_container_width=True
            )

# =========================================================
# SAYFA 4: KOMPAKT ÇOKLU REGRESYON (STANDARDIZE GÖRSEL SEÇENEĞİ)
# =========================================================
elif page == "4. Regression":
    st.header("4. Multivariate Regression Summary")
    st.markdown("""
    This module builds regression models for each parameter separately.
    It reports the effect of the **Main Factor** (e.g., PERIOD) adjusted for confounders.
    """)
    
    st.markdown("---")
    
    # 1. DEĞİŞKEN SEÇİMİ
    numeric_candidates = df_f.select_dtypes(include=np.number).columns.tolist()
    
    # A) TARGETS
    # Sizin belirlediğiniz öncelikli liste
    wanted_defaults = [
        "WBC", "NEUT_ABS", "LYMPH_ABS", "MONO_ABS", "PLT", "MPV", 
        "SII", "SIRI", "NLR", "dNLR", "PLR", "dPLR", "MLR", "NMLR", "AISI", "CRP",
    ]
    
    # Bu listedekilerden, sadece yüklenen dosyada mevcut olanları seç (Hata almamak için)
    defaults = [c for c in wanted_defaults if c in numeric_candidates]
    
    targets = st.multiselect("1. Parameters to Analyze (Rows):", 
                             numeric_candidates, default=defaults)
    
    # B) ANA FAKTÖR
    remaining = [c for c in all_cols if c not in targets]
    main_factor_def = remaining.index("PERIOD") if "PERIOD" in remaining else 0
    main_factor = st.selectbox("2. Main Factor (Group/Period):", remaining, index=main_factor_def)
    
    # C) CONFOUNDERS
    std_confounders = ["AGE", "SEX", "BMI", "RACE", "SMOKING_STATUS"]
    avail_conf = [c for c in std_confounders if c in df_f.columns and c not in targets and c != main_factor]
    
    confounders = st.multiselect("3. Adjust for (Confounders):", 
                                 options=[c for c in remaining if c != main_factor],
                                 default=avail_conf)
    
    # ---------------------------------------------------------
    # EKLENTİ: VERİ KAYBI ANALİZİ (Regresyon öncesi kontrol)
    # ---------------------------------------------------------
    if targets:
        # 1. Analize girecek tüm sütunları topla
        all_model_vars = targets + [main_factor] + confounders
        
        # 2. Bu sütunlarda eksik verisi olanları say
        # Mevcut (Flowchart sonrası) veri sayısı
        n_start = len(df_f)
        
        # Sadece seçilen değişkenler için temiz veri sayısı
        df_clean = df_f[all_model_vars].dropna()
        n_end = len(df_clean)
        n_lost = n_start - n_end

        # 3. Kullanıcıya Bilgi Ver
        if n_lost > 0:
            st.warning(f"⚠️ **Note on Sample Size:**")
            st.markdown(f"""
            * **Flowchart Data:** {n_start}
            * **Regression Data:** {n_end} 
            * **Dropped due to missing values:** {n_lost} participants
            """)
            
            # Hangi değişkende ne kadar boş var? (Detay)
            with st.expander("🕵️ Which variable is causing data loss?"):
                missing_counts = df_f[all_model_vars].isnull().sum().sort_values(ascending=False)
                missing_counts = missing_counts[missing_counts > 0]
                if not missing_counts.empty:
                    st.dataframe(missing_counts.rename("Missing Count"), use_container_width=True)
                else:
                    st.write("Data is clean, loss might be due to combined filtering.")
    
    st.markdown("---")

    # --- HESAPLAMA BUTONU ---
    if st.button("Generate Table & Plot"):
        if not targets:
            st.warning("Please select at least one parameter.")
        else:
            summary_data = []
            predictors = [main_factor] + confounders
            progress_bar = st.progress(0)
            
            for i, target_var in enumerate(targets):
                progress_bar.progress((i + 1) / len(targets))
                
                # Model Verisi
                cols = [target_var] + predictors
                model_data = df_f[cols].dropna()
                
                if len(model_data) < 50: continue

                target_std = model_data[target_var].std()

                # Model Kurulumu
                formula = f"{target_var} ~ {' + '.join(predictors)}"
                model = smf.ols(formula, data=model_data).fit()
                
                target_coef_name = None
                for name in model.params.index:
                    if main_factor in name and name != "Intercept":
                        target_coef_name = name
                        break
                
                if target_coef_name:
                    coef = model.params[target_coef_name]
                    conf = model.conf_int().loc[target_coef_name]
                    p_val = model.pvalues[target_coef_name]
                    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
                    
                    # Standardize Beta Hesabı (Grafikteki mantığın aynısı)
                    if target_std != 0:
                        std_beta = coef / target_std
                    else:
                        std_beta = 0
                    
                    summary_data.append({
                        "Parametre": target_var,
                        "Adjusted Beta (Raw)": f"{coef:.2f}",
                        "SD of Dependent Var": f"{target_std:.2f}",  # YENİ SÜTUN (Standart Sapma)
                        "Standardized Beta": f"{std_beta:.3f}",
                        "%95 CI Lower": f"{conf[0]:.2f}",
                        "%95 CI Upper": f"{conf[1]:.2f}",
                        "p-value": p_str,
                        "N": int(model.nobs),
                        "R2": f"{model.rsquared:.3f}",
                        # Gizli veriler
                        "_coef": coef,
                        "_lower": conf[0],
                        "_upper": conf[1],
                        "_p_val": p_val,
                        "_name": target_coef_name,
                        "_std": target_std
                    })
            
            progress_bar.empty()

            # --- SONUCU HAFIZAYA KAYDET (KRİTİK HAMLE) ---
            if summary_data:
                st.session_state['reg_results'] = pd.DataFrame(summary_data)
                st.session_state['reg_confounders'] = confounders # Başlık için sakla
            else:
                st.error("Model could not be built.")

    # --- SONUÇLARI GÖSTER (HAFIZADAN) ---
    # Butona basılmasa bile hafızada veri varsa burası çalışır
    if 'reg_results' in st.session_state:
        df_res = st.session_state['reg_results']
        current_confounders = st.session_state.get('reg_confounders', [])
        
        # --- 1. GELİŞMİŞ TABLO ---
        st.markdown("---")
        st.subheader("📊 Multivariate Regression Table")
        st.caption(f"**Adjusted for:** {', '.join(current_confounders)}")
        
        # Tabloda artık hem Ham hem Standart değer görünecek
        # YENİ SIRALAMA: Raw -> SD -> Standardized
        show_cols = [
            "Parametre", 
            "Adjusted Beta (Raw)", 
            "SD of Dependent Var",   # Yeni eklediğimiz sütun
            "Standardized Beta", 
            "p-value", 
            "%95 CI Lower", 
            "%95 CI Upper", 
            "N", 
            "R2"
        ]
        
        # Güvenlik Kontrolü
        missing_cols = [c for c in show_cols if c not in df_res.columns]
        if not missing_cols:
            st.dataframe(df_res[show_cols], use_container_width=True)
            
            # Tablo İndirme
            csv_table = df_res[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Table (CSV)",
                data=csv_table,
                file_name="multivariate_regression_results.csv",
                mime="text/csv",
            )
            
            # --- 2. FOREST PLOT ---
            st.markdown("---")
            st.subheader("📈 Forest Plot")
            
            # Checkbox artık sayfayı yenilese bile veri 'st.session_state' içinde durduğu için grafik çizilir
            use_std = st.checkbox("✅ Standardize Graph (Visual Improvement)", value=True, 
                                  help="Scales different units to be comparable.")
            
            if use_std:
                st.info("💡 **Info:** Values are divided by SD to get **'Standardized Beta'**.")
            
            # Grafik Verisi
            plot_df = df_res.iloc[::-1]
            
            fig, ax = plt.subplots(figsize=(8, len(plot_df)*0.5 + 2))
            y_pos = range(len(plot_df))
            
            for i, (idx, row) in enumerate(plot_df.iterrows()):
                val_coef = row["_coef"]
                val_low = row["_lower"]
                val_high = row["_upper"]
                
                if use_std:
                    s = row["_std"]
                    val_coef /= s
                    val_low /= s
                    val_high /= s
                
                c = 'firebrick' if row["_p_val"] < 0.05 else 'gray'
                alpha_line = 1.0 if row["_p_val"] < 0.05 else 0.5
                
                ax.hlines(y=i, xmin=val_low, xmax=val_high, color=c, linewidth=2, alpha=alpha_line, zorder=1)
                ax.plot(val_coef, i, 'o', color=c, markersize=8, markeredgecolor='black', zorder=2)

            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_df["Parametre"], fontweight="bold", fontsize=10)
            
            xlabel = "Standardized Beta (Effect Size)" if use_std else f"Raw Adjusted Beta"
            ax.set_xlabel(xlabel, fontweight="bold")
            
            ax.grid(axis='x', linestyle=':', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
            # Grafik İndirme
            st.markdown("---")
            col_rd1, col_rd2 = st.columns([3, 1])
            with col_rd2:
                import io 
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', transparent=True)
                fname = "forest_plot_std.png" if use_std else "forest_plot_raw.png"
                st.download_button(label="📥 Download Plot (300 DPI)", data=buf.getvalue(), file_name=fname, mime="image/png", use_container_width=True)
        else:
            st.warning("Please click 'Generate Table & Plot' again to update the table structure.")

    # ==========================================
    # ⬇️ 5. DETAYLI MODEL İNCELEMESİ (PRE REFERANS -> POST GÖSTERİMİ) ⬇️
    # ==========================================
    st.markdown("---")
    st.header("5. Detailed Factor Analysis")
    st.markdown("""
    Compare the effect size of **all variables** (Age, BMI, Sex, Season, etc.) for a single selected parameter.
    **Note:** This model uses 'Pre-Pandemic' as the reference, showing the effect of the **Post-Pandemic** period.
    """)

    # 1. Predictors Listesini Hazırla
    predictors = [main_factor] + confounders
    
    # Eğer veride SEASON_CODE varsa ve seçilmemişse ekle
    if "SEASON_CODE" in df_f.columns and "SEASON_CODE" not in predictors:
        predictors.append("SEASON_CODE")

    # 2. Hangi parametreye bakacağız?
    if not targets:
        st.info("Please select at least one parameter above.")
    else:
        detail_target = st.selectbox("🔍 Select Parameter to Inspect:", targets)

        if detail_target:
            # Model Verisi Hazırlama
            cols = [detail_target] + predictors
            model_data = df_f[cols].dropna()
            
            if len(model_data) > 30:
                y_std = model_data[detail_target].std()
                
                # --- FORMÜL AYARI (KRİTİK KISIM) ---
                # PERIOD değişkenini bulup, referansını 'Pre' yapıyoruz.
                # Böylece sonuç 'Post' olarak çıkar.
                formula_terms = []
                for p in predictors:
                    if p == "PERIOD":
                        formula_terms.append("C(PERIOD, Treatment(reference='Pre'))")
                    else:
                        formula_terms.append(p)
                
                formula = f"{detail_target} ~ {' + '.join(formula_terms)}"
                model = smf.ols(formula, data=model_data).fit()
                
                # --- VERİLERİ TOPLA ---
                params = model.params.drop("Intercept", errors='ignore')
                conf = model.conf_int().drop("Intercept", errors='ignore')
                pvals = model.pvalues.drop("Intercept", errors='ignore')
                
                detail_rows = []
                for var_name in params.index:
                    coef = params[var_name]
                    
                    # Standardize Beta Hesabı
                    if y_std != 0:
                        std_beta = coef / y_std
                        lower = conf.loc[var_name][0] / y_std
                        upper = conf.loc[var_name][1] / y_std
                    else:
                        std_beta, lower, upper = 0, 0, 0
                    
                    # --- İSİM TEMİZLEME (Grafik güzel görünsün) ---
                    clean_name = var_name
                    # Karmaşık statsmodels ismini sadeleştir: C(PERIOD, ...)[T.Post] -> PERIOD [Post]
                    if "PERIOD" in var_name and "Post" in var_name:
                        clean_name = "PERIOD [Post]"
                    elif "SEASON_CODE" in var_name:
                        clean_name = "SEASON"
                    
                    detail_rows.append({
                        "Faktör": clean_name, # Temiz isim
                        "Std. Beta": std_beta,
                        "Lower": lower,
                        "Upper": upper,
                        "p-value": pvals[var_name]
                    })
                
                df_detail = pd.DataFrame(detail_rows)
                
                # Sıralama: Etki gücüne göre
                df_detail["Abs_Effect"] = df_detail["Std. Beta"].abs()
                df_detail = df_detail.sort_values("Abs_Effect", ascending=False)
                
                # --- GRAFİK ÇİZİMİ ---
                fig_d, ax_d = plt.subplots(figsize=(8, len(df_detail) * 0.6 + 2))
                y_pos = range(len(df_detail))
                
                for i, (idx, row) in enumerate(df_detail.iterrows()):
                    # Renk ve Şeffaflık
                    c = '#d62728' if row["p-value"] < 0.05 else '#bdbdbd'
                    alpha = 1.0 if row["p-value"] < 0.05 else 0.5
                    
                    # Çubuklar
                    ax_d.errorbar(x=row["Std. Beta"], y=i, 
                                  xerr=[[row["Std. Beta"] - row["Lower"]], [row["Upper"] - row["Std. Beta"]]],
                                  fmt='o', color=c, ecolor=c, capsize=4, elinewidth=2, alpha=alpha)
                    
                    # Değer Etiketi
                    ax_d.text(row["Std. Beta"], i - 0.3, f"{row['Std. Beta']:.2f}", 
                              ha='center', va='center', fontsize=9, color=c, fontweight='bold')

                # Eksenler
                ax_d.set_yticks(y_pos)
                ax_d.set_yticklabels(df_detail["Faktör"], fontweight="bold", fontsize=10)
                ax_d.axvline(x=0, color='black', linestyle='--', linewidth=1)
                
                ax_d.set_xlabel("Standardized Effect Size (Beta / SD)", fontweight="bold")
                ax_d.set_title(f"Determinants of {detail_target}\n(Reference: Pre-Pandemic -> Effect of Post)", pad=10)
                
                # Temizlik
                ax_d.spines['top'].set_visible(False)
                ax_d.spines['right'].set_visible(False)
                ax_d.grid(axis='x', linestyle=':', alpha=0.5)
                
                st.pyplot(fig_d)
                
                st.info(f"📌 **Interpretation:** This chart shows what drives **{detail_target}**. The **PERIOD [Post]** bar specifically shows how the Post-Pandemic era differs from the Pre-Pandemic era.")
                # =========================================================
                # YENİ EKLENTİ: MODEL VARSAYIM KONTROLLERİ (DIAGNOSTICS)
                # =========================================================
                st.markdown("---")
                st.subheader("6. Model Diagnostics & Assumptions")
                
                diag_tab1, diag_tab2 = st.tabs(["⚠️ Multicollinearity (VIF)", "📉 Residual Analysis"])
                
                # --- 1. ÇOKLU BAĞLANTI (VIF) KONTROLÜ ---
                with diag_tab1:
                    st.markdown("**Assumption:** No Multicollinearity (VIF should be < 5 or 10)")
                    
                    # VIF Hesaplama için statsmodels kütüphanesi gerekir
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    
                    # Model verisinden sadece X'leri (bağımsız değişkenleri) al
                    # Intercept (Sabit terim) statsmodels formülünde otomatik eklenir ama VIF için manuel dummyler gerekebilir.
                    # Burada en temiz yol: model.model.exog matrisini kullanmaktır.
                    
                    exog = model.model.exog
                    exog_names = model.model.exog_names
                    
                    vif_data = []
                    for i in range(exog.shape[1]):
                        # Intercept genelde ilk sütundur, VIF'i sonsuz çıkabilir, onu atlayabiliriz veya gösterebiliriz.
                        if exog_names[i] == "Intercept": continue
                            
                        vif_val = variance_inflation_factor(exog, i)
                        vif_data.append({"Variable": exog_names[i], "VIF": vif_val})
                    
                    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
                    
                    col_v1, col_v2 = st.columns([1, 2])
                    with col_v1:
                        st.dataframe(vif_df, use_container_width=True, hide_index=True)
                    
                    with col_v2:
                        # Yorum
                        high_vif = vif_df[vif_df["VIF"] > 5]
                        if not high_vif.empty:
                            st.error(f"⚠️ **Warning:** High Multicollinearity detected in: {', '.join(high_vif['Variable'].tolist())}. Consider removing one of them.")
                        else:
                            st.success("✅ **Pass:** All variables have VIF < 5. No multicollinearity issues.")

                # --- 2. NORMALLİK VE EŞVARYANSLILIK (RESIDUALS) ---
                with diag_tab2:
                    residuals = model.resid
                    fitted = model.fittedvalues
                    
                    st.markdown("**Assumptions:** Normality of Residuals & Homoscedasticity")
                    
                    fig_res, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # A) Residuals vs Fitted (Homoscedasticity)
                    # Huni şekli olmamalı, rastgele dağılmalı
                    sns.scatterplot(x=fitted, y=residuals, ax=ax_r1, alpha=0.5)
                    ax_r1.axhline(0, color='red', linestyle='--')
                    ax_r1.set_title("Residuals vs Fitted\n(Check for Homoscedasticity)")
                    ax_r1.set_xlabel("Fitted Values")
                    ax_r1.set_ylabel("Residuals")
                    
                    # B) Histogram & Q-Q Plot (Normality)
                    from scipy.stats import probplot
                    probplot(residuals, dist="norm", plot=ax_r2)
                    ax_r2.set_title("Q-Q Plot\n(Check for Normality)")
                    
                    st.pyplot(fig_res)
                    
                    st.caption("""
                    * **Left Plot:** Points should be randomly scattered around the red line (No funnel shape).
                    * **Right Plot:** Points should follow the red line (Normality). 
                    * *Note:* In large samples (N > 1000), slight deviations from normality are acceptable (Central Limit Theorem).
                    """)
                    
            else:
                st.warning("Not enough data for this analysis.")
