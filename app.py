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
    "RIDAGEYR": "AGE", "RIAGENDR": "SEX", "RIDRETH3": "RACE",
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
    # ==========================================
    # ⬇️ BURAYA YAPIŞTIRIN (TANSİYON HESABI) ⬇️
    # ==========================================
    
    # 0. TANSİYON VE NABIZ ORTALAMALARI (Yeni Ekleme)
    # Önce sütunların veride olup olmadığını kontrol edip sayıya çevirelim
    bp_cols = ['BPXOSY2', 'BPXOSY3', 'BPXODI2', 'BPXODI3', 'BPXOPLS2', 'BPXOPLS3']
    for col in bp_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # Ortalamaları Hesaplama (Varsa hesapla, yoksa NaN bırak)
    if 'BPXOSY2' in out.columns and 'BPXOSY3' in out.columns:
        out['SYSTOLICBP'] = out[['BPXOSY2', 'BPXOSY3']].mean(axis=1, skipna=True)
    
    if 'BPXODI2' in out.columns and 'BPXODI3' in out.columns:
        out['DIASTOLICBP'] = out[['BPXODI2', 'BPXODI3']].mean(axis=1, skipna=True)
        
    if 'BPXOPLS2' in out.columns and 'BPXOPLS3' in out.columns:
        out['PULSEAVG'] = out[['BPXOPLS2', 'BPXOPLS3']].mean(axis=1, skipna=True)
    
    # 1. Sayısal Dönüşüm
    cols_to_numeric = [
        "NEUT_ABS", "LYMPH_ABS", "MONO_ABS", "PLT", "WBC", "CRP",
        "AGE", "AGE_STARTED", "AGE_FIRST_CIG", "CIGS_PER_DAY_NOW", "SMOKE_LIFE_100", "SMOKE_NOW"
    ]
    for c in cols_to_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 2. İndeksler
    # Mevcutlar:
        # NLR
    if "NEUT_ABS" in out.columns and "LYMPH_ABS" in out.columns:
        out["NLR"] = out["NEUT_ABS"] / out["LYMPH_ABS"]
        
        # dNLR (Derived NLR) -> Formül: Nötrofil / (WBC - Nötrofil)
        if "WBC" in out.columns:
             # Paydanın 0 olmasını engellemek için küçük bir güvenlik önlemi alınabilir ama genelde gerekmez.
             denom = out["WBC"] - out["NEUT_ABS"]
             out["dNLR"] = out["NEUT_ABS"] / denom
        
        # SII ve PLR
        if "PLT" in out.columns:
            out["SII"] = (out["PLT"] * out["NEUT_ABS"]) / out["LYMPH_ABS"]
            out["PLR"] = out["PLT"] / out["LYMPH_ABS"]
        # SIRI
        if "MONO_ABS" in out.columns:
            out["SIRI"] = (out["NEUT_ABS"] * out["MONO_ABS"]) / out["LYMPH_ABS"]
            
            # MLR (Monocyte to Lymphocyte Ratio)
            out["MLR"] = out["MONO_ABS"] / out["LYMPH_ABS"]
            
            # NMLR (Neutrophil + Monocyte to Lymphocyte Ratio)
            # Not: Eğer kastettiğiniz sadece Nötrofil/Monosit ise formülü: out["NEUT_ABS"] / out["MONO_ABS"] yapın.
            out["NMLR"] = (out["NEUT_ABS"] + out["MONO_ABS"]) / out["LYMPH_ABS"]

            # AISI
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
    # ==========================================
    # ⬇️ IRK ETİKETLEME (BURAYA YAPIŞTIRIN) ⬇️
    # ==========================================
    if "RACE" in out.columns:
        # NHANES RIDRETH3 Kodları
        race_mapping = {
            1: "Mexican American",
            2: "Other Hispanic",
            3: "Non-Hispanic White",
            4: "Non-Hispanic Black",
            6: "Non-Hispanic Asian",
            7: "Other/Multi-Racial"
        }
        # Sadece sütun sayısal ise çevir (Hata almamak için)
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
default_vars = ["SII", "NLR", "dNLR", "PLR", "MLR", "NMLR", "SYSTOLICBP", "DIASTOLICBP", "PULSEAVG", "CRP", "WBC", "AGE", "BMI", "WAIST_CM", "SMOKING_STATUS"]
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
# SAYFA 1: ÖZET TABLO (SİMGELİ ETKİ BÜYÜKLÜKLERİ: d, e)
# =========================================================
if page == "1. Özet Tablo":
    st.header("1. Özet İstatistikler")
    
    rows = []
    posthoc_results = {}
    
    # --- TEST VE ETKİ SİMGELERİ ---
    SYM_T = "ᵃ"       # T-Test
    SYM_MWU = "ᵇ"     # Mann-Whitney U
    SYM_CHI = "ᶜ"     # Chi-Square
    SYM_DELTA = "ᵈ"   # Cliff's Delta (Sayısal)
    SYM_V = "ᵉ"       # Cramer's V (Kategorik - User isteği 'e')

    # --- YARDIMCI FONKSİYONLAR ---
    def fmt_median_iqr(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return "NA"
        med = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return f"{med:.3g} [{q1:.3g}–{q3:.3g}]"

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
                d_pre = fmt_median_iqr(pre_vals)
                d_post = fmt_median_iqr(post_vals)
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
elif page == "2. Grafikler":
    st.header("2. Grafikler (Yayına Hazırlık Modu)")

    # --- 1. DEĞİŞKEN SEÇİMİ ---
    plot_vars = st.multiselect(
        "Çizilecek Değişkenleri Seçin:", 
        vars_to_analyze, 
        default=vars_to_analyze[:min(4, len(vars_to_analyze))] if len(vars_to_analyze) > 0 else None
    )

    if plot_vars:
        # --- 2. GRAFİK AYARLARI (EXPANDER) ---
        with st.expander("⚙️ Grafik Görünüm & İstatistik Ayarları", expanded=True):
            tab1, tab2, tab3 = st.tabs(["📊 Veri & Eksen", "📏 Çizgi Stili", "🏷️ Etiketler"])
            
            with tab1:
                col_data1, col_data2 = st.columns(2)
                with col_data1:
                    st.markdown("##### 🔍 Veri Gösterimi")
                    # Outlier Filtresi
                    remove_outliers = st.checkbox("Aykırı Değerleri Gizle (Outliers - IQR Yöntemi)", value=False, help="Grafiği bastıran çok yüksek değerleri (Q3 + 1.5*IQR üzerini) gizler. Medyan farklarını görmek için idealdir.")
                    # Log Scale
                    use_log = st.checkbox("Logaritmik Ölçek (Log Scale)", value=False, help="Aşırı büyük ve küçük değerleri aynı grafikte dengeli gösterir.")
                    
                    # İstatistik Tipi
                    plot_type = st.radio(
                        "İstatistiksel Özet:", 
                        ["Ortalama ± SD (Mean ± SD)", "Medyan ± IQR (Median ± IQR)"],
                        horizontal=True
                    )
                with col_data2:
                    st.markdown("##### 🎨 Görünüm")
                    cols_num = st.slider("Yan Yana Grafik", 1, 4, 2)
                    dot_size = st.slider("Nokta Büyüklüğü", 1, 15, 4)
                    c1 = st.color_picker("Pre Rengi", "#4c72b0")
                    c2 = st.color_picker("Post Rengi", "#c44e52")

            with tab2:
                st.markdown("##### Hata Çubukları (Error Bars)")
                col_err1, col_err2, col_err3 = st.columns(3)
                with col_err1:
                    err_linewidth = st.slider("Çizgi Kalınlığı", 0.5, 5.0, 1.5)
                with col_err2:
                    err_capsize = st.slider("Şapka Genişliği", 0, 20, 8)
                with col_err3:
                    err_capthick = st.slider("Şapka Kalınlığı", 0.5, 5.0, 1.5)

            with tab3:
                st.info("Tek değişken seçiliyse başlıkları özelleştirebilirsiniz.")
                custom_title = st.text_input("Grafik Başlığı", value="")
                col_lbl1, col_lbl2 = st.columns(2)
                with col_lbl1: custom_xlabel = st.text_input("X Ekseni", value="Grup")
                with col_lbl2: custom_ylabel = st.text_input("Y Ekseni", value=plot_vars[0] if len(plot_vars)==1 else "Değer")

        is_parametric_plot = "Ortalama" in plot_type
        
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
            
            # --- OUTLIER FİLTRELEME (Sadece Grafik İçin) ---
            # Orijinal df_f bozulmaz, plot_data geçici oluşturulur
            plot_data = df_f.copy()
            
            if remove_outliers and pd.api.types.is_numeric_dtype(plot_data[v]):
                Q1 = plot_data[v].quantile(0.25)
                Q3 = plot_data[v].quantile(0.75)
                IQR = Q3 - Q1
                upper_limit = Q3 + 1.5 * IQR
                # Alt limit genelde biyolojik veride 0'dır ama yine de yazalım
                lower_limit = Q1 - 1.5 * IQR
                
                # Filtrele
                plot_data = plot_data[(plot_data[v] <= upper_limit) & (plot_data[v] >= lower_limit)]
                
                # Kaç veri atıldığını konsola veya başlığa yazabiliriz (Opsiyonel)
                # st.write(f"{v}: {len(df_f) - len(plot_data)} aykırı değer gizlendi.")

            # Kategorik kontrolü
            is_categorical = (v in forced_cat_vars) or (plot_data[v].dtype == 'object')
            
            if is_categorical:
                counts = plot_data.groupby(["PERIOD", v]).size().reset_index(name="Count")
                sns.barplot(data=counts, x="PERIOD", y="Count", hue=v, ax=ax, palette="Set2")
                ax.set_title(f"{v} Dağılımı")
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

                # --- İSTATİSTİK HESABI ---
                periods = ["Pre", "Post"]
                for j, period in enumerate(periods):
                    subset = plot_data[plot_data["PERIOD"] == period][v].dropna()
                    if len(subset) == 0: continue

                    if is_parametric_plot:
                        center = subset.mean()
                        spread = subset.std()
                        yerr = spread 
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

                # --- LOG SCALE AYARI ---
                if use_log:
                    ax.set_yscale('log')
                    # Logaritmik skalada 0 hatası olmasın diye minik bir değer ekleme gerekebilir
                    # ama Seaborn genelde bunu handle eder. Ederse de etiketleri düzeltelim:
                    from matplotlib.ticker import ScalarFormatter
                    ax.yaxis.set_major_formatter(ScalarFormatter())

            # --- ETİKETLER ---
            if len(plot_vars) == 1:
                ax.set_title(custom_title if custom_title else v, fontweight="bold", fontsize=14)
                ax.set_xlabel(custom_xlabel, fontsize=12, fontweight="bold")
                ax.set_ylabel(custom_ylabel, fontsize=12, fontweight="bold")
            else:
                ax.set_title(v, fontweight="bold")
                ax.set_xlabel("")

            # Temizlik
            ax.grid(axis='y', linestyle='--', alpha=0.3, which='both') # Log için 'both'
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
elif page == "3. Korelasyon":
    st.header("3. Korelasyon Analizi (Heatmap)")

    # 1. Sadece Sayısal Değişkenleri Seç
    num_cols = [c for c in vars_to_analyze if pd.api.types.is_numeric_dtype(df_f[c]) and c not in forced_cat_vars]

    if len(num_cols) < 2:
        st.warning("⚠️ Korelasyon analizi yapabilmek için sol taraftan en az 2 adet sayısal değişken seçmelisiniz.")
    else:
        # --- AYARLAR MENÜSÜ ---
        with st.expander("⚙️ Korelasyon ve Görünüm Ayarları", expanded=True):
            tab1, tab2 = st.tabs(["📊 Analiz & Stil", "📐 Boyutlar"])
            
            with tab1:
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    corr_method = st.selectbox("Yöntem", ["spearman", "pearson", "kendall"], help="Normal dağılım yoksa Spearman önerilir.")
                    mask_upper = st.checkbox("Üst Üçgeni Gizle (Mask Upper)", value=True, help="Simetrik tekrarı önler, daha sade görünür.")
                with col_c2:
                    cmap_choice = st.selectbox("Renk Paleti", ["coolwarm", "RdBu_r", "viridis", "magma", "seismic", "icefire"], index=0)
                    show_annot = st.checkbox("Değerleri Göster", value=True)
                with col_c3:
                    annot_font_size = st.slider("Yazı Puntosu (Font Size)", 6, 24, 10)
                    decimals = st.slider("Ondalık Basamak", 1, 4, 2)

            with tab2:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_width = st.slider("Grafik Genişliği", 6, 30, 12)
                with col_d2:
                    fig_height = st.slider("Grafik Yüksekliği", 6, 30, 10)

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
            annot=show_annot,           # Değerleri yaz/yazma
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

# =========================================================
# SAYFA 4: KOMPAKT ÇOKLU REGRESYON (STANDARDIZE GÖRSEL SEÇENEĞİ)
# =========================================================
elif page == "4. Regresyon":
    st.header("4. Çoklu Regresyon Özeti (Compact Summary)")
    st.markdown("""
    Bu modül, seçilen tüm parametreler (SII, NLR, PLR vb.) için ayrı ayrı regresyon modeli kurar.
    Karmaşayı önlemek için sadece **Ana Faktörün** (Örn: PERIOD) etkisini raporlar.
    """)
    
    st.markdown("---")
    
    # 1. DEĞİŞKEN SEÇİMİ
    numeric_candidates = df_f.select_dtypes(include=np.number).columns.tolist()
    
    # A) TARGETS
    defaults = [c for c in ["SII", "NLR", "PLR", "dNLR", "CRP"] if c in numeric_candidates]
    targets = st.multiselect("1. Analiz Edilecek Parametreler (Satırlar):", 
                             numeric_candidates, default=defaults)
    
    # B) ANA FAKTÖR
    remaining = [c for c in all_cols if c not in targets]
    main_factor_def = remaining.index("PERIOD") if "PERIOD" in remaining else 0
    main_factor = st.selectbox("2. Ana Faktör (Grup/Dönem):", remaining, index=main_factor_def)
    
    # C) CONFOUNDERS
    std_confounders = ["AGE", "SEX", "BMI", "RACE", "SMOKING_STATUS"]
    avail_conf = [c for c in std_confounders if c in df_f.columns and c not in targets and c != main_factor]
    
    confounders = st.multiselect("3. Düzeltme Faktörleri (Adjusted for):", 
                                 options=[c for c in remaining if c != main_factor],
                                 default=avail_conf)
    
    st.markdown("---")

    if st.button("Tablo ve Grafiği Oluştur"):
        if not targets:
            st.warning("Lütfen en az bir parametre seçin.")
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

                # Standart Sapma (Görsel Düzeltme İçin Lazım Olacak)
                target_std = model_data[target_var].std()

                # Model Kurulumu
                formula = f"{target_var} ~ {' + '.join(predictors)}"
                model = smf.ols(formula, data=model_data).fit()
                
                # Hedef Katsayıyı Bul
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
                    
                    summary_data.append({
                        "Parametre": target_var,
                        "Adjusted Beta (Grup)": f"{coef:.2f}",
                        "%95 GA": f"{conf[0]:.2f} – {conf[1]:.2f}",
                        "p": p_str,
                        # Gizli veriler (Grafik için)
                        "_coef": coef,
                        "_lower": conf[0],
                        "_upper": conf[1],
                        "_p_val": p_val,
                        "_name": target_coef_name,
                        "_std": target_std  # Görsel ölçekleme için
                    })
            
            progress_bar.empty()

            if summary_data:
                # --- TABLO ---
                df_res = pd.DataFrame(summary_data)
                display_cols = ["Parametre", "Adjusted Beta (Grup)", "%95 GA", "p"]
                
                st.subheader("📊 Multivariate Regression Summary")
                st.caption(f"**Adjusted for:** {', '.join(confounders)}")
                st.table(df_res[display_cols])
                
                # --- FOREST PLOT ---
                st.markdown("---")
                st.subheader("📈 Forest Plot: Etki Karşılaştırması")
                
                # SEÇENEK: Standardizasyon
                use_std = st.checkbox("✅ Grafiği Standardize Et (Görsel İyileştirme)", value=True, 
                                      help="Farklı birimlerdeki (Örn: SII vs NLR) değişkenleri aynı ölçeğe getirir. Gri çizgilerin görünmesini sağlar.")
                
                if use_std:
                    st.info("💡 **Bilgi:** Değerler standart sapmalarına bölünerek **'Standardized Beta'**ya dönüştürüldü. Bu sayede küçük değerler (NLR) ile büyük değerler (SII) aynı grafikte net görülür.")
                
                # Grafik Verisi
                plot_df = df_res.iloc[::-1]
                
                fig, ax = plt.subplots(figsize=(8, len(plot_df)*0.8 + 2))
                y_pos = range(len(plot_df))
                
                # MANUEL DÖNGÜ (Hatasız Çizim)
                for i, (idx, row) in enumerate(plot_df.iterrows()):
                    # Verileri Çek
                    val_coef = row["_coef"]
                    val_low = row["_lower"]
                    val_high = row["_upper"]
                    
                    # Eğer Standardizasyon Seçildiyse:
                    if use_std:
                        s = row["_std"]
                        val_coef /= s
                        val_low /= s
                        val_high /= s
                    
                    # Renk Seçimi
                    c = 'firebrick' if row["_p_val"] < 0.05 else 'gray'
                    # Anlamsız olanları biraz daha soluk yapalım ki karışmasın
                    alpha_line = 1.0 if row["_p_val"] < 0.05 else 0.5
                    
                    # 1. Çizgi
                    ax.hlines(y=i, xmin=val_low, xmax=val_high, 
                              color=c, linewidth=2, alpha=alpha_line, zorder=1)
                    
                    # 2. Nokta
                    ax.plot(val_coef, i, 'o', 
                            color=c, markersize=8, markeredgecolor='black', zorder=2)

                # 0 Referans Çizgisi
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
                
                # Eksenler
                ax.set_yticks(y_pos)
                ax.set_yticklabels(plot_df["Parametre"], fontweight="bold", fontsize=10)
                
                xlabel = "Standardized Beta (Etki Gücü)" if use_std else f"Raw Adjusted Beta ({summary_data[0]['_name']})"
                ax.set_xlabel(xlabel, fontweight="bold")
                
                # Grid ve Temizlik
                ax.grid(axis='x', linestyle=':', alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

            else:
                st.error("Model kurulamadı.")
