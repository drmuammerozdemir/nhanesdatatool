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
# SAYFA 4: REGRESYON (GELİŞMİŞ & GÖRSELLEŞTİRMELİ)
# =========================================================
elif page == "4. Regresyon":
    st.header("4. Çok Değişkenli Doğrusal Regresyon (Multivariate Linear Regression)")
    st.markdown("""
    Bu modül, **En Küçük Kareler Yöntemi (OLS)** kullanarak bağımlı değişken (Y) üzerindeki etkileri analiz eder.
    * **Hedef (Y):** Etkilenen değişken (Örn: SII, NLR).
    * **Ana Faktör (X):** Asıl merak edilen etken (Örn: PERIOD).
    * **Confounders:** Sonucu etkileyebilecek ve düzeltilmesi gereken yan faktörler (Örn: Yaş, Sigara, BMI).
    """)
    
    st.markdown("---")
    
    # 1. DEĞİŞKEN SEÇİMİ (TAM ÖZGÜRLÜK)
    # Sadece sayısal sütunları Y adayı yap, ama kullanıcı isterse hepsini görsün
    numeric_candidates = df_f.select_dtypes(include=np.number).columns.tolist()
    # Varsayılan olarak SII seçmeye çalış
    default_ix = numeric_candidates.index("SII") if "SII" in numeric_candidates else 0
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("1. Bağımlı Değişkeni Seçin (Y - Hedef)", 
                              options=all_cols, # Tüm sütunlar
                              index=all_cols.index("SII") if "SII" in all_cols else 0)
        
    with col2:
        # Y seçildikten sonra kalanları listele
        remaining_cols = [c for c in all_cols if c != target]
        main_factor = st.selectbox("2. Ana Faktörü Seçin (X - İlgi Odağı)", 
                                   options=remaining_cols,
                                   index=remaining_cols.index("PERIOD") if "PERIOD" in remaining_cols else 0)

    # Confounders Seçimi
    confounder_candidates = [c for c in remaining_cols if c != main_factor]
    default_confounders = [c for c in ["AGE", "SEX", "BMI", "SMOKING_STATUS", "RACE"] if c in confounder_candidates]
    
    confounders = st.multiselect("3. Düzeltme Faktörlerini Ekleyin (Confounders)", 
                                 options=confounder_candidates,
                                 default=default_confounders)

    st.markdown("---")

    if st.button("Regresyon Modelini Kur ve Çiz"):
        # 1. FORMÜLÜ OLUŞTUR
        # Patsy formülü: "SII ~ PERIOD + AGE + BMI..."
        all_covars = [main_factor] + confounders
        formula_str = f"{target} ~ {' + '.join(all_covars)}"
        
        st.info(f"**Kurulan Model:** `{formula_str}`")
        
        try:
            # 2. VERİYİ HAZIRLA (Eksikleri At)
            model_data = df_f[[target] + all_covars].dropna()
            n_used = len(model_data)
            
            if n_used < 10:
                st.error(f"Hata: Analiz için yeterli veri kalmadı (N={n_used}). Seçilen değişkenlerde çok fazla eksik (NaN) veri olabilir.")
            else:
                # 3. MODELİ ÇALIŞTIR
                model = smf.ols(formula_str, data=model_data).fit()
                
                # 4. SONUÇ TABLOSU
                st.subheader(f"📊 Model Sonuçları (N={n_used})")
                st.code(model.summary().as_text())
                
                # 5. GRAFİK (FOREST PLOT / KATSAYI GRAFİĞİ)
                st.subheader("📈 Katsayı Etki Grafiği (Forest Plot)")
                st.caption("Nokta: Katsayı Değeri (Coef) | Çizgi: %95 Güven Aralığı. Çizgi 0 noktasını kesiyorsa sonuç anlamsızdır.")
                
                # Grafik verisini hazırla
                err_series = model.conf_int()
                err_series.columns = ['Lower', 'Upper']
                err_series['Coef'] = model.params
                
                # Intercept genelde grafiği bozar, onu çıkarıyoruz
                plot_data = err_series.drop("Intercept", errors="ignore")
                
                if not plot_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # --- DÜZELTİLEN KISIM: MANUEL ÇİZİM DÖNGÜSÜ ---
                    # errorbar yerine hlines ve plot kullanarak hatayı engelliyoruz
                    
                    # Y ekseni için pozisyonlar
                    y_pos = range(len(plot_data))
                    yticks = []
                    yticklabels = []
                    
                    for i, (idx, row) in enumerate(plot_data.iterrows()):
                        # Rengi belirle
                        if row['Lower'] > 0: 
                            c = 'firebrick'   # Pozitif ve Anlamlı (Kırmızı)
                        elif row['Upper'] < 0: 
                            c = 'steelblue'   # Negatif ve Anlamlı (Mavi)
                        else: 
                            c = 'gray'        # Anlamsız (Gri)
                        
                        # 1. Çizgiyi Çiz (Güven Aralığı)
                        ax.hlines(y=i, xmin=row['Lower'], xmax=row['Upper'], color=c, linewidth=2)
                        
                        # 2. Noktayı Koy (Katsayı)
                        ax.plot(row['Coef'], i, marker='o', color=c, markersize=8, markeredgecolor='black')
                        
                        # Etiketleri sakla
                        yticks.append(i)
                        yticklabels.append(idx)
                    
                    # Eksen Ayarları
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)
                    
                    # 0 Noktasına Referans Çizgisi
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                    
                    ax.set_xlabel(f"{target} Change (Unit)")
                    ax.set_title(f"Independent Effect of Factors on {target}")
                    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
                    
                    st.pyplot(fig)
                    import io
                    buf = io.BytesIO()
                    # dpi=300: Baskı kalitesi (High Resolution)
                    # bbox_inches='tight': Kenar boşluklarını otomatik kırpar
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    
                    st.download_button(
                        label="📥 Grafiği İndir (300 DPI - PNG)",
                        data=buf.getvalue(),
                        file_name="grafik_yuksek_cozunurluk.png", # İsim değiştirebilirsiniz
                        mime="image/png"
                    )
                else:
                    st.warning("Grafik çizilecek katsayı bulunamadı (Sadece Intercept var).")
                    
# ... (Üst tarafta Forest Plot kodlarınız var) ...
                
                # --- YENİ EKLENECEK KISIM BAŞLANGICI: DÜZELTİLMİŞ ORTALAMALAR ---
                st.markdown("---")
                st.subheader("⚖️ Heterojenlik Düzeltmesi: Düzeltilmiş Ortalamalar (Adjusted Means)")
                st.info(f"Aşağıdaki grafik, **{target}** üzerindeki demografik farkları (Irk, Yaş, Cinsiyet vb.) matematiksel olarak eşitleyerek **{main_factor}** değişkeninin 'saf' etkisini gösterir. Hakem eleştirisi için bu grafik kullanılır.")

                # Sadece Ana Faktör Kategorik ise (Örn: PERIOD) bu grafiği çiz
                # Eğer sayısal bir X seçtiyseniz (Örn: BMI) bu grafik mantıklı olmaz.
                is_categorical_factor = (model_data[main_factor].dtype == 'object') or (len(model_data[main_factor].unique()) < 10)

                if is_categorical_factor:
                    # 1. Yapay Veri Seti Oluştur (Confounder'ları Sabitle)
                    adj_data = model_data.copy()
                    
                    # Confounder'ları (Karıştırıcıları) ortalama veya mod değerine sabitle
                    for c in confounders:
                        if pd.api.types.is_numeric_dtype(adj_data[c]):
                            mean_val = adj_data[c].mean()
                            adj_data[c] = mean_val
                        else:
                            mode_val = adj_data[c].mode()[0]
                            adj_data[c] = mode_val
                    
                    # 2. Ana Faktörün her seviyesi için tahmin yap
                    levels = sorted(model_data[main_factor].unique())
                    adj_means = []
                    
                    for lvl in levels:
                        temp_df = adj_data.copy()
                        temp_df[main_factor] = lvl # Herkesi bu gruba ata
                        pred_mean = model.predict(temp_df).mean() # Tahmin et ve ortalamasını al
                        adj_means.append(pred_mean)
                    
                    # 3. Bar Grafiği Çiz
                    fig_adj, ax_adj = plt.subplots(figsize=(8, 6))
                    # Renk paleti
                    bar_colors = sns.color_palette("muted", len(levels))
                    
                    bars = ax_adj.bar(levels, adj_means, color=bar_colors, alpha=0.9, edgecolor='black')
                    
                    # Barların üzerine değerleri yaz
                    for bar in bars:
                        height = bar.get_height()
                        ax_adj.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.2f}',
                                    ha='center', va='bottom', fontsize=12, fontweight='bold')

                    ax_adj.set_ylabel(f"Düzeltilmiş {target} Ortalaması")
                    ax_adj.set_xlabel(main_factor)
                    ax_adj.set_title(f"Kovaryatlara Göre Düzeltilmiş Etki\n(Sabitlenenler: {', '.join(confounders)})")
                    ax_adj.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    st.pyplot(fig_adj)
                    
                    # Yorum
                    diff = adj_means[-1] - adj_means[0]
                    st.success(f"**Yorum:** Gruplar arasındaki demografik farklar (Irk, Yaş vb.) eşitlendiğinde bile, **{levels[-1]}** grubu **{levels[0]}** grubuna göre ortalama **{diff:.2f}** birim fark göstermektedir.")
                
                else:
                    st.warning(f"Seçilen Ana Faktör ({main_factor}) sayısal olduğu için Düzeltilmiş Ortalama Bar Grafiği çizilmedi.")
                # --- YENİ EKLENECEK KISIM BİTİŞİ ---

        except Exception as e:
            st.error(f"Model Hatası: {e}")
            st.warning("İpucu: Seçilen değişkenlerin veri tiplerini kontrol edin. Sayısal olmayan veriler (String) modelde otomatik kategoriye çevrilir.")
