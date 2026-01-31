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
st.title("NHANES Pre vs Post – Detaylı Analiz (SII Bazlı Filtre)")

st.write(
    "Bu analizde **sadece SII (Systemic Immune-Inflammation Index) hesaplanabilen** katılımcılar dahil edilmiştir. "
    "Tablo altında kullanılan test metodolojisi açıklanmıştır."
)

# ---------------------------
# 1. HARİTALAMA
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
    
    # --- VÜCUT ÖLÇÜMLERİ ---
    "BMXWT":    "WEIGHT_KG",
    "BMXHT":    "HEIGHT_CM",
    "BMXBMI":   "BMI",
    "BMXWAIST": "WAIST_CM",
    "BMXHIP":   "HIP_CM",

    # --- SİGARA DEĞİŞKENLERİ ---
    "SMQ020": "SMOKE_LIFE_100",  # Hayatında 100+ içti mi?
    "SMQ040": "SMOKE_NOW",       # Şu an içiyor mu?
    "SMD030": "AGE_STARTED",     # Kaç yaşında düzenli içmeye başladı?
    "SMD650": "CIGS_PER_DAY_NOW",# (Aktif) Günde ortalama kaç tane içiyor?
    "SMD057": "CIGS_PER_DAY_QUIT",# (Bırakan) Bıraktığında kaç içiyordu?
    "SMQ050Q": "TIME_SINCE_QUIT", # Ne kadar süredir içmiyor? (Sayı)
    "SMQ050U": "UNIT_SINCE_QUIT", # Birim (1=Gün, 2=Hafta, 3=Ay, 4=Yıl)

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
    # Sigara
    "SMQ020": "SMOKE_LIFE",  # Hayatında 100 tane içti mi? (1=Evet, 2=Hayır)
    "SMQ040": "SMOKE_NOW",   # Şu an içiyor mu? (1=Her gün, 2=Bazen, 3=Hiç),
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
    """Bizim kuralımız: Anlamlıysa < sembolü, değilse gerçek değer."""
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}"

def check_normality(data):
    """Shapiro-Wilk p değerini döndürür."""
    try:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if len(data) < 3:
            return np.nan
        stat, p = shapiro(data)
        return p
    except:
        return np.nan

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    NLR, PLR, SII, SIRI, LMR, AISI, dNLR, MLR, NMLR hesaplar.
    """
    out = ensure_upper_cols(df.copy())
    
    # 1. Gerekli tüm sütunları sayıya çevir (LBXWBCSI Eklendi!)
    numeric_candidates = [
        "LBDNENO", "LBDLYMNO", "LBDMONO", "LBXPLTSI", "LBXWBCSI",
        "LBXNEPCT", "LBXLYPCT", "LBXMOPCT"
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 2. Mutlak Sayılar (Absolute Counts)
    neut_abs = out["LBDNENO"] if "LBDNENO" in out.columns else None
    lymph_abs = out["LBDLYMNO"] if "LBDLYMNO" in out.columns else None
    mono_abs  = out["LBDMONO"] if "LBDMONO" in out.columns else None
    plt_count = out["LBXPLTSI"] if "LBXPLTSI" in out.columns else None
    wbc_count = out["LBXWBCSI"] if "LBXWBCSI" in out.columns else None # dNLR için lazım

    # 3. Yüzdeler (Percentages) - Yedek
    neut_pct = out["LBXNEPCT"] if "LBXNEPCT" in out.columns else None
    lymph_pct = out["LBXLYPCT"] if "LBXLYPCT" in out.columns else None
    mono_pct  = out["LBXMOPCT"] if "LBXMOPCT" in out.columns else None

    # --- HESAPLAMA ---
    
    # A) Mutlak Sayılar Varsa (En Hassas Yöntem)
    if neut_abs is not None and lymph_abs is not None:
        out["NLR"] = neut_abs / lymph_abs.replace(0, np.nan)
        out["INDEX_MODE"] = "absolute (LBD)"
        
        # PLR, SII
        if plt_count is not None:
            out["PLR"] = plt_count / lymph_abs.replace(0, np.nan)
            out["SII"] = (plt_count * neut_abs) / lymph_abs.replace(0, np.nan)
        
        # Monosit varsa: LMR, SIRI, MLR, NMLR, AISI
        if mono_abs is not None:
            out["LMR"] = lymph_abs / mono_abs.replace(0, np.nan)  # Lymph / Mono
            out["MLR"] = mono_abs / lymph_abs.replace(0, np.nan)  # Mono / Lymph (YENİ)
            out["SIRI"] = (neut_abs * mono_abs) / lymph_abs.replace(0, np.nan)
            
            # NMLR = (Neut + Mono) / Lymph (YENİ)
            out["NMLR"] = (neut_abs + mono_abs) / lymph_abs.replace(0, np.nan)

            if plt_count is not None:
                out["AISI"] = (neut_abs * plt_count * mono_abs) / lymph_abs.replace(0, np.nan)
        
        # dNLR (YENİ) -> Neut / (WBC - Neut)
        if wbc_count is not None:
            denom = wbc_count - neut_abs
            out["dNLR"] = neut_abs / denom.replace(0, np.nan)

    # B) Sadece Yüzdeler Varsa (Yedek)
    elif neut_pct is not None and lymph_pct is not None:
        out["NLR"] = neut_pct / lymph_pct.replace(0, np.nan)
        out["INDEX_MODE"] = "percent (LBX)"
        
        if mono_pct is not None:
            out["LMR"] = lymph_pct / mono_pct.replace(0, np.nan)
            out["MLR"] = mono_pct / lymph_pct.replace(0, np.nan) # YENİ
            out["NMLR"] = (neut_pct + mono_pct) / lymph_pct.replace(0, np.nan) # YENİ
        
        # dNLR yüzdelerle hesaplanabilir mi? Teorik olarak Neut% / (100 - Neut%) ama WBC sayısı olmadan riskli.
        # PLR, SII, AISI, dNLR, SIRI için mutlak sayı şart diyoruz.
        out["PLR"] = np.nan; out["SII"] = np.nan; out["AISI"] = np.nan
        out["SIRI"] = np.nan; out["dNLR"] = np.nan
        
    else:
        # Hiçbiri yoksa
        cols_to_nan = ["NLR", "PLR", "SII", "LMR", "MLR", "AISI", "SIRI", "dNLR", "NMLR"]
        for c in cols_to_nan:
            out[c] = np.nan
        out["INDEX_MODE"] = "NA"
# --- SİGARA DURUMU SINIFLANDIRMA ---
    # Eğer gerekli sütunlar varsa hesapla
    if "SMOKE_LIFE" in out.columns and "SMOKE_NOW" in out.columns:
        s_life = out["SMOKE_LIFE"] # SMQ020
        s_now = out["SMOKE_NOW"]   # SMQ040
        
        # Kategorileri Belirle
        conditions = [
            (s_life == 2),  # 2: Hayır (Hiç 100 tane içmemiş) -> Never Smoker
            (s_life == 1) & (s_now == 3), # 1:Evet içmiş AMA 3:Artık içmiyor -> Former Smoker
            (s_life == 1) & ((s_now == 1) | (s_now == 2)) # 1:Evet içmiş VE (1 veya 2):Hala içiyor -> Current Smoker
        ]
        choices = ["Never Smoker", "Former Smoker", "Current Smoker"]
        
        out["SMOKING"] = np.select(conditions, choices, default=np.nan)
        out["SMOKING"] = out["SMOKING"].replace("nan", np.nan)
    else:
        out["SMOKING"] = np.nan

    # --- SİGARA & PACK-YEARS HESAPLAMA ---
    
    # Gerekli sütunlar var mı kontrol et
    req_cols = ["SMOKE_LIFE_100", "SMOKE_NOW", "AGE", "AGE_STARTED"]
    if all(col in out.columns for col in req_cols):
        
        # 1. SMOKING BINARY (Yes/No) - Frekans için
        # Yes = Current Smoker, No = Never + Former
        # SMQ040: 1=Every day, 2=Some days -> Current
        is_current = (out["SMOKE_LIFE_100"] == 1) & (out["SMOKE_NOW"].isin([1, 2]))
        out["SMOKING_STATUS"] = np.where(is_current, "Yes (Current)", "No (Former/Never)")
        
        # 2. PACK-YEARS HESAPLAMA
        # Önce boş bir sütun oluştur
        out["PACK_YEARS"] = np.nan
        
        # A) HİÇ İÇMEYENLER (Pack-Year = 0)
        # SMQ020 == 2 (Hayır)
        out.loc[out["SMOKE_LIFE_100"] == 2, "PACK_YEARS"] = 0
        
        # B) AKTİF İÇİCİLER (Current)
        # Süre = (Şu anki Yaş - Başlama Yaşı)
        # Adet = SMD650 (Son 30 gündeki ortalama)
        if "CIGS_PER_DAY_NOW" in out.columns:
            mask_curr = (out["SMOKE_STATUS"] == "Yes (Current)") if "SMOKE_STATUS" in out.columns else is_current
            
            duration = out["AGE"] - out["AGE_STARTED"]
            packs_per_day = out["CIGS_PER_DAY_NOW"] / 20
            
            # Negatif süre çıkarsa (veri hatası) 0 yap
            duration = duration.clip(lower=0)
            
            out.loc[mask_curr, "PACK_YEARS"] = duration * packs_per_day

        # C) ESKİ İÇİCİLER (Former)
        # Bu biraz karışık çünkü "Ne zaman bıraktı?" sorusunun birimi değişiyor.
        # SMQ050U: 1=Gün, 2=Hafta, 3=Ay, 4=Yıl
        if "CIGS_PER_DAY_QUIT" in out.columns and "TIME_SINCE_QUIT" in out.columns and "UNIT_SINCE_QUIT" in out.columns:
            mask_former = (out["SMOKE_LIFE_100"] == 1) & (out["SMOKE_NOW"] == 3)
            
            # Bırakma süresini YIL cinsine çevirelim
            q_val = out["TIME_SINCE_QUIT"]
            q_unit = out["UNIT_SINCE_QUIT"]
            
            years_quit = pd.Series(np.zeros(len(out)), index=out.index)
            years_quit[q_unit == 1] = q_val / 365.25  # Gün
            years_quit[q_unit == 2] = q_val / 52.14   # Hafta
            years_quit[q_unit == 3] = q_val / 12.0    # Ay
            years_quit[q_unit == 4] = q_val           # Yıl
            
            # İçme Süresi = (Şimdiki Yaş - Bırakalı Geçen Yıl) - Başlama Yaşı
            duration = (out["AGE"] - years_quit) - out["AGE_STARTED"]
            duration = duration.clip(lower=0) # Eksi çıkarsa 0 yap
            
            packs_per_day = out["CIGS_PER_DAY_QUIT"] / 20
            
            # Hesapla ve Yaz
            out.loc[mask_former, "PACK_YEARS"] = duration * packs_per_day

    else:
        # Veri yoksa boş geç
        out["SMOKING_STATUS"] = np.nan
        out["PACK_YEARS"] = np.nan
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
            return f"{val:.2f} ± {q1_sd:.2f}" 
        else:
            return f"{val:.3g} [{q1_sd:.3g}–{q3_sd:.3g}]"
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
            color_line = 'red'
        else:
            center_stat = np.median(vals) if len(vals) else np.nan
            color_line = 'black'

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
# Sidebar: NAVİGASYON VE AYARLAR
# ---------------------------
st.sidebar.title("Menü & Ayarlar")

# 1. NAVİGASYON (SAYFA SEÇİMİ)
page = st.sidebar.radio(
    "Görüntülemek İstediğiniz Analiz:",
    ["1. Özet Tablo", "2. Grafikler", "3. Korelasyon", "4. Regresyon"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Filtreler")

# SII Filtresi (Sidebar'da tanımlı kalmalı)
filter_missing_sii = st.sidebar.checkbox(
    "Sadece SII hesaplanabilenleri dahil et", 
    value=True, 
    help="İşaretliyse; hemogram verisi eksik olan (SII=NaN) satırlar analizden tamamen atılır."
)

# Cinsiyet Filtresi
gender_filter = st.sidebar.radio("Cinsiyet:", ["Tümü", "Kadınlar (2)", "Erkekler (1)"], index=0)

# Yaş ve Diğerleri
age_min = st.sidebar.number_input("Min Yaş", 18, 120, 18)
age_max = st.sidebar.number_input("Max Yaş", 18, 120, 120)
exclude_crp_gt10 = st.sidebar.checkbox("CRP > 10 dışla", False)
log_transform_crp = st.sidebar.checkbox("Log(CRP) dönüşümü", True)
point_size = st.sidebar.slider("Grafik Nokta Boyutu", 3, 25, 9)
dpi_out = st.sidebar.selectbox("İndirme Kalitesi (DPI)", [150, 300, 600], 1)

# ---------------------------
# DEĞİŞKEN SEÇİMİ (SOL MENÜYE ALINDI)
# ---------------------------
# Sayfalar arası geçişte seçim kaybolmasın diye buraya sabitliyoruz.
st.sidebar.markdown("---")
st.sidebar.subheader("Değişken Seçimi")
default_candidates = ["CRP", "WBC", "NLR", "dNLR", "PLR", "SII", "SIRI", "LMR", "MLR", "NMLR", "AISI", "PLT", "AGE", "BMI", "WAIST_CM", "SMOKING_STATUS", "PACK_YEARS"]
# Veri setinde olanları bul
available_defaults = [c for c in default_candidates if c in df.columns]
# Tüm sütunları aday göster
all_columns = sorted(list(df.columns))

vars_to_analyze = st.sidebar.multiselect(
    "Analiz Edilecek Değişkenler", 
    options=all_columns, 
    default=available_defaults
)

if not vars_to_analyze:
    st.warning("Lütfen sol menüden en az bir değişken seçin.")
    st.stop()

# ---------------------------
# FİLTRELEME MANTIĞI (GLOBAL)
# ---------------------------
# Bu kısım sayfa seçiminden bağımsız her zaman çalışmalı
df_f = df.copy()

# 1. SII Eksiklik Filtresi
if filter_missing_sii:
    missing_rows = df_f[df_f["SII"].isna()]
    n_missing = len(missing_rows)
    df_f = df_f.dropna(subset=["SII"]) 
    
    # Uyarıyı sadece 'Özet Tablo' sayfasındaysak gösterelim ki diğer sayfaları kirletmesin
    if n_missing > 0 and page == "1. Özet Tablo":
        n_pre = len(missing_rows[missing_rows["PERIOD"].astype(str).str.contains("Pre", case=False, na=False)])
        n_post = len(missing_rows[missing_rows["PERIOD"].astype(str).str.contains("Post", case=False, na=False)])
        st.warning(f"⚠️ **{n_missing} kişi** SII verisi eksik olduğu için çıkarıldı (Pre: {n_pre}, Post: {n_post}).")
else:
    if page == "1. Özet Tablo":
        st.info(f"Tüm katılımcılar dahil. (N: {len(df_f)})")

# 2. Cinsiyet
if gender_filter == "Kadınlar (2)":
    if "SEX" in df_f.columns: df_f = df_f[df_f["SEX"] == 2]
elif gender_filter == "Erkekler (1)":
    if "SEX" in df_f.columns: df_f = df_f[df_f["SEX"] == 1]

# 3. Yaş
if "AGE" in df_f.columns:
    df_f = df_f[(df_f["AGE"] >= age_min) & (df_f["AGE"] <= age_max)]

# 4. CRP
if exclude_crp_gt10 and "CRP" in df_f.columns:
    df_f = df_f[~(pd.to_numeric(df_f["CRP"], errors="coerce") > 10)]
if log_transform_crp and "CRP" in df_f.columns:
    crp_num = pd.to_numeric(df_f["CRP"], errors="coerce").replace(0, np.nan)
    df_f["CRP_LOG10"] = np.log10(crp_num)

pre_f = df_f[df_f["PERIOD"].str.contains("pre", case=False, na=False)].copy()
post_f = df_f[df_f["PERIOD"].str.contains("post", case=False, na=False)].copy()

# Test Zorlama Ayarı (Sidebar'da en alta veya uygun yere koyabilirsiniz, burada global tanımlı olsun)
force_parametric = st.sidebar.checkbox("⚠️ Parametrik Teste Zorla", value=False)


# =========================================================
# SAYFA 1: ÖZET TABLO (KATEGORİK DESTEKLİ)
# =========================================================
if page == "1. Özet Tablo":
    st.header("1. Özet İstatistikler ve Hipotez Testleri")
    
    # Kategorik olarak işlem görecek değişkenleri tanımla
    categorical_vars = ["SEX", "RACE"]
    
    rows = []
    used_tests_info = set()

    for v in vars_to_analyze:
        # Veri kontrolü
        if v not in pre_f.columns or v not in post_f.columns: continue

        # --- DURUM A: KATEGORİK DEĞİŞKEN (SEX, RACE) ---
        if v in categorical_vars:
            # 1. Çapraz Tablo (Crosstab) Oluştur: Değişken vs Dönem
            # Sadece analizdeki değişken ve Period sütununu alıp temizleyelim
            cat_data = df_f[[v, "PERIOD"]].dropna()
            
            # Eğer veri yoksa atla
            if len(cat_data) < 10: continue

            # Crosstab (Satır: Değişkenin Değerleri, Sütun: Pre/Post)
            ct = pd.crosstab(cat_data[v], cat_data["PERIOD"])
            
            # Sütunlarda Pre ve Post olduğundan emin ol
            cols_needed = [c for c in ct.columns if "Pre" in str(c)] + [c for c in ct.columns if "Post" in str(c)]
            if len(cols_needed) < 2: continue # Pre veya Post eksikse geç
            
            ct_ordered = ct[cols_needed] # Sıralamayı garantiye al

            # 2. İstatistiksel Test: Chi-Square (Ki-Kare)
            chi2, p_val, dof, expected = chi2_contingency(ct_ordered)
            used_tests_info.add("Chi-Square")
            
            # 3. Etki Büyüklüğü: Cramer's V
            n_total = ct_ordered.sum().sum()
            min_dim = min(ct_ordered.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
            
            # 4. Tablo İçin Gösterim (Yüzdeler)
            # Pre Grubu Dağılımı
            # --- BU FONKSİYONU GÜNCELLEYİN ---
            def format_cat_dist(series):
                total = series.sum()
                if total == 0: return "NA"
                
                parts = []
                for val, count in series.items():
                    pct = 100 * count / total
                    
                    # Etiketleme (NHANES kodları: 1=Erkek, 2=Kadın)
                    label = str(val)
                    if v == "SEX":
                        label = "Male" if val == 1 else "Female"
                    
                    # İSTEDİĞİNİZ FORMAT: "Male n:1500 (%50.1)"
                    parts.append(f"**{label}** n:{count} (%{pct:.1f})")
                    
                return " / ".join(parts)
            # ---------------------------------

            col_pre = [c for c in ct.columns if "Pre" in str(c)][0]
            col_post = [c for c in ct.columns if "Post" in str(c)][0]
            
            pre_disp = format_cat_dist(ct[col_pre])
            post_disp = format_cat_dist(ct[col_post])
            
            # Değişim yüzdesi kategorik için anlamsızdır, yerine tire koyuyoruz
            pct_change = "—"
            
            # Tabloya Ekle
            rows.append({
                "Variable": v,
                "Pre (Ref)": pre_disp,
                "Post": post_disp,
                "% Change": pct_change,
                "Normality (P)": "N/A (Categorical)",
                "p (Raw)": p_label_detailed(p_val),
                "p (Age Adj.)": "—", # Kategorik için ANCOVA yerine Logistic Reg gerekir, şimdilik boş geçiyoruz
                "Cliff's δ": f"V={cramers_v:.2f}", # Cramer's V
                "n": f"{ct[col_pre].sum()}/{ct[col_post].sum()}"
            })

        # --- DURUM B: SAYISAL DEĞİŞKEN (MEVCUT KOD) ---
        else:
            pre_vals = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
            post_vals = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()

            if len(pre_vals) < 3 or len(post_vals) < 3: continue

            # Normallik
            p_sw_pre = check_normality(pre_vals)
            p_sw_post = check_normality(post_vals)
            is_normal = ((p_sw_pre > 0.05) and (p_sw_post > 0.05)) if (np.isfinite(p_sw_pre) and np.isfinite(p_sw_post)) else False
            
            # Test Seçimi
            use_parametric = force_parametric or is_normal
            
            if use_parametric:
                used_tests_info.add("Welch t-test")
                stat, p_raw = ttest_ind(pre_vals, post_vals, equal_var=False)
                pre_disp = format_val_disp(*mean_sd(pre_f[v]), 0, True)
                post_disp = format_val_disp(*mean_sd(post_f[v]), 0, True)
            else:
                used_tests_info.add("Mann-Whitney U")
                stat, p_raw = mannwhitneyu(pre_vals, post_vals, alternative="two-sided")
                pre_disp = format_val_disp(*median_iqr(pre_f[v]), False)
                post_disp = format_val_disp(*median_iqr(post_f[v]), False)

            # Değişim & Etki
            center_pre = np.mean(pre_vals) if use_parametric else np.median(pre_vals)
            center_post = np.mean(post_vals) if use_parametric else np.median(post_vals)
            pct_change = 100.0 * (center_post - center_pre) / center_pre if center_pre != 0 else np.nan
            delta = cliffs_delta(pre_vals, post_vals)

            # ANCOVA (Sayısal için)
            p_adj = np.nan
            if "AGE" in df_f.columns:
                try:
                    temp_df = df_f[[v, "PERIOD", "AGE"]].dropna()
                    temp_df.columns = ["Target", "Group", "Age"]
                    if len(temp_df) > 20:
                        model = smf.ols("Target ~ Group + Age", data=temp_df).fit()
                        p_keys = [k for k in model.pvalues.index if "Group" in k]
                        if p_keys: p_adj = model.pvalues[p_keys[0]]
                except: pass

            rows.append({
                "Variable": v,
                "Pre (Ref)": pre_disp,
                "Post": post_disp,
                "% Change": f"{pct_change:.1f}%",
                "Normality (P)": f"Pre {p_label_detailed(p_sw_pre)} / Post {p_label_detailed(p_sw_post)}",
                "p (Raw)": p_label_detailed(p_raw),
                "p (Age Adj.)": p_label_detailed(p_adj),
                "Cliff's δ": f"{delta:.3f}",
                "n": f"{len(pre_vals)}/{len(post_vals)}"
            })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
        st.caption(f"Kullanılan Testler: {', '.join(sorted(list(used_tests_info)))}. (Kategorik veriler için Chi-Square kullanılmıştır).")
        st.download_button("Tabloyu İndir (CSV)", summary.to_csv(index=False).encode("utf-8-sig"), "nhanes_table.csv", "text/csv")

# =========================================================
# SAYFA 2: GRAFİKLER
# =========================================================
elif page == "2. Grafikler":
    st.header("2. Karşılaştırmalı Grafikler")
    
    # Kullanıcıdan grafik için alt seçim isteyebiliriz veya hepsini basabiliriz
    # Burayı temiz tutmak için vars_to_analyze listesinden ilk 6 tanesini veya hepsini çizdirelim
    plot_vars = st.multiselect("Çizilecekleri Seçin:", vars_to_analyze, default=vars_to_analyze[:min(6, len(vars_to_analyze))])
    
    if plot_vars:
        n = len(plot_vars)
        cols_grid = int(np.ceil(np.sqrt(n)))
        rows_grid = int(np.ceil(n / cols_grid))
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 4.5, rows_grid * 4.5))
        if n == 1: axes = [axes]
        axes = np.array(axes).reshape(-1)

        for i, v in enumerate(plot_vars):
            ax = axes[i]
            pre_vals = pd.to_numeric(pre_f[v], errors="coerce").dropna().to_numpy()
            post_vals = pd.to_numeric(post_f[v], errors="coerce").dropna().to_numpy()
            
            p_sw_pre = check_normality(pre_vals)
            p_sw_post = check_normality(post_vals)
            is_norm = ((p_sw_pre > 0.05) and (p_sw_post > 0.05)) if (np.isfinite(p_sw_pre) and np.isfinite(p_sw_post)) else False
            use_para = force_parametric or is_norm
            
            if use_para:
                _, p_g = ttest_ind(pre_vals, post_vals, equal_var=False)
                title_end = "(T-test)"
            else:
                _, p_g = mannwhitneyu(pre_vals, post_vals)
                title_end = "(MWU)"

            stripplot_with_p(ax, [pre_vals, post_vals], ["Pre", "Post"], 
                             p_label_detailed(p_g),
                             title=f"{v} {title_end}", ylabel=v, point_size=point_size,
                             show_mean=use_para)

        for j in range(i + 1, len(axes)): axes[j].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=int(dpi_out), bbox_inches="tight")
        buf.seek(0)
        st.download_button(f"Grafiği İndir", buf, "nhanes_plots.png", "image/png")


# =========================================================
# SAYFA 3: KORELASYON
# =========================================================
elif page == "3. Korelasyon":
    st.header("3. Korelasyon Analizi (Heatmap)")
    st.write("Değişkenler arasındaki ilişki gücü (Spearman Rank Correlation).")

    # Sadece sayısal sütunları al
    corr_cols = [c for c in vars_to_analyze if c in df_f.select_dtypes(include=np.number).columns]
    
    if len(corr_cols) > 1:
        corr_matrix = df_f[corr_cols].corr(method="spearman")
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        try:
            import seaborn as sns
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr, square=True)
        except ImportError:
            cax = ax_corr.imshow(corr_matrix, cmap="coolwarm")
            fig_corr.colorbar(cax)
            ax_corr.set_xticks(range(len(corr_cols)))
            ax_corr.set_yticks(range(len(corr_cols)))
            ax_corr.set_xticklabels(corr_cols, rotation=90)
            ax_corr.set_yticklabels(corr_cols)
            for (j,i),label in np.ndenumerate(corr_matrix):
                ax_corr.text(i,j,f"{label:.2f}",ha='center',va='center')

        st.pyplot(fig_corr)
    else:
        st.warning("Korelasyon matrisi için en az 2 sayısal değişken seçmelisiniz.")


# =========================================================
# SAYFA 4: REGRESYON
# =========================================================
elif page == "4. Regresyon":
    st.header("4. Çok Değişkenli Regresyon Analizi")
    st.write(
        "Bu model; **Yaş, Cinsiyet, BMI ve Dönem (Pre/Post)** faktörlerinin, "
        "hedef değişken üzerindeki **bağımsız etkisini** ölçer."
    )
    
    # Hedef Değişken
    target_options = [v for v in vars_to_analyze if v not in ["AGE", "BMI", "WAIST_CM", "SEX", "PERIOD"]]
    target_var = st.selectbox("Bağımlı Değişkeni Seçin (Hedef):", 
                              options=target_options,
                              index=0 if target_options else None)
    
    if not target_var:
        st.error("Regresyon için uygun hedef değişken bulunamadı.")
        st.stop()

    # Kovaryatlar
    covariates = ["PERIOD", "AGE"]
    if "SEX" in df_f.columns: covariates.append("SEX")
    if "BMI" in df_f.columns: covariates.append("BMI")
    if "WAIST_CM" in df_f.columns: covariates.append("WAIST_CM")
    
    selected_covars = st.multiselect("Modele Dahil Edilecek Faktörler (X):", 
                                     options=["PERIOD", "AGE", "SEX", "BMI", "WAIST_CM", "RACE"],
                                     default=[c for c in covariates if c in df_f.columns])
    
    if st.button("Regresyon Modelini Kur"):
        formula_str = f"{target_var} ~ " + " + ".join(selected_covars)
        try:
            reg_df = df_f[[target_var] + selected_covars].dropna()
            model = smf.ols(formula_str, data=reg_df).fit()
            
            st.markdown(f"### Sonuç: {target_var} Analizi (N={len(reg_df)})")
            st.code(model.summary().as_text())
            
            st.info(
                "**İpucu:** 'PERIOD[T.Post]' veya benzeri satırların P değeri < 0.05 ise, "
                "diğer faktörler (Kilo, Yaş vb.) eşitlense bile Pandeminin etkisi anlamlıdır."
            )
        except Exception as e:
            st.error(f"Model hatası: {e}")
