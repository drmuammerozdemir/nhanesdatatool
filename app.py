# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="NHANES XPT Batch Merge → CSV", layout="wide")
st.title("NHANES Pre vs Post: Tek Seferde XPT Yükle → Birleştir → CSV")

st.write(
    "Tüm .xpt dosyalarını **tek seferde** yükle. Uygulama dosya adlarına göre otomatik ayırır ve SEQN ile birleştirir."
)

# ----------------------------
# Helpers
# ----------------------------
def read_xpt(uploaded_file) -> pd.DataFrame:
    """Read NHANES .XPT (SAS transport) file into pandas DataFrame."""
    df = pd.read_sas(uploaded_file, format="xport", encoding="utf-8")

    # bytes -> str
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x
            )

    df.columns = [str(c).upper() for c in df.columns]
    return df


def pick_first_existing(dfs: dict, keys: list[str]) -> pd.DataFrame | None:
    """Return first df from dfs dict whose key exists and df not None."""
    for k in keys:
        if k in dfs and dfs[k] is not None and not dfs[k].empty:
            return dfs[k]
    return None


def merge_cycle(demo: pd.DataFrame, crp: pd.DataFrame, cbc: pd.DataFrame, period_label: str) -> pd.DataFrame:
    for name, df in [("DEMO", demo), ("CRP", crp), ("CBC", cbc)]:
        if df is None or df.empty:
            raise ValueError(f"{name} dosyası bulunamadı/boş.")
        if "SEQN" not in df.columns:
            raise ValueError(f"{name} dosyasında SEQN yok.")

    demo = demo.drop_duplicates(subset=["SEQN"])
    crp = crp.drop_duplicates(subset=["SEQN"])
    cbc = cbc.drop_duplicates(subset=["SEQN"])

    merged = demo.merge(crp, on="SEQN", how="inner", suffixes=("", "_CRP"))
    merged = merged.merge(cbc, on="SEQN", how="inner", suffixes=("", "_CBC"))
    merged["PERIOD"] = period_label
    return merged


def find_crp_column(df: pd.DataFrame) -> str | None:
    """Try to find CRP variable column name (usually LBXCRP)."""
    candidates = ["LBXCRP", "LBDCRP", "CRP", "HSCRP", "LBXCRP_SI"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any column containing 'CRP'
    for c in df.columns:
        if "CRP" in c:
            return c
    return None


def normalize_file_key(name: str) -> str:
    """
    Convert uploaded filename to a canonical key for matching.
    Examples:
      P_DEMO.xpt -> P_DEMO
      DEMO_L.xpt -> DEMO_L
      HSCRP_L.xpt -> HSCRP_L
    """
    base = name.rsplit(".", 1)[0]
    return base.upper()


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader(
    "Tüm NHANES .xpt dosyalarını buraya tek seferde bırak (multi-upload)",
    type=["xpt"],
    accept_multiple_files=True
)

period_pre_label = st.text_input("Pre etiketi", value="Pre-pandemic (2017-2018)")
period_post_label = st.text_input("Post etiketi", value="Post-pandemic (2021-2023)")

with st.expander("Opsiyonel filtreler", expanded=False):
    apply_age_filter = st.checkbox("Sadece ≥18 yaş (RIDAGEYR varsa)", value=True)
    apply_preg_excl = st.checkbox("Gebeleri hariç tut (RIDEXPRG varsa)", value=False)

run = st.button("Birleştir ve CSV hazırla", type="primary")

if uploaded:
    st.markdown("### Yüklenen dosyalar")
    st.write([f.name for f in uploaded])

if run:
    try:
        if not uploaded:
            raise ValueError("Önce .xpt dosyalarını yükle.")

        # Read all files into dict
        with st.spinner("Dosyalar okunuyor..."):
            dfs: dict[str, pd.DataFrame] = {}
            for f in uploaded:
                key = normalize_file_key(f.name)
                dfs[key] = read_xpt(f)

        # Expected keys (we accept both BIOPRO and HSCRP variants)
        expected = {
            "PRE":  ["P_DEMO", "P_CBC", "P_HSCRP", "P_BIOPRO"],
            "POST": ["DEMO_L", "CBC_L", "HSCRP_L", "BIOPRO_L"]
        }

        st.markdown("### Otomatik eşleştirme sonucu")
        pre_demo = pick_first_existing(dfs, ["P_DEMO"])
        pre_cbc  = pick_first_existing(dfs, ["P_CBC"])

        # CRP: prefer HSCRP, else BIOPRO
        pre_crp  = pick_first_existing(dfs, ["P_HSCRP", "P_HSCRP", "P_BIOPRO"])
        post_demo = pick_first_existing(dfs, ["DEMO_L"])
        post_cbc  = pick_first_existing(dfs, ["CBC_L"])
        post_crp  = pick_first_existing(dfs, ["HSCRP_L", "BIOPRO_L"])

        # Show missing info
        def present(k): return "✅" if (k in dfs and dfs[k] is not None and not dfs[k].empty) else "❌"

        colA, colB = st.columns(2)
        with colA:
            st.write("**Pre (P_*)**")
            st.write("P_DEMO:", present("P_DEMO"))
            st.write("P_CBC:", present("P_CBC"))
            st.write("P_HSCRP:", present("P_HSCRP") or present("P_HSCRP"))  # harmless
            st.write("P_HSCRP / P_HSCRP:", "✅" if ("P_HSCRP" in dfs and not dfs["P_HSCRP"].empty) or ("P_HSCRP" in dfs and not dfs["P_HSCRP"].empty) else "❌")
            st.write("P_BIOPRO (fallback):", present("P_BIOPRO"))
        with colB:
            st.write("**Post (*_L)**")
            st.write("DEMO_L:", present("DEMO_L"))
            st.write("CBC_L:", present("CBC_L"))
            st.write("HSCRP_L:", present("HSCRP_L"))
            st.write("BIOPRO_L (fallback):", present("BIOPRO_L"))

        if pre_demo is None or pre_cbc is None:
            raise ValueError("Pre tarafında DEMO veya CBC eksik. (P_DEMO.xpt ve P_CBC.xpt gerekli)")
        if post_demo is None or post_cbc is None:
            raise ValueError("Post tarafında DEMO veya CBC eksik. (DEMO_L.xpt ve CBC_L.xpt gerekli)")
        if pre_crp is None:
            raise ValueError("Pre tarafında CRP dosyası yok. (P_HSCRP.xpt veya P_BIOPRO.xpt gerekli)")
        if post_crp is None:
            raise ValueError("Post tarafında CRP dosyası yok. (HSCRP_L.xpt veya BIOPRO_L.xpt gerekli)")

        # Ensure CRP column exists
        pre_crp_col = find_crp_column(pre_crp)
        post_crp_col = find_crp_column(post_crp)
        if pre_crp_col is None:
            raise ValueError("Pre CRP dosyasında CRP değişkeni bulunamadı (LBXCRP beklenir).")
        if post_crp_col is None:
            raise ValueError("Post CRP dosyasında CRP değişkeni bulunamadı (LBXCRP beklenir).")

        # If column name differs, standardize to LBXCRP for downstream consistency
        if pre_crp_col != "LBXCRP":
            pre_crp = pre_crp.rename(columns={pre_crp_col: "LBXCRP"})
        if post_crp_col != "LBXCRP":
            post_crp = post_crp.rename(columns={post_crp_col: "LBXCRP"})

        # Merge
        with st.spinner("SEQN ile birleştiriliyor..."):
            pre_merged = merge_cycle(pre_demo, pre_crp[["SEQN", "LBXCRP"]], pre_cbc, period_pre_label)
            post_merged = merge_cycle(post_demo, post_crp[["SEQN", "LBXCRP"]], post_cbc, period_post_label)

        # Optional filters
        def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if apply_age_filter and "RIDAGEYR" in out.columns:
                out = out[out["RIDAGEYR"].notna() & (out["RIDAGEYR"] >= 18)]
            if apply_preg_excl and "RIDEXPRG" in out.columns:
                out = out[~(out["RIDEXPRG"] == 1)]
            return out

        pre_merged = apply_filters(pre_merged)
        post_merged = apply_filters(post_merged)

        st.success("✅ Birleştirme tamam! (Pre ve Post ayrı CSV)")
      
        c1, c2 = st.columns(2)
        c1.metric("Pre satır", f"{len(pre_merged):,}")
        c2.metric("Post satır", f"{len(post_merged):,}")

        # Pre preview + download
        st.markdown("### Pre-pandemic örnek (ilk 20 satır)")
        st.dataframe(pre_merged.head(20), use_container_width=True)

        pre_csv_bytes = pre_merged.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Pre CSV indir",
            data=pre_csv_bytes,
            file_name="nhanes_pre_pandemic.csv",
            mime="text/csv"
        )

        st.divider()

        # Post preview + download
        st.markdown("### Post-pandemic örnek (ilk 20 satır)")
        st.dataframe(post_merged.head(20), use_container_width=True)

        post_csv_bytes = post_merged.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Post CSV indir",
            data=post_csv_bytes,
            file_name="nhanes_post_pandemic.csv",
            mime="text/csv"
        )


st.divider()

# Post preview + download
st.markdown("### Post-pandemic örnek (ilk 20 satır)")
st.dataframe(post_merged.head(20), use_container_width=True)

post_csv_bytes = post_merged.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Post CSV indir",
    data=post_csv_bytes,
    file_name="nhanes_post_pandemic.csv",
    mime="text/csv"
)


        # Download
        csv_bytes = master.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Master CSV indir",
            data=csv_bytes,
            file_name="nhanes_pre_post_master.csv",
            mime="text/csv"
        )

        # Optional: also show which columns exist in CBC if user wants
        with st.expander("CBC sütunlarının tamamını göster", expanded=False):
            st.write("Pre CBC columns:", list(pre_cbc.columns))
            st.write("Post CBC columns:", list(post_cbc.columns))

    except Exception as e:
        st.error(f"Hata: {e}")

st.divider()
st.subheader("Lokal çalıştırma")
st.code(
"""pip install streamlit pandas pyreadstat
streamlit run app.py
""",
language="bash"
)

