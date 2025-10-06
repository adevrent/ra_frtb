
# app.py (paths fixed via environment variables)
import os
import io
import runpy
import tempfile
import contextlib
import types
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Deposit Analytics Suite", layout="wide")
st.title("📊 Deposit Analytics Suite")

st.write(
    "Analiz türünü seçin, **Turkey_Holidays.xlsx** ve ilgili **DATA** dosyasını yükleyin. "
    "Uygulama scripti kendi akışında çalıştırır ve grafik/tablo çıktısını ekranda gösterir."
)

with st.sidebar:
    st.header("⚙️ Ayarlar")
    analysis_choice = st.radio(
        "Analiz Türü",
        ["Core Analysis", "Early Withdrawal Analysis", "Prepayment Analysis"],
        index=0,
    )

    holidays_file = st.file_uploader("Turkey_Holidays.xlsx yükleyin", type=["xlsx"])

    data_label = {
        "Core Analysis": "CORE_DEPOSIT_DATA.xlsx",
        "Early Withdrawal Analysis": "EARLY_WITHDRAWAL_DATA.xlsx",
        "Prepayment Analysis": "PREPAYMENT_DATA.xlsx",
    }[analysis_choice]

    data_file = st.file_uploader(f"{data_label} yükleyin", type=["xlsx"])

    run_btn = st.button("🚀 Analizi Çalıştır")

SCRIPT_PATHS = {
    "Core Analysis": "core_analysis.py",
    "Early Withdrawal Analysis": "early_withdrawal_analysis.py",
    "Prepayment Analysis": "prepayment_analysis.py",
}

st.info(
    "Bu sürüm, scriptlerdeki **sabit Windows path** kullanımını kaldırır. "
    "Yüklediğiniz dosyalar için ortam değişkenleri ayarlanır ve scriptler bu yolları kullanır."
)

# --- Helpers to capture display/fig outputs ---
def _patch_ipython_display(captured_html: List[str]):
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")

    class HTML:
        def __init__(self, data): self.data = data
        def __str__(self): return str(self.data)
    class Markdown:
        def __init__(self, data): self.data = data
        def __str__(self): return str(self.data)

    def display(obj):
        s = getattr(obj, "data", obj)
        captured_html.append(str(s))

    disp.HTML = HTML
    disp.Markdown = Markdown
    disp.display = display
    ipy.display = disp

    import sys
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = disp

def _patch_plotly_capture(captured_figs: List[object]):
    try:
        import plotly.graph_objects as go
        def patched_show(self, *args, **kwargs):
            captured_figs.append(self)
            return None
        go.Figure.show = patched_show  # type: ignore
    except Exception:
        pass

def _patch_matplotlib_noop():
    try:
        import matplotlib.pyplot as plt
        def _noop_show(*args, **kwargs): return None
        plt.show = _noop_show  # type: ignore
    except Exception:
        pass

def run_script(script_path: str, holidays_path: str, data_path: str, analysis_choice: str):
    # Set environment variables expected by the scripts
    os.environ["TURKEY_HOLIDAYS_PATH"] = holidays_path
    if analysis_choice == "Core Analysis":
        os.environ["CORE_DEPOSIT_DATA_PATH"] = data_path
    elif analysis_choice == "Early Withdrawal Analysis":
        os.environ["EARLY_WITHDRAWAL_DATA_PATH"] = data_path
    else:
        os.environ["PREPAYMENT_DATA_PATH"] = data_path

    captured_html: List[str] = []
    captured_figs: List[object] = []

    _patch_ipython_display(captured_html)
    _patch_plotly_capture(captured_figs)
    _patch_matplotlib_noop()

    with contextlib.redirect_stdout(io.StringIO()):
        globs = runpy.run_path(script_path, run_name="__main__")

    return globs, captured_figs, captured_html

def _display_results(globs: Dict[str, Any], captured_figs: List[object], captured_html: List[str]):
    # 1) Try to find a plotly Figure either as global "fig" or via captured .show()
    fig = globs.get("fig", None)
    if fig is None and captured_figs:
        fig = captured_figs[-1]
    if fig is not None:
        st.subheader("📈 Grafik")
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly grafiği çizilemedi: {e}")

    # 2) Any DataFrames named table_a/b/c or *analysis*/*summary*/*result*
    tables = {}
    for key, val in globs.items():
        if isinstance(val, pd.DataFrame):
            name_low = key.lower()
            if key in ("table_a", "table_b", "table_c") or any(tok in name_low for tok in ("analysis", "summary", "result")):
                tables[key] = val

    # Also parse HTML tables captured via display(HTML(...))
    for html in captured_html:
        if "<table" in html.lower():
            try:
                dfs = pd.read_html(html)
                for i, df in enumerate(dfs, start=1):
                    tables[f"html_table_{i}_{len(tables)+i}"] = df
            except Exception:
                pass

    if tables:
        st.subheader("🧾 Tablolar")
        for name, df in tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Gösterilecek tablo yakalanamadı. (Script tablo üretmemiş olabilir.)")

if run_btn:
    if not holidays_file or not data_file:
        st.error("Lütfen **Turkey_Holidays.xlsx** ve seçtiğiniz analize ait **DATA** dosyasını yükleyin.")
    else:
        script_file = SCRIPT_PATHS[analysis_choice]
        if not os.path.exists(script_file):
            st.error(f"Script bulunamadı: {script_file}. `app.py` ile aynı klasörde olmalı.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                h_path = os.path.join(tmpdir, "Turkey_Holidays.xlsx")
                d_path = os.path.join(tmpdir, os.path.basename(data_label))
                with open(h_path, "wb") as f: f.write(holidays_file.getbuffer())
                with open(d_path, "wb") as f: f.write(data_file.getbuffer())

                st.success("Analiz başlatıldı. Script çalıştırılıyor...")
                try:
                    globs, cap_figs, cap_html = run_script(script_file, h_path, d_path, analysis_choice)
                    _display_results(globs, cap_figs, cap_html)
                    st.success("Analiz tamamlandı.")
                except Exception as e:
                    st.error(f"Çalışma sırasında bir hata oluştu:\n\n{e}")
