# app.py
import os
import io
import sys
import types
import runpy
import tempfile
import contextlib
from typing import List

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Deposit Analytics Suite", layout="wide")
st.title("📊 Deposit Analytics Suite")

st.write(
    "Analiz türünü seçin, **Turkey_Holidays.xlsx** ve ilgili **DATA** dosyasını yükleyin. "
    "Uygulama scripti kendi akışında çalıştırır ve grafik/tablo çıktısını ekranda gösterir."
)

# --- Sidebar ---
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

# Script adları (aynı klasörde)
SCRIPT_PATHS = {
    "Core Analysis": "core_analysis.py",
    "Early Withdrawal Analysis": "early_withdrawal_analysis.py",
    "Prepayment Analysis": "prepayment_analysis.py",
}

st.info(
    "Bu sürüm, scriptlerdeki **mutlak Excel yollarını** otomatik olarak yüklediğiniz dosyalara yönlendirir; "
    "yani `C:\\...\\CORE_DEPOSIT_DATA.xlsx` gibi path'ler sorun çıkarmaz."
)

def _save_uploaded(tmpdir: str, up, filename: str) -> str:
    path = os.path.join(tmpdir, filename)
    with open(path, "wb") as f:
        f.write(up.getbuffer())
    return path

def _build_pandas_shim(path_map: dict):
    """
    pandas.read_excel'i sarmalayan bir shim oluşturur.
    path_map: {'CORE_DEPOSIT_DATA.xlsx': '/tmp/.../CORE_DEPOSIT_DATA.xlsx', ...}
    """
    import pandas as _real_pd
    shim = types.ModuleType("pandas")

    # Tüm öznitelikleri gerçek pandas'tan kopyala
    for name in dir(_real_pd):
        setattr(shim, name, getattr(_real_pd, name))

    real_read_excel = _real_pd.read_excel

    def redirected_read_excel(io_arg, *args, **kwargs):
        # io_arg string/Path ise dosya adını baz alıp yönlendirme yap
        try:
            p_str = str(io_arg)
        except Exception:
            p_str = io_arg

        if isinstance(io_arg, (str, os.PathLike)):
            base = os.path.basename(p_str)
            if base in path_map:
                io_arg = path_map[base]
        return real_read_excel(io_arg, *args, **kwargs)

    shim.read_excel = redirected_read_excel
    return shim

def _patch_ipython_display(captured_html: List[str]):
    """
    IPython.display.display/HTML/Markdown'i minimal stub ile patch eder.
    display(HTML(...)) çağrılarını yakalayıp HTML stringini captured_html listesine ekler.
    """
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")

    class HTML:
        def __init__(self, data):
            self.data = data
        def __str__(self):
            return str(self.data)

    class Markdown:
        def __init__(self, data):
            self.data = data
        def __str__(self):
            return str(self.data)

    def display(obj):
        # HTML/Markdown objelerini stringe çevirip sakla
        s = getattr(obj, "data", obj)
        captured_html.append(str(s))

    disp.HTML = HTML
    disp.Markdown = Markdown
    disp.display = display
    ipy.display = disp

    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = disp

def _patch_plotly_capture(captured_figs: List[object]):
    """plotly.graph_objects.Figure.show() çağrılarını yakala."""
    import plotly.graph_objects as go

    def patched_show(self, *args, **kwargs):
        captured_figs.append(self)  # sadece yakala; render etme
        return None

    go.Figure.show = patched_show  # type: ignore

def _patch_matplotlib_noop():
    """plt.show()'u no-op yap (GUI açmaya çalışmasın)."""
    try:
        import matplotlib.pyplot as plt
        def _noop_show(*args, **kwargs): 
            return None
        plt.show = _noop_show  # type: ignore
    except Exception:
        pass

def run_script_with_redirects(script_path: str, holidays_path: str, data_path: str, data_label: str):
    """
    - pandas.read_excel -> yüklenen dosyalara yönlendirilir
    - IPython.display.display/HTML -> yakalanır
    - plotly Figure.show -> yakalanır
    - matplotlib plt.show -> no-op
    """
    captured_html: List[str] = []
    captured_figs: List[object] = []

    # 1) IPython display patch
    _patch_ipython_display(captured_html)

    # 2) plotly show patch
    _patch_plotly_capture(captured_figs)

    # 3) matplotlib show no-op
    _patch_matplotlib_noop()

    # 4) pandas shim'i sys.modules'a yerleştir
    path_map = {
        "Turkey_Holidays.xlsx": holidays_path,
        # Her analiz scripti kendi data yolunu sabit tutsa da, base name ile eşleştiriyoruz:
        "CORE_DEPOSIT_DATA.xlsx": data_path,
        "EARLY_WITHDRAWAL_DATA.xlsx": data_path,
        "PREPAYMENT_DATA.xlsx": data_path,
    }
    shim = _build_pandas_shim(path_map)
    sys.modules["pandas"] = shim  # script içindeki import pandas as pd -> bu shim'i alacak

    # 5) Scripti çalıştır
    with contextlib.redirect_stdout(io.StringIO()):  # konsol çıktısını sessize al (isterseniz kaldırabilirsiniz)
        globs = runpy.run_path(script_path, run_name="__main__")

    return globs, captured_figs, captured_html

def _display_results(globs, captured_figs: List[object], captured_html: List[str]):
    # 1) Grafik
    fig = globs.get("fig", None)
    if fig is None and captured_figs:
        fig = captured_figs[-1]
    if fig is not None:
        st.subheader("📈 Grafik")
        try:
            import plotly.graph_objects as go  # noqa
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly grafiği çizilemedi: {e}")

    # 2) Tablolar (display(HTML(...)) ile gelenler)
    tables = []
    for html in captured_html:
        if "<table" in html.lower():
            try:
                dfs = pd.read_html(html)
                tables.extend(dfs)
            except Exception:
                pass

    if tables:
        st.subheader("🧾 Tablolar")
        for i, df in enumerate(tables, start=1):
            st.markdown(f"**Table {i}**")
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Gösterilecek tablo yakalanamadı. (Script HTML tablo basmamış olabilir.)")

if run_btn:
    if not holidays_file or not data_file:
        st.error("Lütfen **Turkey_Holidays.xlsx** ve seçtiğiniz analize ait **DATA** dosyasını yükleyin.")
    else:
        script_file = SCRIPT_PATHS[analysis_choice]
        if not os.path.exists(script_file):
            st.error(f"Script bulunamadı: {script_file}. `app.py` ile aynı klasörde olmalı.")
        else:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Dosyaları geçici dizine, beklenen base name'lerle kaydediyoruz
                    holidays_path = _save_uploaded(tmpdir, holidays_file, "Turkey_Holidays.xlsx")
                    data_path = _save_uploaded(tmpdir, data_file, data_label)

                    st.success("Analiz başlatıldı. Script çalıştırılıyor...")
                    globs, cap_figs, cap_html = run_script_with_redirects(
                        script_path=script_file,
                        holidays_path=holidays_path,
                        data_path=data_path,
                        data_label=data_label,
                    )

                    _display_results(globs, cap_figs, cap_html)
                    st.success("Analiz tamamlandı.")
            except Exception as e:
                st.error(f"Çalışma sırasında bir hata oluştu:\n\n{e}")
