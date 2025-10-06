# app.py
import os
import io
import runpy
import tempfile
import contextlib
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

# ----- App Config -----
st.set_page_config(page_title="Deposit Analytics Suite", layout="wide")

st.title("📊 Deposit Analytics Suite")
st.write(
    "Aşağıdan analiz türünü seçip **Turkey_Holidays.xlsx** dosyasını ve seçtiğiniz analize ait DATA dosyasını yükleyin. "
    "Uygulama, ilgili scripti (Core / Early Withdrawal / Prepayment) kendi orijinal akışıyla çalıştırır ve "
    "‘# RUNNING THE ANALYSIS’ kısmındaki grafik ve tabloları ekranda gösterir."
)

# ----- Sidebar Inputs -----
with st.sidebar:
    st.header("⚙️ Ayarlar")

    analysis_choice = st.radio(
        "Analiz Türü",
        ["Core Analysis", "Early Withdrawal Analysis", "Prepayment Analysis"],
        index=0,
    )

    holidays_file = st.file_uploader(
        "Turkey_Holidays.xlsx yükleyin", type=["xlsx"], accept_multiple_files=False
    )

    data_label = {
        "Core Analysis": "CORE_DEPOSIT_DATA.xlsx",
        "Early Withdrawal Analysis": "EARLY_WITHDRAWAL_DATA.xlsx",
        "Prepayment Analysis": "PREPAYMENT_DATA.xlsx",
    }[analysis_choice]

    data_file = st.file_uploader(
        f"{data_label} yükleyin", type=["xlsx"], accept_multiple_files=False
    )

    run_btn = st.button("🚀 Analizi Çalıştır")


# ----- Script Paths (aynı klasörde oldukları varsayımıyla) -----
SCRIPT_PATHS = {
    "Core Analysis": "core_analysis.py",
    "Early Withdrawal Analysis": "early_withdrawal_analysis.py",
    "Prepayment Analysis": "prepayment_analysis.py",
}

# Uyarı bloğu
st.info(
    "📂 Bu uygulama, scriptlerin içinde sabitlenen dosya yollarını bozmamak için "
    "yüklediğiniz dosyaları geçici bir klasöre yazar ve ilgili **global değişkenleri** "
    "script çalıştırmadan önce o klasöre işaret eder (örn. `wd_PRAM`, `*_filepath`)."
)

def save_uploaded_file(tmpdir: str, uploaded, filename: str) -> str:
    """Streamlit uploaded file'ı tmpdir altına kaydedip tam yolu döndürür."""
    path = os.path.join(tmpdir, filename)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def pick_data_var_for_analysis(analysis: str) -> str:
    """Seçilen analize göre veri dosyası global değişken adını döndürür."""
    if analysis == "Core Analysis":
        return "core_deposit_filepath"
    elif analysis == "Early Withdrawal Analysis":
        return "early_withdrawal_filepath"
    elif analysis == "Prepayment Analysis":
        return "prepayment_filepath"
    else:
        raise ValueError("Unknown analysis type")

def collect_result_tables(globs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Script çalıştıktan sonra sonuç tablolarını topla.
    Öncelik: table_a / table_b / table_c.
    Ek olarak, *Analysis* isimli özet df'leri de yakala (ham data ve ara df'ler hariç).
    """
    out = {}

    # Öncelik: table_a/b/c
    for key in ["table_a", "table_b", "table_c"]:
        if key in globs:
            val = globs[key]
            if isinstance(val, dict):
                out[key] = pd.DataFrame(val)
            elif isinstance(val, pd.DataFrame):
                out[key] = val

    # Ek özetler: *Analysis* ile biten/başlayan DataFrame'ler
    for key, val in globs.items():
        if isinstance(val, pd.DataFrame):
            name_low = key.lower()
            if any(token in name_low for token in ["analysis", "summary", "result"]):
                # ham girdiler (Report_*, Max_Report_Date vb.) ve çok büyük tabloları gösterme
                if not name_low.startswith("report_") and not name_low.startswith("max_report_date"):
                    if key not in out:
                        out[key] = val

    return out

def run_analysis(analysis: str, holidays_xlsx, data_xlsx) -> Dict[str, Any]:
    """
    Seçilen scripti, enjekte edilen global değişkenler ile runpy.run_path üzerinden çalıştır.
    Çıktı globals sözlüğünü döndür.
    """
    script_filename = SCRIPT_PATHS[analysis]
    script_path = os.path.join(os.path.dirname(__file__), script_filename)
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Script bulunamadı: {script_filename}. "
            "Bu app.py dosyasını, ilgili scriptlerle aynı klasöre koyup deploy edin."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Tatil dosyasını tmp'e yaz ve wd_PRAM ona işaret etsin
        holidays_path = save_uploaded_file(tmpdir, holidays_xlsx, "Turkey_Holidays.xlsx")

        # 2) Veri dosyasını tmp'e yaz ve ilgili *_filepath değişkenini buna işaret ettir
        data_var = pick_data_var_for_analysis(analysis)
        data_path = save_uploaded_file(tmpdir, data_xlsx, data_label)

        # run_path'e enjekte edeceğimiz global değişkenler
        injected_globals = {
            # Scriptler tatil dosyasını f"{wd_PRAM}/Turkey_Holidays.xlsx" olarak okuyor
            "wd_PRAM": tmpdir,
            # Her analiz özelindeki veri dosyası yolu
            data_var: data_path,
        }

        # Bazı scriptler run sırasında stdout'a (display/print) yazabiliyor;
        # stream'i yakalayıp sessiz çalıştırıyoruz (gerekirse log’a alabilirsiniz).
        with contextlib.redirect_stdout(io.StringIO()):
            globs = runpy.run_path(script_path, init_globals=injected_globals)

        # Çalışmanın sonuçlarını geri döndür
        return globs


# ----- Run Button -----
if run_btn:
    if not holidays_file or not data_file:
        st.error("Lütfen hem **Turkey_Holidays.xlsx** hem de analiz için gerekli **DATA** dosyasını yükleyin.")
    else:
        try:
            st.success("Analiz başlatıldı. Script çalıştırılıyor...")
            globs = run_analysis(analysis_choice, holidays_file, data_file)

            # 1) Grafik (Plotly) yakala
            fig = globs.get("fig", None)
            if fig is not None:
                st.subheader("📈 Grafik")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Script içinde `fig` nesnesi bulunamadı. (Grafik üretmeyen bir koşul olabilir.)")

            # 2) Tabloları yakala ve göster
            result_tables = collect_result_tables(globs)
            if result_tables:
                st.subheader("🧾 Tablolar")
                for name, df in result_tables.items():
                    st.markdown(f"**{name}**")
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("Gösterilecek özet tablo yakalanamadı. Script, tabloları yalnızca HTML/print ile göstermiş olabilir.")

            # 3) İndirilebilir çıktı (opsiyonel): yakalanan tabloları zip’leyebilirsiniz.
            # (İsterseniz burada CSV export da eklenebilir.)

            st.success("Analiz tamamlandı.")

        except Exception as e:
            st.error(f"Çalışma sırasında bir hata oluştu:\n\n{e}")

# ----- Footer -----
st.caption(
    "Gereksinimler `requirements.txt` ile uyumludur (örn. lifelines, plotly, scikit-survival, openpyxl). "
    "App dosyasını, üç analiz scripti ve requirements ile aynı dizinde kullanın. "
    "• Script referansları: core_analysis.py, early_withdrawal_analysis.py, prepayment_analysis.py"
)
