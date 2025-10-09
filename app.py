
import io
import os
import re
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nbformat
except ImportError:
    nbformat = None


def save_uploaded_file(uploaded_file, target_path: str):
    """Save a Streamlit UploadedFile to the given target_path."""
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def load_notebook_source(nb_path: str) -> str:
    """Read a .ipynb and concatenate all code cells into a single Python source string."""
    if nbformat is None:
        raise RuntimeError("nbformat is not installed. Please `pip install nbformat`.")
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    src_parts = []
    for cell in code_cells:
        src_parts.append("\n# ---- NOTEBOOK CELL ----\n" + (cell.get("source") or ""))
    return "\n".join(src_parts)


def patch_notebook_source(source: str, analysis: str, params: dict) -> str:
    """Patch common hard-coded bits so the notebook consumes UI params without manual edits."""
    patched = source

    if analysis == "Prepayment Analysis":
        patched = re.sub(
            r"Business_Days\s*=\s*pd\.date_range\(\s*start\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]\s*,\s*end\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]\s*,\s*freq\s*=\s*['\"]B['\"]\s*\)",
            "Business_Days = pd.date_range(start=start, end=end, freq='B')",
            patched
        )
        patched = re.sub(r"start_date\s*=\s*(None|['\"]\d{4}-\d{2}-\d{2}['\"])", "start_date=start_date", patched)
        patched = re.sub(r"end_date\s*=\s*(None|['\"]\d{4}-\d{2}-\d{2}['\"])", "end_date=end_date", patched)

    elif analysis == "Early Withdrawal Analysis":
        patched = re.sub(
            r"Business_Days\s*=\s*pd\.date_range\(\s*start\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]\s*,\s*end\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]\s*,\s*freq\s*=\s*['\"]B['\"]\s*\)",
            "Business_Days = pd.date_range(start=start_date, end=end_date, freq='B')",
            patched
        )
        patched = re.sub(r"\bstart_date\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]", "start_date = start_date", patched)
        patched = re.sub(r"\bend_date\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]", "end_date = end_date", patched)

    elif analysis == "Core Analysis":
        patched = re.sub(r"\bdate_start\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]", "date_start = date_start", patched)
        patched = re.sub(r"\bdate_end\s*=\s*['\"]\d{4}-\d{2}-\d{2}['\"]", "date_end = date_end", patched)

    # Generic Windows path -> basename for pd.read_excel / read_csv
    patched = re.sub(
        r"pd\.(read_excel|read_csv)\(\s*([ru])?['\"]([A-Za-z]:\\\\[^'\"]+?\\\\([^'\"]+\.(xlsx|csv)))['\"]\s*\)",
        r"pd.\1('\4')",
        patched
    )

    return patched


def execute_notebook_code(source: str, predefs: dict, workdir: str):
    """Execute notebook code in isolated namespace, inject pd/np/plt, redirect filepaths, collect outputs."""
    glb = {
        "__name__": "__notebook__",
        "__file__": os.path.join(workdir, "_injected_notebook.py"),
        # Pre-inject common libs so early cells/functions can use them
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    glb.update(predefs)
    lcl = {}
    old_cwd = os.getcwd()

    try:
        os.chdir(workdir)

        try:
            from IPython.display import HTML
        except Exception:
            class HTML(str): pass
        def display_stub(*args, **kwargs):
            return None
        glb["display"] = display_stub
        glb["HTML"] = HTML

        # Intercept pd.read_excel to map absolute paths -> uploaded basenames in workdir
        _orig_read_excel = pd.read_excel
        def _read_excel_intercept(path, *args, **kwargs):
            try:
                base = os.path.basename(path) if isinstance(path, str) else path
                candidate = os.path.join(workdir, base) if isinstance(base, str) else None
                if isinstance(candidate, str) and os.path.exists(candidate):
                    return _orig_read_excel(candidate, *args, **kwargs)
            except Exception:
                pass
            return _orig_read_excel(path, *args, **kwargs)
        pd.read_excel = _read_excel_intercept

        exec(compile(source, glb["__file__"], "exec"), glb, lcl)

        pd.read_excel = _orig_read_excel

        ns = {**glb, **lcl}
        dataframes = [(k, v) for k, v in ns.items() if isinstance(v, pd.DataFrame)]
        figures = [plt.figure(num) for num in plt.get_fignums()]
        return ns, dataframes, figures

    finally:
        os.chdir(old_cwd)


def render_results(dataframes, figures):
    if figures:
        st.subheader("Generated Plots")
        for fig in figures:
            st.pyplot(fig, clear_figure=False)
    if dataframes:
        st.subheader("Generated Tables")
        for name, df in dataframes:
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True)


st.set_page_config(page_title="FRTB Analysis Suite", layout="wide")
st.title("FRTB Analysis Suite (Auto‑Patch v2)")

analysis = st.sidebar.selectbox(
    "Choose Analysis",
    ["Core Analysis", "Prepayment Analysis", "Early Withdrawal Analysis"]
)

st.sidebar.header("Upload Required Files")
holidays_file = st.sidebar.file_uploader("Turkey_Holidays.xlsx", type=["xlsx"])

if analysis == "Core Analysis":
    data_label = "CORE_DEPOSIT_DATA.xlsx"
elif analysis == "Prepayment Analysis":
    data_label = "PREPAYMENT_DATA.xlsx"
else:
    data_label = "EARLY_WITHDRAWAL_DATA.xlsx"

data_file = st.sidebar.file_uploader(data_label, type=["xlsx"])

st.sidebar.header("Parameters")
params = {}

if analysis == "Core Analysis":
    params["date_start"] = st.sidebar.text_input("date_start (YYYY-MM-DD)", "2017-01-01")
    params["date_end"] = st.sidebar.text_input("date_end (YYYY-MM-DD)", "2018-01-01")
    params["currency"] = st.sidebar.selectbox("currency", ["USD", "EUR", "TRY"], index=2)
    params["freq"] = st.sidebar.selectbox("freq", ["W", "M", "Y"], index=1)
    params["type"] = st.sidebar.selectbox("type", ["mean", "roll"], index=0)
    params["branch"] = None
    params["product"] = None
    params["time_bucket"] = None

elif analysis == "Prepayment Analysis":
    params["start"] = st.sidebar.text_input("Business_Days start (YYYY-MM-DD)", "2017-01-01")
    params["end"] = st.sidebar.text_input("Business_Days end (YYYY-MM-DD)", "2025-12-31")
    params["currency"] = st.sidebar.selectbox("currency", ["EUR", "USD", "TRY"], index=2)
    params["start_date"] = params["start"]
    params["end_date"] = params["end"]

else:
    params["start_date"] = st.sidebar.text_input("start_date (YYYY-MM-DD)", "2017-01-01")
    params["end_date"] = st.sidebar.text_input("end_date (YYYY-MM-DD)", "2025-12-31")
    params["currency"] = st.sidebar.selectbox("currency", ["EUR", "USD", "TRY"], index=2)
    params["product"] = st.sidebar.text_input("product", "Vadeli.Mevduat.Ticari")
    params["branch"] = None
    params["time_bucket"] = None

run = st.sidebar.button("Run Analysis")

NOTEBOOK_FILES = {
    "Core Analysis": "Core Analysis - Demand Deposit.ipynb",
    "Prepayment Analysis": "Prepayment Analysis.ipynb",
    "Early Withdrawal Analysis": "Early Withdrawal Analysis.ipynb",
}

EXPECTED_FILENAMES = {
    "Core Analysis": ["Turkey_Holidays.xlsx", "CORE_DEPOSIT_DATA.xlsx"],
    "Prepayment Analysis": ["Turkey_Holidays.xlsx", "PREPAYMENT_DATA.xlsx"],
    "Early Withdrawal Analysis": ["Turkey_Holidays.xlsx", "EARLY_WITHDRAWAL_DATA.xlsx"],
}

if run:
    missing = []
    if holidays_file is None:
        missing.append("Turkey_Holidays.xlsx")
    if data_file is None:
        missing.append(data_label)
    if missing:
        st.error(f"Please upload the required file(s): {', '.join(missing)}")
        st.stop()

    workdir = tempfile.mkdtemp(prefix="frtb_run_")
    st.info(f"Working in: {workdir}")

    expected = EXPECTED_FILENAMES[analysis]
    save_uploaded_file(holidays_file, os.path.join(workdir, expected[0]))
    save_uploaded_file(data_file, os.path.join(workdir, expected[1]))

    nb_path = os.path.join(os.getcwd(), NOTEBOOK_FILES[analysis])
    if not os.path.exists(nb_path):
        st.error(f"Notebook not found: {NOTEBOOK_FILES[analysis]}")
        st.stop()

    source = load_notebook_source(nb_path)
    source = patch_notebook_source(source, analysis, params)

    predefs = {}
    predefs.update(params)

    try:
        ns, dfs, figs = execute_notebook_code(source, predefs, workdir)
    except Exception as e:
        st.error("Error executing notebook:")
        st.exception(e)
        st.stop()

    render_results(dfs, figs)
    st.success("Analysis complete.")
else:
    st.markdown(
        "**Instructions:**\n"
        "1. Choose an analysis on the left.\n"
        "2. Upload Turkey_Holidays.xlsx and the corresponding data file.\n"
        "3. Enter parameters.\n"
        "4. Click Run Analysis."
    )
