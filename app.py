
import io
import os
import re
import sys
import json
import shutil
import types
import tempfile
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# Optional imports guarded to keep app resilient
try:
    import nbformat
except Exception:
    nbformat = None

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def save_uploaded_file(uploaded_file, target_path: str):
    \"\"\"Save a Streamlit UploadedFile to the given target_path.\"\"\"
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def load_notebook_source(nb_path: str) -> str:
    \"\"\"Read a .ipynb and concatenate all code cells into a single Python source string.\"\"\"
    if nbformat is None:
        raise RuntimeError("nbformat is not installed. Please `pip install nbformat`.")
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    src_parts = []
    for cell in code_cells:
        # Keep cell boundaries as comments to ease debugging if needed
        src_parts.append(\"\"\"\\n# ---- NOTEBOOK CELL ----\\n\"\"\" + (cell.get("source") or ""))
    return "\\n".join(src_parts)

def execute_notebook_code(source: str, predefs: dict, workdir: str):
    \"\"\"Execute notebook source inside an isolated namespace.
    - Inject `predefs` into the globals.
    - Run inside `workdir` (so relative reads/writes land in our temp area).
    - Capture any created DataFrames and matplotlib figures.
    Returns: (namespace, dataframes, figures)
    \"\"\"
    # Prepare isolated globals/locals
    glb = {
        "__name__": "__notebook__",
        "__file__": os.path.join(workdir, "_injected_notebook.py"),
    }
    glb.update(predefs)

    lcl = {}

    # Make sure relative paths work within temp workdir
    old_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        # Provide no-op display/HTML so notebook calls don't crash
        try:
            from IPython.display import HTML
        except Exception:
            class HTML(str):
                pass
        def _display_stub(*args, **kwargs):
            # We don't render IPython display() here; Streamlit will render DataFrames/figures later.
            return None
        glb["display"] = _display_stub
        glb["HTML"] = HTML

        # Snapshot objects before exec to detect new DataFrames
        before_keys = set(glb.keys()) | set(lcl.keys())

        # Actually execute the notebook code
        exec(compile(source, glb["__file__"], "exec"), glb, lcl)

        # Merge dicts (objects may be assigned into locals)
        ns = {}
        ns.update(glb)
        ns.update(lcl)

        # Collect newly created DataFrames
        new_keys = set(ns.keys()) - before_keys
        dataframes = []
        for k in sorted(new_keys):
            obj = ns[k]
            if isinstance(obj, pd.DataFrame):
                dataframes.append((k, obj))

        # Collect current matplotlib figures
        figures = []
        for num in plt.get_fignums():
            figures.append(plt.figure(num))

        return ns, dataframes, figures
    finally:
        os.chdir(old_cwd)

def render_results(dataframes, figures):
    \"\"\"Render collected DataFrames and figures in Streamlit.\"\"\"
    if figures:
        st.subheader("Generated Plots")
        for i, fig in enumerate(figures, start=1):
            st.pyplot(fig, clear_figure=False)

    if dataframes:
        st.subheader("Generated Tables")
        for name, df in dataframes:
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True)


# -----------------------------
# Sidebar: Analysis Picker & Inputs
# -----------------------------
st.set_page_config(page_title="FRTB Analysis Suite", layout="wide")
st.title("FRTB Analysis Suite (Streamlit)")

analysis = st.sidebar.selectbox(
    "Choose Analysis",
    ["Core Analysis", "Prepayment Analysis", "Early Withdrawal Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.header("Upload Required Files")

# Common: Turkey Holidays (required for all)
holidays_file = st.sidebar.file_uploader("Turkey_Holidays.xlsx", type=["xlsx"])

# Data file varies by analysis
if analysis == "Core Analysis":
    data_label = "CORE_DEPOSIT_DATA.xlsx"
elif analysis == "Prepayment Analysis":
    data_label = "PREPAYMENT_DATA.xlsx"
else:
    data_label = "EARLY_WITHDRAWAL_DATA.xlsx"

data_file = st.sidebar.file_uploader(data_label, type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("Parameters")

# Parameter inputs per analysis
params = {}
if analysis == "Core Analysis":
    params["date_start"] = st.sidebar.text_input("date_start (YYYY-MM-DD)", "2017-01-01")
    params["date_end"]   = st.sidebar.text_input("date_end (YYYY-MM-DD)", "2018-01-01")
    params["currency"]   = st.sidebar.selectbox("currency", ["USD", "EUR", "TRY"], index=2)
    params["freq"]       = st.sidebar.selectbox("freq", ["W", "M", "Y"], index=1)
    params["type"]       = st.sidebar.selectbox("type", ["mean", "roll"], index=0)

    # Not entered by user (forced None)
    params["branch"]      = None
    params["product"]     = None
    params["time_bucket"] = None

elif analysis == "Prepayment Analysis":
    params["start"]     = st.sidebar.text_input("Business_Days start (YYYY-MM-DD)", "2017-01-01")
    params["end"]       = st.sidebar.text_input("Business_Days end (YYYY-MM-DD)",   "2025-12-31")
    params["currency"]  = st.sidebar.selectbox("currency", ["EUR", "USD", "TRY"], index=2)

    # Derived / mirrored params for data_model()
    params["start_date"] = params["start"]
    params["end_date"]   = params["end"]

elif analysis == "Early Withdrawal Analysis":
    params["start_date"] = st.sidebar.text_input("start_date (YYYY-MM-DD)", "2017-01-01")
    params["end_date"]   = st.sidebar.text_input("end_date (YYYY-MM-DD)",   "2025-12-31")
    params["currency"]   = st.sidebar.selectbox("currency", ["EUR", "USD", "TRY"], index=2)
    params["product"]    = st.sidebar.text_input("product", "Vadeli.Mevduat.Ticari")  # user input allowed
    # Fixed None for others
    params["branch"]      = None
    params["time_bucket"] = None

st.sidebar.markdown("---")

run = st.sidebar.button("Run Analysis")

# -----------------------------
# Execution
# -----------------------------
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
    # Validate uploads
    missing = []
    if holidays_file is None:
        missing.append("Turkey_Holidays.xlsx")
    if data_file is None:
        missing.append(data_label)

    if missing:
        st.error(f"Please upload the required file(s): {', '.join(missing)}")
        st.stop()

    # Create a per-run temp directory and save uploads with the expected exact names
    workdir = tempfile.mkdtemp(prefix="frtb_run_")
    st.info(f"Working in: {workdir}")

    expected_names = EXPECTED_FILENAMES[analysis]
    try:
        # Save Turkey_Holidays.xlsx
        save_uploaded_file(holidays_file, os.path.join(workdir, expected_names[0]))
        # Save data file as the exact expected name for the chosen analysis
        save_uploaded_file(data_file, os.path.join(workdir, expected_names[1]))
    except Exception as e:
        st.error(f"Failed to save uploaded files: {e}")
        st.stop()

    # Load notebook source
    nb_path = os.path.join(os.getcwd(), NOTEBOOK_FILES[analysis])
    if not os.path.exists(nb_path):
        st.error(f"Notebook not found: {NOTEBOOK_FILES[analysis]}. "
                 f"Place it in the same folder as this app.")
        st.stop()

    try:
        source = load_notebook_source(nb_path)
    except Exception as e:
        st.error("Error reading notebook: " + str(e))
        st.stop()

    # Prepare predefinitions injected into the notebook's global namespace
    predefs = {}

    if analysis == "Core Analysis":
        # Vars expected by your notebook cells
        predefs.update({
            "date_start": params["date_start"],
            "date_end":   params["date_end"],
            "branch":     params["branch"],
            "product":    params["product"],
            "time_bucket":params["time_bucket"],
            "currency":   params["currency"],
            "freq":       params["freq"],
            "type":       params["type"],
            # In case your code reads these standard filenames:
            "HOLIDAYS_XLSX": "Turkey_Holidays.xlsx",
            "CORE_DATA_XLSX": "CORE_DEPOSIT_DATA.xlsx",
        })

    elif analysis == "Prepayment Analysis":
        predefs.update({
            # Business_Days range
            "start": params["start"],
            "end":   params["end"],
            # data_model mirror vars
            "start_date": params["start_date"],
            "end_date":   params["end_date"],
            # df_prepayment currency
            "currency": params["currency"],
            # Common expected filenames
            "HOLIDAYS_XLSX": "Turkey_Holidays.xlsx",
            "PREPAYMENT_DATA_XLSX": "PREPAYMENT_DATA.xlsx",
        })

    elif analysis == "Early Withdrawal Analysis":
        predefs.update({
            "start_date": params["start_date"],
            "end_date":   params["end_date"],
            "currency":   params["currency"],
            "product":    params["product"],
            "branch":     params["branch"],
            "time_bucket":params["time_bucket"],
            # Expected filenames
            "HOLIDAYS_XLSX": "Turkey_Holidays.xlsx",
            "EARLY_WITHDRAWAL_DATA_XLSX": "EARLY_WITHDRAWAL_DATA.xlsx",
        })

    # Execute the notebook code with our injected params inside the temp workdir
    try:
        ns, dfs, figs = execute_notebook_code(source, predefs, workdir)
    except Exception as e:
        st.error("Error while executing notebook. See traceback below.")
        st.exception(e)
        st.stop()

    # Heuristics:
    # If the notebook created a variable named `table_a` and it's a DataFrame, show it first.
    prioritized = []
    rest = []
    seen = set()
    for name, df in dfs:
        if name == "table_a":
            prioritized.append((name, df))
            seen.add(name)
    for name, df in dfs:
        if name not in seen:
            rest.append((name, df))

    render_results(prioritized + rest, figs)

    # Friendly hint about where outputs landed (in case the notebook writes files)
    with st.expander("Run details & outputs"):
        st.code(json.dumps({
            "workdir": workdir,
            "written_files": sorted(os.listdir(workdir))
        }, indent=2))

    st.success("Analysis complete.")
else:
    st.markdown(
        \"\"\"
        **Instructions**  
        1. Pick an analysis on the left.  
        2. Upload **Turkey_Holidays.xlsx** and the matching data file shown.  
        3. Provide the parameters.  
        4. Click **Run Analysis**.  
        
        This app will run your original Jupyter notebook logic using only the uploaded files
        (saved with the exact filenames your notebooks expect), and render any resulting plots
        and tables (e.g., `table_a`) right here.
        \"\"\"
    )
