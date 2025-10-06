
    # prepayment_analysis.py (v2)

import os
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")

# --- Path resolver: prefer ENV, else local basename in CWD, else legacy default if exists ---
def _resolve_path(env_var: str, basename: str, legacy_default: str = None) -> str:
    env = os.environ.get(env_var)
    if env and os.path.exists(env):
        return env
    local = os.path.abspath(basename)
    if os.path.exists(local):
        return local
    if legacy_default and os.path.exists(legacy_default):
        return legacy_default
    # last resort: return local (read will raise clear error if missing)
    return local

def _resolve_holidays() -> str:
    env = os.environ.get("TURKEY_HOLIDAYS_PATH")
    if env and os.path.exists(env):
        return env
    local = os.path.abspath("Turkey_Holidays.xlsx")
    if os.path.exists(local):
        return local
    return local  # allow pandas to raise if still missing

    _LEGACY_PREP = r"C:\Users\adevr\riskactive_main\FRTB\From_Sait\Prepayment Analysis\PREPAYMENT_DATA.xlsx"

    prepayment_filepath = _resolve_path("PREPAYMENT_DATA_PATH", "PREPAYMENT_DATA.xlsx", _LEGACY_PREP)
    holidays_path = _resolve_holidays()

    data = pd.read_excel(prepayment_filepath)
    data.columns = data.columns.astype(str).str.replace(".", "_")

    Turkey_Holidays = pd.read_excel(holidays_path)
    if 'TURKEY_HOLIDAYS' not in Turkey_Holidays.columns:
        Turkey_Holidays.columns = ['TURKEY_HOLIDAYS'] + list(Turkey_Holidays.columns[1:])
    Turkey_Holidays['TURKEY_HOLIDAYS'] = pd.to_datetime(Turkey_Holidays['TURKEY_HOLIDAYS'])

    # ---------- # RUNNING THE ANALYSIS (user's original logic should follow below) ----------
    # ...
