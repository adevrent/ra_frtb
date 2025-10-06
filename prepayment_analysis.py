
# prepayment_analysis.py (paths fixed)
import os
import numpy as np
import pandas as pd
import pyreadr
import warnings
import lifelines
import matplotlib.pyplot as plt

from scipy import stats
from datetime import timedelta
from timeit import default_timer as timer
from itertools import combinations, batched
from pathlib import Path
from tqdm import tqdm
from rich.progress import track
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sksurv.nonparametric import kaplan_meier_estimator
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from lifelines import KaplanMeierFitter, WeibullAFTFitter, CoxPHFitter
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, PiecewiseExponentialFitter, NelsonAalenFitter, SplineFitter
from lifelines.utils import find_best_parametric_model, median_survival_times

from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.simplefilter('ignore')

def _pick_file(env_var: str, basename: str, default_path: str = None) -> str:
    env = os.environ.get(env_var)
    if env and os.path.exists(env):
        return env
    local = os.path.abspath(basename)
    if os.path.exists(local):
        return local
    if default_path and os.path.exists(default_path):
        return default_path
    return env or local

wd_SAVE = os.getcwd()
wd_PRAM_default = os.getcwd()

prepayment_default = r"C:\Users\adevr\riskactive_main\FRTB\From_Sait\Prepayment Analysis\PREPAYMENT_DATA.xlsx"
prepayment_filepath = _pick_file("PREPAYMENT_DATA_PATH", "PREPAYMENT_DATA.xlsx", prepayment_default)

holidays_path = os.environ.get("TURKEY_HOLIDAYS_PATH")
if not holidays_path or not os.path.exists(holidays_path):
    holidays_path = os.path.join(wd_PRAM_default, "Turkey_Holidays.xlsx")

progress_bar = Progress(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TimeRemainingColumn())

data = pd.read_excel(prepayment_filepath)
data.columns = data.columns.astype(str).str.replace(".", "_")

Turkey_Holidays = pd.read_excel(holidays_path)
if 'TURKEY_HOLIDAYS' not in Turkey_Holidays.columns:
    Turkey_Holidays.columns = ['TURKEY_HOLIDAYS'] + list(Turkey_Holidays.columns[1:])
Turkey_Holidays['TURKEY_HOLIDAYS'] = pd.to_datetime(Turkey_Holidays['TURKEY_HOLIDAYS'])

Business_Days = pd.date_range(start="2017-01-01", end="2025-12-31", freq='B')
Turkey_Business_Days = Business_Days[~Business_Days.isin(Turkey_Holidays['TURKEY_HOLIDAYS'])]

Monthly_Business_Days = (
    pd.DataFrame({"Business_Days": Turkey_Business_Days})
    .assign(YearMonth=lambda df: df["Business_Days"].dt.to_period("M"))
    .groupby("YearMonth")["Business_Days"]
    .agg(["first", "last"])
    .reset_index()
)

def next_business_day(date, holidays):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day

def classify_time_bucket(months):
    if 0 <= months <= 6:   return "0-6M"
    elif 6 < months <= 12: return "6-12M"
    elif 12 < months <= 24: return "1-2Y"
    elif 24 < months <= 36: return "2-3Y"
    elif 36 < months <= 48: return "3-4Y"
    elif 48 < months <= 60: return "4-5Y"
    elif 60 < months <= 120: return "5-10Y"
    elif months > 120:     return "+10Y"
    else:                  return "--"

def rolling_dates(data, year = 5):
    rolling_dates = []
    period = year*12
    for i in range(len(data) - (period-1)):
        START_DATE = data.loc[i, "first"]
        END_DATE = data.loc[i + (period-1), "last"]
        rolling_dates.append({"START_DATE": START_DATE, "END_DATE": END_DATE})
    rolling_dates = pd.DataFrame(rolling_dates)
    return rolling_dates

# ... rest of the user's analysis stays as-is ...
