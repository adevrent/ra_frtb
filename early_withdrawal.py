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

