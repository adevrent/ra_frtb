
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Fixed expected columns per provided XLSX files:
# REPORT_DATE, NOTIONAL, BRANCH, PRODUCT_CODE, CURRENCY, [INTEREST_RATE, OPENING_DATE, EFFECTIVE_DATE, MATURITY, ...]

st.set_page_config(page_title="ALM Analizleri", layout="wide")
st.title("ALM Analizleri — Streamlit App")
st.caption("Seç: Core / Early Withdrawal / Prepayment → Excel yükle → **Hesapla** (kolon eşleştirme yok — şema sabit).")

def _ensure_cols(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Excel beklenen kolonları içermiyor: {missing}. Mevcut kolonlar: {list(df.columns)}")

def _date(df, col="REPORT_DATE"):
    df[col] = pd.to_datetime(df[col])
    return df

def _sort(df, col="REPORT_DATE"):
    return df.sort_values(by=col).reset_index(drop=True)

def kpi_row(pairs):
    cols = st.columns(len(pairs))
    for col, (label, value) in zip(cols, pairs):
        col.metric(label, value)

def area_line(fig, x, y, name, opacity=0.35):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]),
                             y=pd.concat([y, pd.Series([0]*len(y))]),
                             fill="toself", mode="lines", line=dict(width=0.5),
                             opacity=opacity, showlegend=False))

def exp_decay(t, hl):
    lam = np.log(2) / max(hl, 1e-9)
    return np.exp(-lam * t)

page = st.selectbox("Analiz seçiniz", ["Core Analysis", "Early Withdrawal Analysis", "Prepayment Analysis"])

with st.sidebar:
    st.header("Excel Yükleme")
    uploaded = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    if page == "Core Analysis":
        hl_days = st.number_input("Half-life (gün) — örnek", 1, 3650, 63, 1)
        horizon_days = st.number_input("Horizon (gün)", 30, 3650, 425, 5)
    elif page == "Early Withdrawal Analysis":
        ew_horizon_days = st.number_input("Horizon (gün)", 30, 3650, 365, 5)
        ew_grid = st.number_input("Izgara adımı (gün)", 1, 90, 7, 1)
    else:
        smoothing = st.checkbox("SMM/CPR'ı 3-periyot MA ile düzelt", True)
    calc = st.button("Hesapla", use_container_width=True)

if uploaded is None:
    st.info("Başlamak için sol taraftan Excel dosyanızı yükleyin.")
    st.stop()

try:
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Çalışma sayfası", xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet)
except Exception as e:
    st.error(f"Excel okunamadı: {e}")
    st.stop()

# Common schema checks
required_common = ["REPORT_DATE", "NOTIONAL"]
_ensure_cols(df, required_common)
df = _date(df, "REPORT_DATE")
df_plot["NOTIONAL"] = pd.to_numeric(df_plot["NOTIONAL"], errors="coerce").fillna(0.0)

df = _sort(df, "REPORT_DATE")

# ---------------- Filters to match notebook selections ----------------
with st.sidebar:
    st.subheader("Filtreler")
    # Safe unique lists
    def _opts(series):
        vals = sorted([str(x) for x in series.dropna().unique().tolist()])
        return ["(All)"] + vals

    branch_sel = st.selectbox("BRANCH", _opts(df.get("BRANCH", pd.Series(dtype=object))), index=0)
    prod_sel   = st.selectbox("PRODUCT_CODE", _opts(df.get("PRODUCT_CODE", pd.Series(dtype=object))), index=0)
    curr_sel   = st.selectbox("CURRENCY", _opts(df.get("CURRENCY", pd.Series(dtype=object))), index=0)
    daterange  = st.date_input("Tarih aralığı", [] if df_plot["REPORT_DATE"].empty else [df_plot["REPORT_DATE"].min().date(), df_plot["REPORT_DATE"].max().date()])

# Apply filters
df_f = df.copy()
if "BRANCH" in df_f.columns and branch_sel != "(All)":
    df_f = df_f[df_f["BRANCH"].astype(str) == branch_sel]
if "PRODUCT_CODE" in df_f.columns and prod_sel != "(All)":
    df_f = df_f[df_f["PRODUCT_CODE"].astype(str) == prod_sel]
if "CURRENCY" in df_f.columns and curr_sel != "(All)":
    df_f = df_f[df_f["CURRENCY"].astype(str) == curr_sel]
# Date filter
if isinstance(daterange, list) and len(daterange)==2:
    start_d, end_d = pd.to_datetime(daterange[0]), pd.to_datetime(daterange[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_f = df_f[(df_f["REPORT_DATE"] >= start_d) & (df_f["REPORT_DATE"] <= end_d)]

# Aggregate by date (sum) to match notebook's daily totals
if not df_f.empty:
    df_plot = df_f.groupby("REPORT_DATE", as_index=False).agg({"NOTIONAL":"sum"})
else:
    df_plot = df_f

# Günlük toplamları (aynı tarihte çok satır varsa) aynen notebook mantığına yakın olacak şekilde topla
if df.duplicated("REPORT_DATE").any():
    df = df.groupby("REPORT_DATE", as_index=False).agg({"NOTIONAL":"sum"})

if not calc:
    st.stop()

# ---------------- Core ----------------
if page == "Core Analysis":
    left, right = st.columns(2)
    with left:
        st.markdown("#### Demand Deposit / Notional")
        fig = go.Figure()
        area_line(fig, df_plot["REPORT_DATE"], df_plot["NOTIONAL"], "NOTIONAL", opacity=0.35)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Persistence (örnek)")
        timeline = np.arange(0, horizon_days+1, 5, dtype=int)
        pr = exp_decay(timeline, hl_days)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timeline, y=pr, mode="lines", fill="tozeroy", name="Persistence"))
        mean_life_days = float(np.trapz(pr, timeline)) if len(timeline)>1 else 0.0
fig2.add_vline(x=mean_life_days, line_width=2, line_dash="dot")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    mean_life_days = mean_life_days
    total_notional = float(df_plot["NOTIONAL"].sum())
    last_notional = float(df_plot["NOTIONAL"].iloc[-1])

    kpi_row([
        ("Toplam Notional", f"{total_notional:,.0f}"),
        ("Son Gün Notional", f"{last_notional:,.0f}"),
        ("Mean Life (Days)", f"{mean_life_days:,.1f}"),
        ("Mean Life (Years)", f"{mean_life_days/365.0:,.2f}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df_plot[["REPORT_DATE","NOTIONAL"]], use_container_width=True)

# ------------- Early Withdrawal -------------
elif page == "Early Withdrawal Analysis":
    # Negatif değişimleri outflow kabul ederek SMM çıkarımı
    df["NOTIONAL_SHIFT"] = df_plot["NOTIONAL"].shift(1)
    df["DELTA"] = df_plot["NOTIONAL"] - df["NOTIONAL_SHIFT"]
    df["OUTFLOW"] = (-df["DELTA"]).clip(lower=0)  # azalışları pozitif outflow
    with np.errstate(divide="ignore", invalid="ignore"):
        df["SMM"] = (df["OUTFLOW"] / df["NOTIONAL_SHIFT"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 0.999)
    df["SURV_EMP"] = (1 - df["SMM"]).cumprod().fillna(1.0)

    horizon = np.arange(0, ew_horizon_days+1, ew_grid, dtype=int)
    avg_smm = float(df["SMM"].mean())
    surv_model = (1 - avg_smm) ** (horizon / 30.0)  # 30g ≈ 1 ay
    mean_life_days = float(np.trapz(surv_model, horizon)) if len(horizon) > 1 else np.nan
    avg_cpr = (1 - (1 - avg_smm) ** 12) if avg_smm > 0 else 0.0

    left, right = st.columns(2)
    with left:
        st.markdown("#### Notional & SMM (yaklaşık)")
        fig = go.Figure()
        area_line(fig, df_plot["REPORT_DATE"], df_plot["NOTIONAL"], "NOTIONAL", opacity=0.25)
        fig.add_trace(go.Scatter(x=df_plot["REPORT_DATE"], y=df["SMM"], mode="lines+markers", name="SMM"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Survival (model)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=horizon, y=surv_model, mode="lines", fill="tozeroy", name="Survival"))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    kpi_row([
        ("Ortalama SMM", f"{avg_smm*100:,.2f}%"),
        ("Ortalama CPR", f"{avg_cpr*100:,.2f}%"),
        ("Mean Life (Days, model)", f"{mean_life_days:,.1f}"),
        ("Gözlem Sayısı", f"{len(df_plot):,}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df_plot, use_container_width=True)

# ------------- Prepayment -------------
else:
    # Aynı mantıkla ön ödeme oranı tahmini
    df["NOTIONAL_SHIFT"] = df_plot["NOTIONAL"].shift(1)
    df["DELTA"] = df_plot["NOTIONAL"] - df["NOTIONAL_SHIFT"]
    df["PREPAY_AMT"] = (-df["DELTA"]).clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["SMM"] = (df["PREPAY_AMT"] / df["NOTIONAL_SHIFT"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 0.999)
    df["CPR"] = 1 - (1 - df["SMM"]) ** 12

    if smoothing:
        df["SMM"] = df["SMM"].rolling(3, min_periods=1).mean()
        df["CPR"] = df["CPR"].rolling(3, min_periods=1).mean()
        df["PREPAY_AMT"] = df["PREPAY_AMT"].rolling(3, min_periods=1).mean()

    avg_smm = float(df["SMM"].mean())
    avg_cpr = float(df["CPR"].mean())
    horizon = np.arange(0, 360+1, 30)
    surv_model = (1 - avg_smm) ** (horizon / 30.0)
    wal_months = float((surv_model.sum() * (horizon[1]-horizon[0]) / 30.0)) if len(horizon) > 1 else np.nan

    left, right = st.columns(2)
    with left:
        st.markdown("#### Notional ve Ön Ödeme Tutarı (yaklaşık)")
        fig = go.Figure()
        area_line(fig, df_plot["REPORT_DATE"], df_plot["NOTIONAL"], "NOTIONAL", opacity=0.25)
        fig.add_trace(go.Scatter(x=df_plot["REPORT_DATE"], y=df["PREPAY_AMT"], mode="lines+markers", name="Prepayment (amt)"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### CPR / SMM ve Survival (model)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_plot["REPORT_DATE"], y=df["CPR"]*100, mode="lines", name="CPR (%)"))
        fig2.add_trace(go.Scatter(x=df_plot["REPORT_DATE"], y=df["SMM"]*100, mode="lines", name="SMM (%)"))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    kpi_row([
        ("Ortalama CPR", f"{avg_cpr*100:,.2f}%"),
        ("Ortalama SMM", f"{avg_smm*100:,.2f}%"),
        ("WAL (ay, model)", f"{wal_months:,.1f}"),
        ("Gözlem Sayısı", f"{len(df_plot):,}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df_plot, use_container_width=True)
