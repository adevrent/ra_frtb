
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
        hl_days = st.number_input("Half-life (gün) — örnek", 1, 3650, 60, 1)
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
df["NOTIONAL"] = pd.to_numeric(df["NOTIONAL"], errors="coerce").fillna(0.0)
df = _sort(df, "REPORT_DATE")

if not calc:
    st.stop()

# ---------------- Core ----------------
if page == "Core Analysis":
    left, right = st.columns(2)
    with left:
        st.markdown("#### Demand Deposit / Notional")
        fig = go.Figure()
        area_line(fig, df["REPORT_DATE"], df["NOTIONAL"], "NOTIONAL", opacity=0.35)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Persistence (örnek)")
        timeline = np.arange(0, horizon_days+1, 5, dtype=int)
        pr = exp_decay(timeline, hl_days)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timeline, y=pr, mode="lines", fill="tozeroy", name="Persistence"))
        fig2.add_vline(x=horizon_days, line_width=2, line_dash="dot")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    mean_life_days = float(np.trapz(exp_decay(np.arange(0, horizon_days+1), hl_days),
                                    np.arange(0, horizon_days+1)))
    total_notional = float(df["NOTIONAL"].sum())
    last_notional = float(df["NOTIONAL"].iloc[-1])

    kpi_row([
        ("Toplam Notional", f"{total_notional:,.0f}"),
        ("Son Gün Notional", f"{last_notional:,.0f}"),
        ("Mean Life (Days)", f"{mean_life_days:,.1f}"),
        ("Mean Life (Years)", f"{mean_life_days/365.0:,.2f}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df[["REPORT_DATE","NOTIONAL"]], use_container_width=True)

# ------------- Early Withdrawal -------------
elif page == "Early Withdrawal Analysis":
    # Negatif değişimleri outflow kabul ederek SMM çıkarımı
    df["NOTIONAL_SHIFT"] = df["NOTIONAL"].shift(1)
    df["DELTA"] = df["NOTIONAL"] - df["NOTIONAL_SHIFT"]
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
        area_line(fig, df["REPORT_DATE"], df["NOTIONAL"], "NOTIONAL", opacity=0.25)
        fig.add_trace(go.Scatter(x=df["REPORT_DATE"], y=df["SMM"], mode="lines+markers", name="SMM"))
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
        ("Gözlem Sayısı", f"{len(df):,}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df[["REPORT_DATE","NOTIONAL","OUTFLOW","SMM","SURV_EMP"]], use_container_width=True)

# ------------- Prepayment -------------
else:
    # Aynı mantıkla ön ödeme oranı tahmini
    df["NOTIONAL_SHIFT"] = df["NOTIONAL"].shift(1)
    df["DELTA"] = df["NOTIONAL"] - df["NOTIONAL_SHIFT"]
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
        area_line(fig, df["REPORT_DATE"], df["NOTIONAL"], "NOTIONAL", opacity=0.25)
        fig.add_trace(go.Scatter(x=df["REPORT_DATE"], y=df["PREPAY_AMT"], mode="lines+markers", name="Prepayment (amt)"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### CPR / SMM ve Survival (model)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["REPORT_DATE"], y=df["CPR"]*100, mode="lines", name="CPR (%)"))
        fig2.add_trace(go.Scatter(x=df["REPORT_DATE"], y=df["SMM"]*100, mode="lines", name="SMM (%)"))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    kpi_row([
        ("Ortalama CPR", f"{avg_cpr*100:,.2f}%"),
        ("Ortalama SMM", f"{avg_smm*100:,.2f}%"),
        ("WAL (ay, model)", f"{wal_months:,.1f}"),
        ("Gözlem Sayısı", f"{len(df):,}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df[["REPORT_DATE","NOTIONAL","PREPAY_AMT","SMM","CPR"]], use_container_width=True)
