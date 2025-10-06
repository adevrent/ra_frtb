
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ALM Analizleri", layout="wide")
st.title("ALM Analizleri — Streamlit App")
st.caption("Seç: Core Analysis / Early Withdrawal Analysis / Prepayment Analysis → Excel yükle → **Hesapla**")

def _coerce_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return s

def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def kpi_row(pairs):
    cols = st.columns(len(pairs))
    for col, (label, value) in zip(cols, pairs):
        col.metric(label, value)

def area_line(fig, x, y, name, opacity=0.35):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([x, x[::-1]]),
        y=pd.concat([y, pd.Series([0]*len(y))]),
        fill="toself", mode="lines", line=dict(width=0.5), opacity=opacity, showlegend=False
    ))

def exp_decay(t, hl):
    lam = np.log(2) / max(hl, 1e-9)
    return np.exp(-lam * t)

page = st.selectbox("Analiz seçiniz", ["Core Analysis", "Early Withdrawal Analysis", "Prepayment Analysis"])

with st.sidebar:
    st.header("Excel Yükleme")
    uploaded = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    st.caption("⚠️ Notebook’lardaki export formatına uygun dosya yükleyin. Aşağıdaki kolon eşleştiricilerle farklı adları uyarlayabilirsiniz.")
    st.divider()
    st.header("Parametreler")
    if page == "Core Analysis":
        hl_days = st.number_input("Half-life (gün) — örnek", 1, 3650, 60, 1)
        horizon_days = st.number_input("Horizon (gün)", 30, 3650, 425, 5)
        area_opacity = st.slider("Alan opaklığı", 0.0, 1.0, 0.35, 0.05)
    elif page == "Early Withdrawal Analysis":
        ew_horizon_days = st.number_input("Horizon (gün)", 30, 3650, 365, 5)
        ew_grid = st.number_input("Izgara adımı (gün)", 1, 90, 7, 1)
    else:
        psa = st.number_input("PSA çarpanı (örnek parametre)", 0, 400, 100, 10)
        smoothing = st.checkbox("Hareketli ortalama ile düzelt", True)
    st.divider()
    calc = st.button("Hesapla", use_container_width=True)

if uploaded is None:
    st.info("Başlamak için sol taraftan Excel dosyanızı yükleyin.")
    st.stop()

try:
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Çalışma sayfası", xls.sheet_names, index=0)
    df_raw = pd.read_excel(xls, sheet_name=sheet)
except Exception as e:
    st.error(f"Excel okunamadı: {e}")
    st.stop()

st.markdown("### Kolon Eşleştir")
all_cols = list(df_raw.columns)

if page == "Core Analysis":
    date_guess = next((c for c in all_cols if "date" in c.lower()), all_cols[0] if all_cols else None)
    dd_guess = next((c for c in all_cols if "demand" in c.lower() or "deposit" in c.lower()), date_guess)
    not_guess = next((c for c in all_cols if "notional" in c.lower() or "core" in c.lower()), dd_guess)

    date_col = st.selectbox("Tarih kolonu", all_cols, index=all_cols.index(date_guess) if date_guess in all_cols else 0)
    dd_col   = st.selectbox("Demand Deposit kolonu", all_cols, index=all_cols.index(dd_guess) if dd_guess in all_cols else 0)
    not_col  = st.selectbox("Notional/Core kolonu (opsiyonel)", ["<none>"] + all_cols, index=0)

elif page == "Early Withdrawal Analysis":
    date_guess = next((c for c in all_cols if "date" in c.lower()), all_cols[0] if all_cols else None)
    bal_guess = next((c for c in all_cols if "bal" in c.lower() or "stock" in c.lower()), date_guess)
    out_guess = next((c for c in all_cols if "withdraw" in c.lower() or "outflow" in c.lower() or "runoff" in c.lower()), bal_guess)

    date_col = st.selectbox("Tarih kolonu", all_cols, index=all_cols.index(date_guess) if date_guess in all_cols else 0)
    bal_col  = st.selectbox("Başlangıç/Mevcut Bakiye kolonu", all_cols, index=all_cols.index(bal_guess) if bal_guess in all_cols else 0)
    wdr_col  = st.selectbox("Erken Çekim (tutar/rate) kolonu", all_cols, index=all_cols.index(out_guess) if out_guess in all_cols else 0)

else:
    date_guess = next((c for c in all_cols if "date" in c.lower()), all_cols[0] if all_cols else None)
    bal_guess = next((c for c in all_cols if "bal" in c.lower() or "principal" in c.lower() or "outstanding" in c.lower()), date_guess)
    prep_guess = next((c for c in all_cols if "prepay" in c.lower() or "cpr" in c.lower() or "smm" in c.lower()), bal_guess)

    date_col = st.selectbox("Tarih kolonu", all_cols, index=all_cols.index(date_guess) if date_guess in all_cols else 0)
    bal_col  = st.selectbox("Kredi Havuzu Bakiye kolonu", all_cols, index=all_cols.index(bal_guess) if bal_guess in all_cols else 0)
    pre_col  = st.selectbox("Ön ödeme (tutar/CPR/SMM) kolonu", all_cols, index=all_cols.index(prep_guess) if prep_guess in all_cols else 0)

if not calc:
    st.stop()

df = df_raw.copy()
if date_col in df.columns:
    df[date_col] = _coerce_datetime(df[date_col])

if page == "Core Analysis":
    df = _to_numeric(df, [dd_col] + ([] if not_col == "<none>" else [not_col]))
    df = df.dropna(subset=[date_col, dd_col]).sort_values(by=date_col)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Notional / Core Deposit")
        fig = go.Figure()
        area_line(fig, df[date_col], df[dd_col], dd_col, opacity=0.35)
        if not_col != "<none>":
            fig.add_trace(go.Scatter(x=df[date_col], y=df[not_col], mode="lines+markers", name=not_col, line=dict(width=2)))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Persistence (örnek exponansiyel)")
        timeline = np.arange(0, horizon_days+1, 5, dtype=int)
        pr = np.exp(-np.log(2)/max(hl_days,1e-9) * timeline)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timeline, y=pr, mode="lines", fill="tozeroy", name="Persistence"))
        fig2.add_vline(x=horizon_days, line_width=2, line_dash="dot")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    core_pct = None
    if not_col != "<none>":
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (df[not_col] / df[dd_col]) * 100.0
            core_pct = float(s.replace([np.inf, -np.inf], np.nan).dropna().mean()) if not s.dropna().empty else None

    mean_life_days = float(np.trapz(np.exp(-np.log(2)/max(hl_days,1e-9) * np.arange(0, horizon_days+1)),
                                    np.arange(0, horizon_days+1)))
    kpi_row([
        ("Core Deposit %", f"{core_pct:.2f} %" if core_pct is not None else "—"),
        ("Core Deposits (sum)", f"{df[not_col].sum():,.0f}" if not_col != "<none>" else "—"),
        ("Mean Life (Days)", f"{mean_life_days:,.1f}"),
        ("Mean Life (Years)", f"{mean_life_days/365.0:,.2f}"),
        ("Horizon (Years)", f"{horizon_days/365.0:,.2f}"),
    ])

    st.markdown("#### Veri")
    st.dataframe(df, use_container_width=True)

elif page == "Early Withdrawal Analysis":
    df = _to_numeric(df, [bal_col, wdr_col])
    df = df.dropna(subset=[date_col, bal_col]).sort_values(by=date_col)

    series = df[wdr_col].copy()
    guess_is_rate = (series.between(0, 1).mean() > 0.5) or (series.between(0, 100).mean() > 0.5 and series.mean() < 50)
    if not guess_is_rate:
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (df[wdr_col] / df[bal_col]).clip(lower=0).fillna(0.0)
    else:
        rate = series.copy()
        if rate.max() > 1.5:
            rate = rate / 100.0
        rate = rate.clip(lower=0).fillna(0.0)

    smm = rate.clip(0, 0.999)
    survival = (1 - smm).cumprod()
    horizon = np.arange(0, ew_horizon_days+1, ew_grid, dtype=int)
    avg_smm = float(smm.mean()) if not smm.empty else 0.0
    surv_model = (1 - avg_smm) ** (horizon / 30.0)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Bakiye ve Erken Çekim Oranı")
        fig = go.Figure()
        area_line(fig, df[date_col], df[bal_col], bal_col, opacity=0.25)
        fig.add_trace(go.Scatter(x=df[date_col], y=smm, mode="lines+markers", name="SMM (yaklaşık)"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Survival / Persistence (model)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=horizon, y=surv_model, mode="lines", fill="tozeroy", name="Survival"))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    mean_life_days = float(np.trapz(surv_model, horizon)) if len(horizon) > 1 else np.nan
    avg_cpr = (1 - (1 - avg_smm) ** 12) if avg_smm > 0 else 0.0

    kpi_row([
        ("Ortalama SMM", f"{avg_smm*100:,.2f}%"),
        ("Ortalama CPR", f"{avg_cpr*100:,.2f}%"),
        ("Mean Life (Days, model)", f"{mean_life_days:,.1f}"),
        ("Gözlem Sayısı", f"{len(df):,}"),
    ])

    st.markdown("#### Veri")
    out = df[[date_col, bal_col, wdr_col]].copy()
    out["SMM_est"] = smm
    out["Survival_emp"] = survival
    st.dataframe(out, use_container_width=True)

else:
    df = _to_numeric(df, [bal_col, pre_col])
    df = df.dropna(subset=[date_col, bal_col]).sort_values(by=date_col)

    pre = df[pre_col].astype(float)
    is_rate = (pre.between(0, 1).mean() > 0.5) or (pre.max() <= 100 and pre.mean() < 50)
    if is_rate:
        if pre.max() > 1.5:
            cpr = pre / 100.0
        else:
            cpr = pre.copy()
        smm = 1 - (1 - cpr) ** (1/12)
        prep_amt = smm * df[bal_col]
    else:
        prep_amt = pre.clip(lower=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            smm = (prep_amt / df[bal_col]).clip(0, 0.999).fillna(0.0)
        cpr = 1 - (1 - smm) ** 12

    if st.session_state.get("smoothing", True):
        smm = smm.rolling(3, min_periods=1).mean()
        cpr = cpr.rolling(3, min_periods=1).mean()
        prep_amt = prep_amt.rolling(3, min_periods=1).mean()

    avg_smm = float(smm.mean()) if not smm.empty else 0.0
    horizon = np.arange(0, 360+1, 30)
    surv_model = (1 - avg_smm) ** (horizon / 30.0)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Bakiye ve Ön Ödeme")
        fig = go.Figure()
        area_line(fig, df[date_col], df[bal_col], bal_col, opacity=0.25)
        fig.add_trace(go.Scatter(x=df[date_col], y=prep_amt, mode="lines+markers", name="Prepayment (amount est.)"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### CPR / SMM ve Survival (model)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df[date_col], y=cpr*100, mode="lines", name="CPR (%)"))
        fig2.add_trace(go.Scatter(x=df[date_col], y=smm*100, mode="lines", name="SMM (%)"))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    avg_cpr = float(cpr.mean()) if not cpr.empty else 0.0
    wal_months = float((surv_model.sum() * (horizon[1]-horizon[0]) / 30.0)) if len(horizon) > 1 else np.nan

    kpi_row([
        ("Ortalama CPR", f"{avg_cpr*100:,.2f}%"),
        ("Ortalama SMM", f"{avg_smm*100:,.2f}%"),
        ("WAL (ay, model)", f"{wal_months:,.1f}"),
        ("Gözlem Sayısı", f"{len(df):,}"),
    ])

    st.markdown("#### Veri")
    out = df[[date_col, bal_col, pre_col]].copy()
    out["SMM_est"] = smm
    out["CPR_est"] = cpr
    st.dataframe(out, use_container_width=True)
