from pathlib import Path

import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(
    page_title="Prevision Prix Mais Benin",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --green-dark: #0f5132;
        --green: #2f855a;
        --green-soft: #e7f5ee;
        --card: #ffffff;
        --text-main: #1f2937;
    }
    .stApp {
        background: radial-gradient(circle at 5% 5%, #f4fff8 0%, #ecf8f1 28%, #f8fafc 100%);
        color: #111827;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #111827;
    }
    h1, h2, h3 {
        color: #0f172a;
    }
    .title-wrap {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f5132 0%, #2f855a 100%);
        color: #ffffff;
        box-shadow: 0 10px 24px rgba(15, 81, 50, 0.28);
        margin-bottom: 1rem;
    }
    .title-wrap h1 {
        margin: 0;
        font-size: 1.9rem;
        color: #ffffff;
    }
    .title-wrap p {
        margin-top: 0.4rem;
        margin-bottom: 0;
        color: #ecfdf3;
        opacity: 0.95;
    }
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid #d9efe2;
        border-left: 5px solid var(--green);
        padding: 0.7rem;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] p {
        color: #374151 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    [data-testid="stMetricDelta"] {
        color: #166534 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fffb 0%, #eefaf3 100%) !important;
        border-right: 1px solid #c8e6d5;
    }
    section[data-testid="stSidebar"] * {
        color: #0b3d2a !important;
    }
    .sidebar-help {
        background: #ffffff;
        border: 1px solid #c8e6d5;
        border-radius: 12px;
        padding: 0.75rem;
        margin-top: 0.8rem;
    }
    .sidebar-help p {
        margin: 0.35rem 0;
        color: #0f172a !important;
        font-size: 0.92rem;
    }
    .about-box {
        background: #ffffff;
        border: 1px solid #d9efe2;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-top: 0.5rem;
    }
    .about-box h3 {
        margin: 0 0 0.45rem 0;
        color: #14532d;
    }
    .about-box ul {
        margin: 0.2rem 0 0 1rem;
    }
    .about-box li {
        margin-bottom: 0.3rem;
    }
    .stPlotlyChart, .stDataFrame {
        background: var(--card);
        border-radius: 12px;
        border: 1px solid #e6ecef;
        padding: 0.35rem;
    }
    [data-testid="stDataFrame"] * {
        color: #111827 !important;
    }
    .stCaption {
        color: #374151 !important;
    }
    [data-testid="collapsedControl"] {
        background: #ffffff !important;
        border: 1px solid #2f855a !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(15, 81, 50, 0.2) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #166534 !important;
        color: #166534 !important;
        stroke: #166534 !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: #ecfdf3 !important;
        border-color: #166534 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-wrap">
      <h1>🌽 Prevision des prix du mais au Benin</h1>
      <p>Visualisation interactive des prix historiques et previsions Prophet (FCFA/tonne).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="about-box">
      <h3>A propos du projet</h3>
      <ul>
        <li><b>Objectif:</b> estimer l'evolution mensuelle du prix du mais au Benin pour aider la planification.</li>
        <li><b>Donnees:</b> serie historique FAO (Producer Price LCU/tonne), nettoyee puis structuree en mensuel.</li>
        <li><b>Modele:</b> Prophet, avec saisonnalites annuelles et semiannuelles.</li>
        <li><b>Resultat:</b> projection future, intervalle d'incertitude et indicateurs de performance.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Ce dashboard prevoit le prix mensuel du mais au Benin avec Prophet. "
    "Utilise la barre laterale pour choisir l'horizon de prevision et afficher plus de details."
)

with st.expander("Comment lire ce dashboard", expanded=False):
    st.markdown(
        """
        - **Historique et previsions**: la courbe montre les prix observes puis la projection future.
        - **Previsions cles**: pic et creux attendus sur l'horizon choisi.
        - **Tableau**: prevision centrale avec bornes basse/haute (incertitude du modele).
        - **Metriques**: MAE et MAPE resumant la precision sur les donnees historiques.
        - **Source**: FAO Producer Prices (maize/corn, LCU/tonne, donnees mensuelles).
        """
    )

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED = BASE_DIR / "data" / "processed" / "maize_prices_monthly.csv"
DEFAULT_RAW = BASE_DIR / "data" / "raw" / "producer-prices_ben.csv"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, Path]:
    if DEFAULT_PROCESSED.exists():
        df = pd.read_csv(DEFAULT_PROCESSED)
        if {"ds", "y"}.issubset(df.columns):
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["ds", "y"]).sort_values("ds")
            return df[["ds", "y"]], DEFAULT_PROCESSED

    if DEFAULT_RAW.exists():
        raw = pd.read_csv(DEFAULT_RAW)
        raw = raw[~raw["Iso3"].astype(str).str.startswith("#")].copy()
        maize = raw[
            (raw["Item"] == "Maize (corn)")
            & (raw["Element"] == "Producer Price (LCU/tonne)")
            & (raw["Months"] != "Annual value")
        ].copy()
        maize["ds"] = pd.to_datetime(maize["StartDate"], errors="coerce")
        maize["y"] = pd.to_numeric(maize["Value"], errors="coerce")
        maize = maize.dropna(subset=["ds", "y"]).sort_values("ds")
        return maize[["ds", "y"]], DEFAULT_RAW

    raise FileNotFoundError(
        f"Aucun CSV trouve. Attendu: {DEFAULT_PROCESSED} ou {DEFAULT_RAW}"
    )


with st.sidebar:
    st.header("Controles")
    horizon = st.slider("Horizon de prevision (mois)", 6, 36, 24)
    show_components = st.checkbox("Afficher decomposition", value=True)
    show_metrics = st.checkbox("Afficher metriques", value=True)

    st.markdown(
        """
        <div class="sidebar-help">
            <p><b>Guide rapide</b></p>
            <p>1) Choisis le nombre de mois a projeter.</p>
            <p>2) Active/deactive les graphiques detailles.</p>
            <p>3) Lis les indicateurs (pic, creux, MAE, MAPE).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


df_historical, source_path = load_data()
st.caption(f"Source de donnees: `{source_path}`")


@st.cache_resource
def get_model():
    model = Prophet(
        changepoint_prior_scale=0.12,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
    )
    model.add_seasonality(name="semiannual", period=6, fourier_order=8)
    model.fit(df_historical)
    return model


m = get_model()
future = m.make_future_dataframe(periods=horizon, freq="MS")
forecast = m.predict(future)
future_forecast = forecast[forecast["ds"] > df_historical["ds"].max()].copy()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Observations historiques", f"{len(df_historical):,}")
kpi2.metric("Date min", df_historical["ds"].min().strftime("%Y-%m"))
kpi3.metric("Date max", df_historical["ds"].max().strftime("%Y-%m"))

col_main, col_side = st.columns([4, 2])

with col_main:
    st.subheader("Historique et previsions")
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(
        height=600,
        title="Evolution mensuelle du prix du mais (FCFA/tonne)",
        xaxis_title="Date",
        yaxis_title="Prix",
        hovermode="x unified",
    )
    fig_forecast.update_traces(line=dict(width=3, color="#166534"))
    st.plotly_chart(fig_forecast, use_container_width=True)

    if show_components:
        with st.expander("Decomposition du modele", expanded=False):
            fig_components = plot_components_plotly(m, forecast)
            fig_components.update_layout(height=720)
            st.plotly_chart(fig_components, use_container_width=True)

with col_side:
    st.subheader("Previsions cles")

    peak = future_forecast.loc[future_forecast["yhat"].idxmax()]
    trough = future_forecast.loc[future_forecast["yhat"].idxmin()]

    st.metric(
        "Pic attendu",
        f"{peak['yhat']:,.0f} FCFA",
        delta=peak["ds"].strftime("%b %Y"),
    )
    st.metric(
        "Creux attendu",
        f"{trough['yhat']:,.0f} FCFA",
        delta=trough["ds"].strftime("%b %Y"),
        delta_color="inverse",
    )

    preview = (
        future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(
            columns={
                "ds": "Date",
                "yhat": "Prevision",
                "yhat_lower": "Borne basse",
                "yhat_upper": "Borne haute",
            }
        )
        .tail(horizon)
    )
    st.dataframe(preview, use_container_width=True)

    csv_bytes = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Telecharger previsions (CSV)",
        data=csv_bytes,
        file_name="previsions_mais_benin.csv",
        mime="text/csv",
    )

if show_metrics:
    st.markdown("---")
    st.subheader("Performance du modele sur historique")
    forecast_in = forecast[forecast["ds"].isin(df_historical["ds"])]
    y_true = df_historical.set_index("ds").loc[forecast_in["ds"], "y"]
    y_pred = forecast_in["yhat"]
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:,.0f} FCFA/tonne")
    c2.metric("MAPE", f"{mape:.1f} %")

st.markdown("---")
st.caption("Donnees: FAO Producer Prices | Modele: Prophet | Dashboard: Streamlit")
