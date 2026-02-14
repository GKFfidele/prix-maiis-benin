from pathlib import Path
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ────────────────────────────────────────────────
#   CONFIGURATION GLOBALE + PAGE
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Prévision Prix Maïs Bénin",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────
#   STYLE CSS – Fond noir data-science + sidebar arrow visible
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0d1117;
        color: #c9d1d9;
    }

    h1, h2, h3 {
        color: #58a6ff !important;
    }

    .stMarkdown, .stText, .stCaption, p, div, span, label {
        color: #c9d1d9 !important;
    }

    /* Sidebar flèche très visible */
    [data-testid="collapsedControl"] {
        background-color: #21262d !important;
        border: 2px solid #58a6ff !important;
        border-radius: 50% !important;
        width: 48px !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="collapsedControl"] svg {
        fill: #58a6ff !important;
        width: 28px !important;
        height: 28px !important;
    }

    [data-testid="collapsedControl"]:hover {
        background-color: #30363d !important;
        transform: scale(1.15) !important;
        box-shadow: 0 0 25px rgba(88, 166, 255, 0.8) !important;
    }

    section[data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }

    .stExpander {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    .stMetric {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }

    hr {
        border-color: #30363d !important;
    }

    .title-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        text-align: center;
    }

    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
    }

    .project-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    }

    .project-box h3 {
        color: #58a6ff !important;
        margin-bottom: 0.6rem;
    }

    .project-box p {
        margin: 0.25rem 0 0.65rem 0;
        color: #c9d1d9 !important;
        line-height: 1.55;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#   HEADER TITRE
# ────────────────────────────────────────────────
st.markdown("""
    <div class="title-card">
        <h1>🌽 Prévision des Prix du Maïs au Bénin</h1>
        <p style="font-size: 1.15rem; color: #8b949e; margin-top: 0.5rem;">
            Modèle Prophet – Précision historique : MAPE 5.1 %
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="project-box">
        <h3>Description du projet</h3>
        <p><b>Objectif :</b> Anticiper l'évolution mensuelle du prix du maïs au Bénin pour soutenir les décisions de production, d'achat et de planification.</p>
        <p><b>Description :</b> Le dashboard exploite des données historiques FAO (prix producteurs) et applique Prophet pour modéliser la tendance, la saisonnalité et la projection future.</p>
        <p><b>Résultats obtenus :</b> L'application fournit une courbe de prévision, des bornes d'incertitude, les mois de pic et de creux attendus, ainsi que des métriques de performance (MAE, MAPE).</p>
        <p><b>Explication :</b> Les graphiques montrent la direction probable des prix; la décomposition aide à comprendre ce qui relève de la tendance de fond et des effets saisonniers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────
#   SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Contrôles")
    st.markdown("---")

    horizon = st.slider("Horizon de prévision (mois)", 6, 36, 24)
    show_components = st.checkbox("Afficher décomposition", value=True)
    show_metrics = st.checkbox("Afficher métriques", value=True)

    st.markdown("### Guide rapide")
    st.markdown(
        """
        1. Choisissez l'horizon selon votre besoin (court, moyen, long terme).
        2. Activez la décomposition pour lire tendance et saisonnalité.
        3. Activez les métriques pour vérifier la qualité (MAE, MAPE).
        4. Comparez le pic et le creux pour mieux planifier les décisions.
        """
    )
    with st.expander("Explorer les graphiques", expanded=True):
        st.markdown(
            """
            - Survolez les courbes pour afficher les valeurs mensuelles.
            - Zoomez sur une période pour examiner les variations locales.
            - Lisez les bornes basse/haute pour comprendre l'incertitude.
            - Téléchargez le CSV pour une analyse avancée dans Excel/Power BI.
            """
        )

    st.markdown("---")
    st.caption("Projet par **GKF Fidele GOUSSIKINDE**")
    st.caption("Data Science & Web Development – Abomey-Calavi, Bénin 🇧🇯")

# ────────────────────────────────────────────────
#   CHARGEMENT DONNÉES (ton code existant)
# ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED = BASE_DIR / "data" / "processed" / "maize_prices_monthly.csv"
DEFAULT_RAW = BASE_DIR / "data" / "raw" / "producer-prices_ben.csv"

@st.cache_data
def load_data():
    if DEFAULT_PROCESSED.exists():
        df = pd.read_csv(DEFAULT_PROCESSED)
        if {"ds", "y"}.issubset(df.columns):
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            return df[["ds", "y"]].dropna().sort_values("ds")
    if DEFAULT_RAW.exists():
        raw = pd.read_csv(DEFAULT_RAW)
        maize = raw[
            (raw["Item"] == "Maize (corn)") &
            (raw["Element"] == "Producer Price (LCU/tonne)") &
            (raw["Months"] != "Annual value")
        ].copy()
        maize["ds"] = pd.to_datetime(maize["StartDate"], errors="coerce")
        maize["y"] = pd.to_numeric(maize["Value"], errors="coerce")
        return maize[["ds", "y"]].dropna().sort_values("ds")
    st.error("Aucune donnée trouvée.")
    st.stop()

df_historical = load_data()

# ────────────────────────────────────────────────
#   MODÈLE PROPHET
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
#   LAYOUT PRINCIPAL
# ────────────────────────────────────────────────
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Observations", f"{len(df_historical):,}", help="Nombre de mois historiques")
kpi2.metric("Date début", df_historical["ds"].min().strftime("%Y-%m"))
kpi3.metric("Date fin", df_historical["ds"].max().strftime("%Y-%m"))

st.markdown("---")

col_main, col_side = st.columns([4, 2])

with col_main:
    st.subheader("Historique et Prévisions")
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(
        height=650,
        title="Évolution et projection du prix du maïs (FCFA/tonne)",
        xaxis_title="Date",
        yaxis_title="Prix",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    if show_components:
        with st.expander("Décomposition du modèle"):
            fig_comp = plot_components_plotly(m, forecast)
            fig_comp.update_layout(
                height=700,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

with col_side:
    st.subheader("Prévisions clés")

    future_forecast = forecast[forecast["ds"] > df_historical["ds"].max()]

    peak = future_forecast.loc[future_forecast["yhat"].idxmax()]
    trough = future_forecast.loc[future_forecast["yhat"].idxmin()]

    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Pic attendu", f"{peak['yhat']:,.0f} FCFA", delta=peak["ds"].strftime("%b %Y"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Creux attendu", f"{trough['yhat']:,.0f} FCFA", delta=trough["ds"].strftime("%b %Y"), delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

    preview = (
        future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(
            columns={
                "ds": "Date",
                "yhat": "Prévision",
                "yhat_lower": "Borne basse",
                "yhat_upper": "Borne haute",
            }
        )
        .tail(horizon)
    )
    st.dataframe(preview, use_container_width=True)

    st.download_button(
        "Téléchargez prévisions CSV",
        forecast.to_csv(index=False).encode('utf-8'),
        "previsions_mais_benin.csv",
        "text/csv"
    )

if show_metrics:
    st.markdown("---")
    st.subheader("Performance historique")
    forecast_in = forecast[forecast["ds"].isin(df_historical["ds"])]
    y_true = df_historical.set_index("ds").loc[forecast_in["ds"], "y"]
    y_pred = forecast_in["yhat"]
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:,.0f} FCFA/tonne")
    c2.metric("MAPE", f"{mape:.1f} %")

st.markdown("---")
st.caption("Données : FAO Producer Prices | Modèle : Prophet | Dashboard : Streamlit")
st.caption("Projet par **Fidele GOUSSIKINDE** – Data Science & Web Dev – Abomey-Calavi, Bénin 🇧🇯")
