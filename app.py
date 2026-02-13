import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import base64

# Configuration page + thème
st.set_page_config(
    page_title="Prévision Prix Maïs Bénin",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour rendre ça plus beau
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stApp {background-image: linear-gradient(rgba(0,0,0,0.05), rgba(0,0,0,0.05));}
    h1 {color: #2c7d3b; text-align: center; font-family: 'Segoe UI', sans-serif;}
    .stMetric {background-color: #e8f5e9; border-radius: 10px; padding: 10px;}
    .stExpander {border: 1px solid #d4edda; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Header avec bannière
st.markdown(
    """
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #006400, #228B22); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>🌽 Prévision des Prix du Maïs au Bénin</h1>
        <p style="font-size: 1.2em;">Modèle Prophet – MAPE historique : 10.6 % | Données FAO & indices CPI</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar stylée
with st.sidebar:
    st.header("⚙️ Contrôles")
    st.markdown("---")
    
    horizon = st.slider("Horizon de prévision (mois)", 6, 36, 24, help="Combien de mois voulez-vous prévoir ?")
    show_components = st.checkbox("Afficher décomposition (trend + saisonnalité)", value=True)
    show_metrics = st.checkbox("Afficher métriques détaillées", value=True)
    theme = st.radio("Thème", ["Clair", "Sombre"], index=0)
    
    if theme == "Sombre":
        st.markdown("""<style> .main {background-color: #0e1117;} </style>""", unsafe_allow_html=True)

# Chargement données (adapte le chemin si besoin)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/maize_prices_monthly.csv')  # Crée ce CSV depuis ton notebook si pas fait
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    except:
        st.warning("Données historiques non trouvées. Utilisation d'un exemple.")
        dates = pd.date_range(start='2016-01-01', periods=100, freq='MS')
        return pd.DataFrame({'ds': dates, 'y': [150000 + i*1000 for i in range(100)]})

df_historical = load_data()

# Modèle Prophet
@st.cache_resource
def get_model():
    m = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative',
        yearly_seasonality=True
    )
    m.add_seasonality(name='semiannual', period=6, fourier_order=10)
    m.fit(df_historical)
    return m

m = get_model()

# Prévisions
future = m.make_future_dataframe(periods=horizon, freq='MS')
forecast = m.predict(future)

# Layout principal
col_main, col_side = st.columns([4, 2])

with col_main:
    st.subheader("📊 Historique et Prévisions")
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(
        height=600,
        title="Évolution et prévisions du prix du maïs (FCFA/tonne)",
        xaxis_title="Date",
        yaxis_title="Prix (FCFA/tonne)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    if show_components:
        with st.expander("🔍 Décomposition du modèle (trend + saisonnalité)", expanded=False):
            fig_components = plot_components_plotly(m, forecast)
            fig_components.update_layout(height=700)
            st.plotly_chart(fig_components, use_container_width=True)

with col_side:
    st.subheader("🔮 Prévisions clés")
    
    future_forecast = forecast[forecast['ds'] > df_historical['ds'].max()]
    
    # Pic et creux stylés
    peak = future_forecast.loc[future_forecast['yhat'].idxmax()]
    trough = future_forecast.loc[future_forecast['yhat'].idxmin()]
    
    st.metric(
        label="**Pic attendu**",
        value=f"{peak['yhat']:,.0f} FCFA",
        delta=f"{peak['ds'].strftime('%b %Y')}",
        delta_color="normal"
    )
    
    st.metric(
        label="**Creux attendu**",
        value=f"{trough['yhat']:,.0f} FCFA",
        delta=f"{trough['ds'].strftime('%b %Y')}",
        delta_color="inverse"
    )
    
    # Tableau prévisions
    st.dataframe(
        future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        .rename(columns={'ds': 'Date', 'yhat': 'Prévision', 'yhat_lower': 'Borne basse', 'yhat_upper': 'Borne haute'})
        .tail(horizon)
        .style.format({
            'Prévision': '{:,.0f}',
            'Borne basse': '{:,.0f}',
            'Borne haute': '{:,.0f}'
        })
        .set_properties(**{'text-align': 'center'})
    )
    
    # Export CSV
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger prévisions (CSV)",
        data=csv,
        file_name="previsions_mais_benin.csv",
        mime="text/csv"
    )

# Métriques en bas
if show_metrics:
    st.markdown("---")
    st.subheader("📏 Performance du modèle (sur données historiques)")
    forecast_in = forecast[forecast['ds'].isin(df_historical['ds'])]
    y_true = df_historical.set_index('ds').loc[forecast_in['ds'], 'y']
    y_pred = forecast_in['yhat']
    
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("MAE (erreur absolue moyenne)", f"{mae:,.0f} FCFA/tonne", help="Erreur typique par tonne")
    col2.metric("MAPE (erreur relative moyenne)", f"{mape:.1f} %", help="Précision globale du modèle")

# Footer
st.markdown("---")
st.caption("Projet réalisé par **Fidele GOUSSIKINDE** – Développeur Fullstack & Data Scientist – Abomey-Calavi, Bénin 🇧🇯")
st.caption("Données sources : FAO Producer Prices & Indices CPI | Modèle : Facebook Prophet | Dashboard : Streamlit")