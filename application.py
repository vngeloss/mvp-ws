import streamlit as st
import ee
from streamlit_folium import folium_static
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample
import json
import datetime


if 'gcp_service_account' in st.secrets:
    creds_dict = dict(st.secrets['gcp_service_account'])
    if isinstance(creds_dict.get('private_key'), str):
        creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
    credentials = ee.ServiceAccountCredentials(
        creds_dict['client_email'],
        key_data=json.dumps(creds_dict)
    )
    ee.Initialize(credentials, project='mvp-water-solution-project')
else:
    ee.Initialize(project='mvp-water-solution-project')

st.set_page_config(page_title="Water Solution — Inflow prediction", layout="wide")

area = ee.Geometry.Rectangle([72.16, 50.81, 72.29, 51.03])
buffered_area = area.buffer(15000)

@st.cache_data(ttl=3600 * 6, show_spinner="Calculating snow area according to the Sentinel-2")
def get_snow_analysis(start_date: str, end_date: str, ndsi_threshold: float = 0.4, scale: int = 30):
    try:
        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(area)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .select(['B3', 'B11', 'SCL'])
        )

        def mask_clouds(img):
            scl = img.select('SCL')
            mask = (
                scl.neq(3)   # cloud shadow
                .And(scl.neq(8))  # cloud medium
                .And(scl.neq(9))  # cloud high
                .And(scl.neq(10)) # thin cirrus
            )
            return img.updateMask(mask)

        collection = sentinel2.map(mask_clouds)
        count = collection.size().getInfo()

        if count == 0:
            return None, None, 0, "Нет подходящих снимков за выбранный период"

        image_m = collection.median().clip(buffered_area)

        ndsi = image_m.normalizedDifference(['B3', 'B11']).rename('NDSI')
        snow_mask = ndsi.gt(ndsi_threshold)

        snow_area_img = snow_mask.multiply(ee.Image.pixelArea())
        stats = snow_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffered_area,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True
        )

        total_km2 = ee.Number(stats.get('NDSI')).divide(1e6).getInfo()
        return total_km2, snow_mask.selfMask(), count, None

    except Exception as e:
        return None, None, 0, str(e)

@st.cache_data
def load_data():
    df = pd.read_excel('data2.xlsx')
    # На всякий случай приводим названия
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

df = load_data()

def train_and_predict(current_snow: float, n_bootstrap: int = 400):
    """Trains the model and returns a prediction + uncertainty interval"""
    X = df[['snow_area_km2']].values
    y = df['water_volume_mln'].values

    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))

    preds = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        m = LinearRegression().fit(X[idx], y[idx])
        preds.append(m.predict([[current_snow]])[0])

    preds = np.array(preds)
    mean_pred = float(np.mean(preds))
    lower = float(np.percentile(preds, 15))
    upper = float(np.percentile(preds, 85))

    return {
        "prediction": mean_pred,
        "lower": lower,
        "upper": upper,
        "r2": r2,
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_)
    }


st.title("🛰Flood monitoring system")

with st.sidebar:
    st.header("⚙️ Parameters of Analysis")

    current_year = datetime.datetime.now().year
    year = st.selectbox(
        "Year for snow analysis",
        options=list(range(current_year, 2018, -1)),
        index=0
    )

    month_period = st.selectbox(
        "time period",
        options=[
            ("March (01–31)", f"{year}-03-01", f"{year}-03-31"),
            ("february–march", f"{year}-02-15", f"{year}-03-31"),
            ("march–april", f"{year}-03-01", f"{year}-04-15"),
        ],
        format_func=lambda x: x[0]
    )
    start_date, end_date = month_period[1], month_period[2]

    ndsi_threshold = st.slider("Border NDSI (snow)", 0.2, 0.6, 0.4, 0.05)
    scale = st.select_slider("calculation resolution (м)", options=[20, 30, 50, 100], value=30)

    st.markdown("---")
    st.caption("The inflow volume data are approximate (5 years). The model's accuracy is limited.")

# --- Расчёт снега ---
current_snow, snow_layer, image_count, error_msg = get_snow_analysis(
    start_date, end_date, ndsi_threshold, scale
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Map")

    Map = geemap.Map(center=[50.92, 72.23], zoom=9)
    Map.add_basemap('HYBRID')

    if snow_layer is not None:
        Map.addLayer(snow_layer, {'palette': ['#00FFFF'], 'opacity': 0.7}, 'Снег (NDSI)')
        Map.addLayer(area, {'color': 'yellow'}, 'Зона интереса')
    else:
        st.warning(error_msg or "Could not find the snow layer")

    folium_static(Map, width=750, height=520)

    if image_count > 0:
        st.caption(f"Used pictures from Sentinel-2: **{image_count}** | Period: {start_date} → {end_date}")

with col2:
    st.subheader("Analysis and Prediction")

    if current_snow is None:
        st.error("Could not find the snow layer")
        st.stop()

    # Метрика площади
    st.metric(
        label="Snow Area",
        value=f"{current_snow:,.1f} км²",
        help="Calculated based on the NDSI for the selected period"
    )

    # Прогноз
    result = train_and_predict(current_snow)

    # Спидометр
    max_water = max(df['water_volume_mln'].max() * 1.3, result["upper"] * 1.1)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result["prediction"],
        number={'suffix': " млн м³", 'valueformat': ".1f"},
        title={'text': "Ожидаемый приток"},
        gauge={
            'axis': {'range': [0, max_water]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, max_water * 0.35], 'color': "#d4edda"},
                {'range': [max_water * 0.35, max_water * 0.65], 'color': "#fff3cd"},
                {'range': [max_water * 0.65, max_water], 'color': "#f8d7da"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.8,
                'value': result["upper"]
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Интервал неопределённости
    st.info(
        f"**Forecast Range (70%):**  \n"
        f"{result['lower']:.1f} — {result['upper']:.1f} млн м³"
    )

    st.caption(f"R² models based on historical data: **{result['r2']:.2f}** (5 points)")

st.markdown("---")
st.subheader("Historical data and accuracy")

col3, col4 = st.columns([1, 1])

with col3:
    st.write("### TAble")
    st.dataframe(
        df.style.format({
            'snow_area_km2': '{:.1f}',
            'water_volume_mln': '{:.1f}'
        }),
        use_container_width=True,
        hide_index=True
    )

with col4:
    st.write("### Snow -> flood")
    fig_scatter = px.scatter(
        df,
        x='snow_area_km2',
        y='water_volume_mln',
        text='year',
        labels={
            'snow_area_km2': 'Snow area(km2)',
            'water_volume_mln': 'Volume of inflow(mln m3)'
        },
        title=f"R² = {result['r2']:.2f}"
    )
    fig_scatter.update_traces(textposition='top center', marker=dict(size=12))

    # Линия регрессии
    x_line = np.linspace(df['snow_area_km2'].min() * 0.9, df['snow_area_km2'].max() * 1.05, 50)
    y_line = result['coef'] * x_line + result['intercept']
    fig_scatter.add_traces(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        name='Линейная модель',
        line=dict(color='red', dash='dash')
    ))

    # Текущая точка
    fig_scatter.add_trace(go.Scatter(
        x=[current_snow],
        y=[result['prediction']],
        mode='markers',
        marker=dict(size=16, color='orange', symbol='star'),
        name='Текущий прогноз'
    ))

    fig_scatter.update_layout(height=380, showlegend=True)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.warning(
    "⚠️ **Important:** The inflow volumes in the table are approximate (estimates)."
    "The model is based on only 5 years of data. Use the forecast only as a guide."
    "For actual operation, official data from hydrological stations is required."
)
