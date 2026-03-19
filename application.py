import streamlit as st
import ee
import geemap
from streamlit_folium import folium_static
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression  # ВОТ ЭТА СТРОКА НУЖНА
import json

if 'gcp_service_account' in st.secrets:
    creds_dict = dict(st.secrets['gcp_service_account'])
    # Исправляем формат ключа, если он считался строкой
    if isinstance(creds_dict.get('private_key'), str):
        creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
    
    credentials = ee.ServiceAccountCredentials(creds_dict['client_email'], key_data=json.dumps(creds_dict))
    ee.Initialize(credentials, project='mvp-water-solution-project')
else:
    ee.Initialize(project='mvp-water-solution-project')
st.set_page_config(page_title="Water Solution", layout="wide")


area = ee.Geometry.Rectangle([72.16, 50.81, 72.29, 51.03])
buffered_area = area.buffer(70000)

# --- ФУНКЦИЯ РАСЧЕТА ---
def get_snow_analysis():
    # Берем снимки (Март 2025)
    sentinel2 = (ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate("2025-03-01", "2025-03-30")
        .filterBounds(area)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)))
    
    image_m = sentinel2.median().clip(buffered_area)
    ndsi = image_m.normalizedDifference(['B3', 'B11']).rename('NDSI')
    snow_mask = ndsi.gt(0.4)
    
    # Считаем общую площадь (км2)
    snow_area = snow_mask.multiply(ee.Image.pixelArea())
    stats = snow_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=buffered_area,
        scale=30, # Используем 30 для скорости на сайте
        maxPixels=1e9
    )
    
    total_km2 = ee.Number(stats.get('NDSI')).divide(1e6).getInfo()
    
    # Возвращаем и число, и саму маску для отрисовки на карте
    return total_km2, snow_mask.selfMask()

# Запускаем расчет
current_snow, snow_layer = get_snow_analysis()

# --- МОДЕЛЬ ИИ ---
# Убедись, что файл data.csv лежит в той же папке!
df = pd.read_excel('data2.xlsx')
X = df[['snow_area_km2']]
y = df['water_volume_mln']
model = LinearRegression().fit(X, y)
prediction = model.predict([[current_snow]])[0]

# --- ИНТЕРФЕЙС STREAMLIT ---
st.title("🛰 Система прогнозирования паводков")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Мониторинг снежного покрова")
    Map = geemap.Map(center=[50.9, 72.2], zoom=8)
    
    # Добавь тестовый слой, чтобы понять, работает ли карта вообще
    Map.add_basemap('HYBRID') 
    
    if snow_layer:
        Map.addLayer(snow_layer, {'palette': 'cyan'}, 'Снег')
    
    # Попробуй этот метод вместо to_streamlit
    folium_static(Map)

with col2:
    st.subheader("Аналитика и прогноз")
    
    # Спидометр. Поставь макс. значение шкалы (range), равное макс. притоку из твоей таблицы
    max_water = df['water_volume_mln'].max() * 1.2
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "Ожидаемый приток (млн м³)"},
        gauge = {
            'axis': {'range': [0, max_water]}, 
            'steps': [
                {'range': [0, max_water*0.4], 'color': "lightgreen"},
                {'range': [max_water*0.4, max_water*0.7], 'color': "orange"},
                {'range': [max_water*0.7, max_water], 'color': "red"}
            ],
            'bar': {'color': "black"}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(label="Текущая площадь снега", value=f"{current_snow:.2f} км²")
    
    # Маленькая таблица для справки
    st.write("### Исторические данные")

    st.dataframe(df.tail(5))





