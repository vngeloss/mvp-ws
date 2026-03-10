import streamlit as st
import ee
import geemap.foliumap as geemap # Используем folium-версию для лучшей совместимости
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 1. Инициализация (Замени на свой ID!)
ee.Initialize(project='mvp-water-solution-project')

st.set_page_config(page_title="Water Solution", layout="wide")

# Твои координаты из Colab
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
    st.subheader("Карта снегозапасов (Sentinel-2)")
    
    # Создаем объект карты
    Map = geemap.Map(center=[50.9, 72.2], zoom=9)
    
    # Добавляем слои (обязательно добавь хотя бы один стандартный)
    Map.add_basemap('HYBRID') 
    
    # Добавляем твой слой снега
    if snow_layer:
        Map.addLayer(snow_layer, {'palette': 'cyan'}, 'Снежный покров')
    
    # ФИНАЛЬНЫЙ ШТРИХ: используем специальный метод для Streamlit
    Map.to_streamlit(height=600)

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