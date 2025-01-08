import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Загрузка данных
def load_data(file):
    data = pd.read_csv(file, parse_dates=['timestamp'])
    return data

# Рассчет скользящего среднего
def calculate_rolling_mean(data):
    data['rolling_mean'] = data['temperature'].rolling(window=30, min_periods=1).mean()
    return data

# Подсчет сезонной статистики
def calculate_seasonal_stats(data):
    stats = data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()
    return stats

# Выявление аномалии
def identify_anomalies(data, stats):
    anomalies = []
    for _, row in stats.iterrows():
        city, season = row['city'], row['season']
        mean, std = row['mean'], row['std']
        city_season_data = data[(data['city'] == city) & (data['season'] == season)]
        mask = (city_season_data['temperature'] < mean - 2 * std) | (city_season_data['temperature'] > mean + 2 * std)
        anomalies.append(city_season_data[mask])
    return pd.concat(anomalies)

# Параллельный подсчет данных
def analyze_city_season(data, city, season):
    city_season_data = data[(data['city'] == city) & (data['season'] == season)]
    mean = city_season_data['temperature'].mean()
    std = city_season_data['temperature'].std()
    mask = (city_season_data['temperature'] < mean - 2 * std) | (city_season_data['temperature'] > mean + 2 * std)
    anomalies = city_season_data[mask]
    return mean, std, anomalies

def parallel_analysis(data):
    cities_seasons = data[['city', 'season']].drop_duplicates()
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_city_season, data, row['city'], row['season']) for _, row in cities_seasons.iterrows()]
        for future in futures:
            results.append(future.result())
    return results

'''
Тестировал в ноутбуке. Скорость в среднем увеличивается в 2 раза.
Обычный анализ - 11.9 µs
Паралельный анализ - 5.01 µs
'''

# Получение даных по api ассинхорнно
async def fetch_temperature_async(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                return None
#Получение данных по api
def fetch_temperature(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['main']['temp']
    else:
        return None
'''
Прирост в скорости не заметен в ассинхронном методе т.к. у нас единичный запрос. 
Если бы нам было необохдимо делать больше количества запрос, можно было бы увеличить скорость
'''
#Реализация streamlit приложения
def main():
    st.title("Анализ температурных данных и мониторинг текущей температуры")

    # Upload historical data
    file = st.file_uploader("Загрузите исторические данные в формате CSV:", type=['csv'])
    if file:
        data = load_data(file)
        st.write("Обзор загруженных данных:", data.head())

        data = calculate_rolling_mean(data)

        parallel_results = parallel_analysis(data)


        city = st.selectbox("Выберите город:", data['city'].unique())
        if city:
            city_data = data[data['city'] == city]
            stats = calculate_seasonal_stats(city_data)
            anomalies = identify_anomalies(city_data, stats)
            st.subheader("Сезонная статистика")
            st.write(stats)
            st.subheader(f"Статистика погоды для города {city}")
            st.write(city_data.describe())

            st.subheader(f"Сезонная статистика для города {city}")
            for season in city_data['season'].unique():
                season_data = city_data[city_data['season'] == season]
                mean = season_data['temperature'].mean()
                std = season_data['temperature'].std()
                st.write(f"{season.capitalize()} - Среднее: {mean:.2f}°C, Отклонение: {std:.2f}°C")

        # Plot temperature time series with anomalies
        st.subheader("Температурные аномалии")
        plt.figure(figsize=(10, 5))
        plt.plot(data['timestamp'], data['temperature'], label='Температура')
        plt.plot(data['timestamp'], data['rolling_mean'], label='Скользящее среднее')
        plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Аномалии')
        plt.legend()
        plt.xlabel('Дата')
        plt.ylabel('Температура (°C)')
        plt.title('Температурный временной ряд')
        st.pyplot(plt)

        # Monitor current temperature
        st.subheader("Отслеживание текущей температуры")
        api_key = st.text_input("Введите OpenWeatherMap API Key:")

        if api_key:
            temp = fetch_temperature(city, api_key)
            if temp is not None:
                current_season = data[data['city'] == city]['season'].iloc[-1]
                season_stats = stats[(stats['city'] == city) & (stats['season'] == current_season)]
                mean, std = season_stats['mean'].values[0], season_stats['std'].values[0]
                normal = mean - 2 * std <= temp <= mean + 2 * std
                st.write(f"Нормальная температура для {city}: {temp}°C")
                st.write(f"Это нормальная температура для {current_season}? {'Да' if normal else 'Нет'}")
            else:
                st.error({"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."})

if __name__ == "__main__":
    main()
