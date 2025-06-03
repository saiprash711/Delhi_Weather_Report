import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuration
st.set_page_config(page_title="NCR Region Forecast", layout="wide")
st.title("🌡️ NCR Region Weather Forecast Dashboard - Hansei Consultancy")
st.markdown("### 🌞Weather forecasts with thermal comfort metrics (2018–2028)😎")

# Constants - Updated for NCR cities
CITIES = {
    "Delhi": (28.61, 77.21),
    "Gurgaon": (28.45, 77.02),
    "Noida": (28.54, 77.39),
    "Faridabad": (28.41, 77.31),
    "Ghaziabad": (28.67, 77.42)
}
YEARS = list(range(2018, 2025))
FORECAST_YEARS = list(range(2025, 2029))
SUMMER_MONTHS = ("03", "04", "05", "06")  # Extended for NCR's longer summer
BASE_TEMP = 18

# Weather data fetching
def fetch_weather_data(lat, lon, city):
    all_data = []
    for year in YEARS:
        start = f"{year}0301"
        end = f"{year}0630"  # Extended to June for NCR
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"start={start}&end={end}&latitude={lat}&longitude={lon}"
            f"&community=RE&parameters=T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN&format=JSON"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                st.warning(f"⚠️ Failed to fetch {city} data for {year}.")
                continue
            data = r.json().get("properties", {}).get("parameter", {})
            for date in data["T2M_MAX"].keys():
                if date[4:6] in SUMMER_MONTHS:
                    all_data.append({
                        "city": city,
                        "date": date,
                        "year": int(date[:4]),
                        "month": int(date[4:6]),
                        "temperature": data["T2M_MAX"][date],
                        "t2m_min": data["T2M_MIN"][date],
                        "humidity": data["RH2M"][date],
                        "wind_speed": data["WS2M"][date],
                        "solar_rad": data["ALLSKY_SFC_SW_DWN"][date]
                    })
        except Exception as e:
            st.warning(f"{city} {year} error: {str(e)}")
    return pd.DataFrame(all_data)

def calculate_heat_index(T, RH):
    return -8.78469475556 + 1.61139411 * T + 2.33854883889 * RH - 0.14611605 * T * RH \
           - 0.012308094 * T**2 - 0.0164248277778 * RH**2 + 0.002211732 * T**2 * RH \
           + 0.00072546 * T * RH**2 - 0.000003582 * T**2 * RH**2

def calculate_dew_point(T, RH):
    a, b = 17.27, 237.7
    alpha = ((a * T) / (b + T)) + np.log(RH / 100.0)
    return (b * alpha) / (a - alpha)

@st.cache_data(ttl=3600)
def load_all_data():
    dfs = []
    for city, (lat, lon) in CITIES.items():
        df_city = fetch_weather_data(lat, lon, city)
        if df_city is not None:
            dfs.append(df_city)
    if not dfs:
        st.error("No city data loaded.")
        st.stop()
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["t2m_avg"] = (df["temperature"] + df["t2m_min"]) / 2
    df["heat_index"] = df.apply(lambda x: calculate_heat_index(x["t2m_avg"], x["humidity"]), axis=1)
    df["dew_point"] = df.apply(lambda x: calculate_dew_point(x["t2m_avg"], x["humidity"]), axis=1)
    df["cdd"] = df["t2m_avg"].apply(lambda x: max(x - BASE_TEMP, 0))
    return df

df = load_all_data()

yearly_avg = df.groupby(["city", "year"]).agg({
    "t2m_avg": "mean",
    "humidity": "mean",
    "wind_speed": "mean",
    "solar_rad": "mean",
    "heat_index": "mean",
    "dew_point": "mean",
    "cdd": "sum"
}).reset_index()

def prophet_forecast(df, target_col):
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    train_df = df[df["year"] < 2024][["year", target_col]].copy()
    train_df.columns = ["ds", "y"]
    train_df["ds"] = pd.to_datetime(train_df["ds"], format="%Y")
    model.fit(train_df)
    future = pd.DataFrame({"ds": [datetime(y, 1, 1) for y in FORECAST_YEARS]})
    forecast = model.predict(future)
    forecast["ds"] = forecast["ds"].dt.year
    return forecast

all_forecasts = {}
for city in CITIES:
    city_data = yearly_avg[yearly_avg["city"] == city]
    temp_forecast = prophet_forecast(city_data, "t2m_avg")
    hum_forecast = prophet_forecast(city_data, "humidity")
    df_forecast = pd.DataFrame({
        "year": temp_forecast["ds"],
        "city": city,
        "Avg Temp (°C)": temp_forecast["yhat"],
        "Humidity (%)": hum_forecast["yhat"]
    })
    df_forecast["Heat Index (°C)"] = calculate_heat_index(
        df_forecast["Avg Temp (°C)"], df_forecast["Humidity (%)"]
    )
    all_forecasts[city] = df_forecast

forecast_combined = pd.concat(all_forecasts.values(), ignore_index=True)
forecast_combined = forecast_combined.round(1)

# ---- DISPLAY SECTION ----

st.header("📊 Historical Temperature Trends")
fig, ax = plt.subplots(figsize=(12, 6))
for city in CITIES:
    sns.lineplot(data=yearly_avg[yearly_avg["city"] == city], x="year", y="t2m_avg", label=city, ax=ax, marker="o")
ax.set_ylabel("Avg Temp (°C)")
st.pyplot(fig)

st.header("❄️ Cooling Degree Days")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=yearly_avg, x="year", y="cdd", hue="city", ax=ax)
st.pyplot(fig)

st.header("🌡️ Heat Index Trends")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=yearly_avg, x="year", y="heat_index", hue="city", ax=ax, marker="o")
st.pyplot(fig)

st.header("💨 Wind Speed vs Solar Radiation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=yearly_avg, x="solar_rad", y="wind_speed", hue="city", size="t2m_avg", sizes=(20, 200), ax=ax)
st.pyplot(fig)

st.header("🔮 Forecasts (2025–2028)")
st.dataframe(
    forecast_combined.pivot(index="year", columns="city", values=["Avg Temp (°C)", "Humidity (%)", "Heat Index (°C)"]),
    use_container_width=True
)

st.header("📈 Forecast Trend Viewer")
metric = st.selectbox("Select a metric to view trend", ["Avg Temp (°C)", "Humidity (%)", "Heat Index (°C)"])
fig, ax = plt.subplots(figsize=(12, 6))
metric_map = {
    "Avg Temp (°C)": "t2m_avg",
    "Humidity (%)": "humidity",
    "Heat Index (°C)": "heat_index"
}
for city in CITIES:
    hist = yearly_avg[yearly_avg["city"] == city]
    fore = forecast_combined[forecast_combined["city"] == city]
    sns.lineplot(data=hist, x="year", y=metric_map[metric], label=f"{city} (Historical)", ax=ax, marker="o")
    sns.lineplot(data=fore, x="year", y=metric, label=f"{city} (Forecast)", ax=ax, linestyle="--", marker="o")
st.pyplot(fig)

st.header("💡 HVAC-Specific Insights for NCR")
st.markdown("""
1. **Delhi** shows extreme temperatures → Focus on high-capacity cooling systems  
2. **Gurgaon/Noida** have high humidity → Prioritize dehumidification systems  
3. **Faridabad/Ghaziabad** show high CDD → Use energy-efficient AC systems  
4. Rising heat index across NCR → Consider hybrid cooling solutions  
5. Low wind speeds in summer → Mechanical ventilation important  
""")

st.header("📥 Download Data")
with st.expander("Export Options"):
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Daily Data", index=False)
        yearly_avg.to_excel(writer, sheet_name="Yearly Averages", index=False)
        forecast_combined.to_excel(writer, sheet_name="Forecasts", index=False)

        # Combine historical + forecast for export
        combined = yearly_avg[["city", "year", "t2m_avg", "humidity", "heat_index"]].copy()
        combined.columns = ["city", "year", "Avg Temp (°C)", "Humidity (%)", "Heat Index (°C)"]
        forecast_export = forecast_combined[["city", "year", "Avg Temp (°C)", "Humidity (%)", "Heat Index (°C)"]]
        full_combined = pd.concat([combined, forecast_export], ignore_index=True).sort_values(["city", "year"])
        full_combined.to_excel(writer, sheet_name="Historical + Forecast", index=False)

    st.download_button(
        label="💾 Download Full Analysis (Excel)",
        data=excel_buffer.getvalue(),
        file_name="ncr_hvac_weather_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.success("✅ NCR Region Analysis complete!")