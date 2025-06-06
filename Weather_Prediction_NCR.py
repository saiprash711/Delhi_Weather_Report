import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
from matplotlib.lines import Line2D # Import Line2D for custom legend entries

# Configuration
st.set_page_config(page_title="NCR Forecast", layout="wide")
st.title("🌡️ NCR Weather Forecast Dashboard - Hansei Consultancy")
st.markdown("### 🌞Weather forecasts with thermal comfort metrics (2018–2028)😎")

# Constants
CITIES = {
    "Delhi": (28.7041, 77.1025),
    "Gurgaon": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
    "Faridabad": (28.4089, 77.3178)
}
YEARS = list(range(2018, 2025))
FORECAST_YEARS = list(range(2025, 2029))
SUMMER_MONTHS = ("03", "04", "05") # March, April, May are typically hot in NCR
BASE_TEMP = 18

# Weather data fetching
def fetch_weather_data(lat, lon, city):
    """Fetches daily weather data for a given city and year range from NASA POWER API."""
    all_data = []
    for year in YEARS:
        start = f"{year}0301"
        end = f"{year}0531"
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"start={start}&end={end}&latitude={lat}&longitude={lon}"
            f"&community=RE&parameters=T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN&format=JSON"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()

            data = r.json().get("properties", {}).get("parameter", {})

            if not data or not data.get("T2M_MAX"):
                st.warning(f"No valid 'T2M_MAX' data found in API response for {city} in {year}. Skipping this year.")
                continue

            for date_str in data["T2M_MAX"].keys():
                if date_str[4:6] in SUMMER_MONTHS:
                    try:
                        all_data.append({
                            "city": city,
                            "date": date_str,
                            "year": int(date_str[:4]),
                            "month": int(date_str[4:6]),
                            "t2m_max": float(data["T2M_MAX"].get(date_str, np.nan)),
                            "t2m_min": float(data["T2M_MIN"].get(date_str, np.nan)),
                            "humidity": float(data["RH2M"].get(date_str, np.nan)),
                            "wind_speed": float(data["WS2M"].get(date_str, np.nan)),
                            "solar_rad": float(data["ALLSKY_SFC_SW_DWN"].get(date_str, np.nan))
                        })
                    except (ValueError, TypeError):
                        st.warning(f"Non-numeric or missing data encountered for {city} on {date_str}. Skipping this date.")
                        continue
        except requests.exceptions.HTTPError as e:
            st.warning(f"HTTP error for {city} in {year} (Status {e.response.status_code}): {e.response.text}. Skipping.")
        except requests.exceptions.ConnectionError:
            st.warning(f"Connection error for {city} in {year}. Check internet connection. Skipping.")
        except requests.exceptions.Timeout:
            st.warning(f"Timeout fetching data for {city} in {year}. API might be slow. Skipping.")
        except requests.exceptions.RequestException as e:
            st.warning(f"General request error for {city} in {year}: {e}. Skipping.")
        except Exception as e:
            st.warning(f"{city} {year} general error during API call or processing: {str(e)}. Skipping.")
    return pd.DataFrame(all_data)

def calculate_heat_index(T, RH):
    """Calculates the heat index using Steadman's formula."""
    if pd.isna(T) or pd.isna(RH):
        return np.nan
    return -8.78469475556 + 1.61139411 * T + 2.33854883889 * RH - 0.14611605 * T * RH \
           - 0.012308094 * T**2 - 0.0164248277778 * RH**2 + 0.002211732 * T**2 * RH \
           + 0.00072546 * T * RH**2 - 0.000003582 * T**2 * RH**2

def calculate_dew_point(T, RH):
    """Calculates the dew point temperature."""
    if pd.isna(T) or pd.isna(RH):
        return np.nan
    a, b = 17.27, 237.7
    RH_clamped = max(0.01, min(100, RH))
    alpha = ((a * T) / (b + T)) + np.log(RH_clamped / 100.0)
    return (b * alpha) / (a - alpha)

@st.cache_data(ttl=3600)
def load_all_data():
    """Loads and processes weather data for all defined cities."""
    dfs = []
    for city, (lat, lon) in CITIES.items():
        df_city = fetch_weather_data(lat, lon, city)
        if not df_city.empty:
            dfs.append(df_city)
            st.info(f"Successfully loaded {len(df_city)} daily data points for **{city}**.")
        else:
            st.warning(f"No daily data fetched for **{city}**. This city might not appear in plots. Check API connectivity or city coordinates.")

    if not dfs:
        st.error("No city data loaded at all. Please check API connectivity, city coordinates, or time ranges for all listed cities.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    # Robust date conversion
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    except ValueError:
        st.error("Error converting date strings to datetime objects. Ensure dates are in `%Y%m%d` format.")
        df["date"] = pd.NaT # Set to Not a Time to avoid further errors

    initial_rows = len(df)
    df.dropna(subset=["t2m_max", "t2m_min"], inplace=True)
    if len(df) < initial_rows:
        st.warning(f"Removed {initial_rows - len(df)} rows due to missing min/max temperature data after initial fetch.")

    df["t2m_avg_for_comfort"] = (df["t2m_max"] + df["t2m_min"]) / 2
    df["heat_index"] = df.apply(lambda x: calculate_heat_index(x["t2m_avg_for_comfort"], x["humidity"]), axis=1)
    df["dew_point"] = df.apply(lambda x: calculate_dew_point(x["t2m_avg_for_comfort"], x["humidity"]), axis=1)

    df["cdd"] = df["t2m_max"].apply(lambda x: max(x - BASE_TEMP, 0))

    df.dropna(subset=["heat_index", "dew_point", "cdd"], inplace=True)

    return df

# Button to clear cache and rerun
if st.sidebar.button("Clear Cache & Refetch Data"):
    st.cache_data.clear()
    st.rerun()

df = load_all_data()

yearly_avg = df.groupby(["city", "year"]).agg(
    t2m_max=("t2m_max", "mean"),
    t2m_min=("t2m_min", "mean"),
    humidity=("humidity", "mean"),
    wind_speed=("wind_speed", "mean"),
    solar_rad=("solar_rad", "mean"),
    heat_index=("heat_index", "mean"),
    dew_point=("dew_point", "mean"),
    cdd=("cdd", "sum")
).reset_index()

# --- DIAGNOSTIC CHECKS FOR MISSING DATA FOR PLOTTING ---
st.subheader("Data Availability Check for Historical Plots")
for city in CITIES:
    city_data_for_plot = yearly_avg[yearly_avg["city"] == city]
    if city_data_for_plot.empty:
        st.error(f"❌ No aggregated historical data found for **{city}** to plot. This city's line will be missing.")
        st.write(f"Please check if API fetching returned data for {city} and if there were any `NaN` values for `t2m_max` or `t2m_min` that might have caused all rows for {city} to be dropped.")
    else:
        nan_check_max = city_data_for_plot['t2m_max'].isna()
        nan_check_min = city_data_for_plot['t2m_min'].isna()
        if nan_check_max.any() or nan_check_min.any():
            st.warning(f"⚠️ **{city}** has missing (NaN) max or min temperature data for year(s): {city_data_for_plot[nan_check_max | nan_check_min]['year'].tolist()}. These points will not be plotted.")
        st.info(f"✅ Data for **{city}** available for years: {city_data_for_plot['year'].tolist()}.")
        with st.expander(f"View raw yearly_avg data for {city}"):
            st.dataframe(city_data_for_plot)
# --- END DIAGNOSTIC CHECKS ---


# Prophet forecasting function
def prophet_forecast(df_city_data, target_col):
    """Trains a Prophet model and generates a forecast for the target column."""
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    train_df = df_city_data[df_city_data["year"] < 2024][["year", target_col]].copy()
    train_df.columns = ["ds", "y"]
    train_df["ds"] = pd.to_datetime(train_df["ds"], format="%Y")

    train_df.dropna(subset=['y'], inplace=True)

    if train_df.empty or len(train_df) < 2:
        st.warning(f"Not enough non-NaN historical data ({len(train_df)} points) for {target_col} to train Prophet model for this city. Requires at least 2 non-NaN data points for effective forecasting.")
        return pd.DataFrame(columns=["ds", "yhat"])

    try:
        model.fit(train_df)
        future = pd.DataFrame({"ds": [datetime(y, 1, 1) for y in FORECAST_YEARS]})
        forecast = model.predict(future)
        forecast["ds"] = forecast["ds"].dt.year
        return forecast
    except Exception as e:
        st.error(f"Prophet forecasting failed for {target_col} for this city: {e}")
        return pd.DataFrame(columns=["ds", "yhat"])


# Generate forecasts for all cities and metrics
all_forecasts = {}
for city in CITIES:
    city_data = yearly_avg[yearly_avg["city"] == city]

    if city_data['t2m_max'].count() < 2 or city_data['t2m_min'].count() < 2 or city_data['humidity'].count() < 2:
        st.warning(f"Insufficient non-NaN historical data for **{city}** for robust Prophet forecasting (needs at least 2 years of data). Skipping forecast for this city.")
        df_forecast = pd.DataFrame(columns=["year", "city", "Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"])
    else:
        max_temp_forecast = prophet_forecast(city_data, "t2m_max")
        min_temp_forecast = prophet_forecast(city_data, "t2m_min")
        hum_forecast = prophet_forecast(city_data, "humidity")

        if (not max_temp_forecast.empty and not min_temp_forecast.empty and not hum_forecast.empty and
            len(max_temp_forecast) == len(min_temp_forecast) == len(hum_forecast)):

            df_forecast = pd.DataFrame({
                "year": max_temp_forecast["ds"],
                "city": city,
                "Max Temp (°C)": max_temp_forecast["yhat"],
                "Min Temp (°C)": min_temp_forecast["yhat"],
                "Humidity (%)": hum_forecast["yhat"]
            })
            df_forecast["Avg Temp for HI (°C)"] = (df_forecast["Max Temp (°C)"] + df_forecast["Min Temp (°C)"]) / 2
            df_forecast["Heat Index (°C)"] = df_forecast.apply(
                lambda x: calculate_heat_index(x["Avg Temp for HI (°C)"], x["Humidity (%)"]), axis=1
            )
            df_forecast.drop(columns=["Avg Temp for HI (°C)"], inplace=True)
        else:
            st.warning(f"Prophet model could not generate complete forecasts for **{city}** (mismatch in temp/hum forecast lengths or empty forecasts).")
            df_forecast = pd.DataFrame(columns=["year", "city", "Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"])

    all_forecasts[city] = df_forecast

forecast_combined = pd.concat(all_forecasts.values(), ignore_index=True)
forecast_combined = forecast_combined.round(1)
forecast_combined.dropna(subset=["Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"], inplace=True)


# ---- DISPLAY SECTION ----

st.header("📊 Historical Temperature Trends (Summer Months)")
fig, ax = plt.subplots(figsize=(12, 6))

plot_data_hist_display = yearly_avg[yearly_avg['city'].isin(CITIES.keys())].copy()
plot_data_hist_display.dropna(subset=['t2m_max', 't2m_min'], inplace=True)

# Add a slight vertical offset to avoid label overlap for cities that might have similar temperatures
# These are heuristic values, adjust them slightly if labels still overlap
city_offsets = {
    "Delhi": 0.15,
    "Gurgaon": -0.15,
    "Noida": 0.05,
    "Faridabad": -0.05
}

if not plot_data_hist_display.empty:
    # Get a consistent color palette for cities
    city_colors = {city: sns.color_palette()[i] for i, city in enumerate(CITIES.keys())}

    # Plot Max Temperature and Min Temperature for each city
    for city in CITIES.keys():
        city_df = plot_data_hist_display[plot_data_hist_display['city'] == city]
        if not city_df.empty:
            color = city_colors.get(city, 'gray')
            offset = city_offsets.get(city, 0)

            # Plot Max Temp line and add labels
            sns.lineplot(x=city_df["year"], y=city_df["t2m_max"], color=color,
                         label=f"{city} (Max)", ax=ax, marker="o", linestyle="-", zorder=2)
            for x_val, y_val in zip(city_df["year"], city_df["t2m_max"]):
                ax.text(x_val, y_val + offset, f'{y_val:.1f}', color=color,
                        fontsize=9, ha='center', va='bottom' if offset >= 0 else 'top', zorder=3)

            # Plot Min Temp line and add labels
            sns.lineplot(x=city_df["year"], y=city_df["t2m_min"], color=color,
                         label=f"{city} (Min)", ax=ax, marker="o", linestyle="--", zorder=2)
            for x_val, y_val in zip(city_df["year"], city_df["t2m_min"]):
                ax.text(x_val, y_val - offset, f'{y_val:.1f}', color=color,
                        fontsize=9, ha='center', va='top' if offset >= 0 else 'bottom', zorder=3)

    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Average Summer Max and Min Temperature Trends (2018-2024)")
    ax.legend(title="City & Temperature Type", loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside

else:
    ax.text(0.5, 0.5, "No historical temperature data available to plot after cleaning.", transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='red')
fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent labels/legend from being cut off
st.pyplot(fig)

# Separator for visual clarity in the app
st.markdown("---")

st.header("❄️ Cooling Degree Days (Summer Months)")
fig, ax = plt.subplots(figsize=(12, 6))
plot_data_cdd = yearly_avg[yearly_avg['city'].isin(CITIES.keys())].dropna(subset=['cdd'])
if not plot_data_cdd.empty:
    barplot = sns.barplot(data=plot_data_cdd, x="year", y="cdd", hue="city", ax=ax)
    for container in barplot.containers:
        # Increase fontsize for bar labels
        barplot.bar_label(container, fmt='%.1f', label_type='edge', fontsize=9, padding=3) # Added padding
    ax.set_ylabel(f"Cooling Degree Days (Base {BASE_TEMP}°C)")
    ax.set_title("Yearly Cooling Degree Days (2018-2024)")
    ax.legend(title="City")
else:
    ax.text(0.5, 0.5, "No historical CDD data available to plot after cleaning.", transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='red')
fig.tight_layout()
st.pyplot(fig)

# Separator for visual clarity in the app
st.markdown("---")

st.header("🌡️ Heat Index Trends (Summer Months)")
fig, ax = plt.subplots(figsize=(12, 6))
plot_data_hi = yearly_avg[yearly_avg['city'].isin(CITIES.keys())].dropna(subset=['heat_index'])

# Apply offsets to heat index trends as well
city_offsets_hi = {
    "Delhi": 0.15,
    "Gurgaon": -0.15,
    "Noida": 0.05,
    "Faridabad": -0.05
}

if not plot_data_hi.empty:
    city_colors = {city: sns.color_palette()[i] for i, city in enumerate(CITIES.keys())}
    for city in CITIES.keys():
        city_df = plot_data_hi[plot_data_hi['city'] == city]
        if not city_df.empty:
            color = city_colors.get(city, 'gray')
            offset = city_offsets_hi.get(city, 0)
            
            sns.lineplot(x=city_df["year"], y=city_df["heat_index"], color=color,
                         label=f"{city}", ax=ax, marker="o", zorder=2)

            for x_val, y_val in zip(city_df["year"], city_df["heat_index"]):
                ax.text(x_val, y_val + offset, f'{y_val:.1f}', color=color,
                        fontsize=9, ha='center', va='bottom' if offset >= 0 else 'top', zorder=3)

    ax.set_ylabel("Heat Index (°C)")
    ax.set_title("Average Summer Heat Index Trends (2018-2024)")
    ax.legend(title="City", loc='center left', bbox_to_anchor=(1, 0.5))
else:
    ax.text(0.5, 0.5, "No historical Heat Index data available to plot after cleaning.", transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='red')
fig.tight_layout(rect=[0, 0, 0.85, 1])
st.pyplot(fig)

# Separator for visual clarity in the app
st.markdown("---")

st.header("💨 Wind Speed vs Solar Radiation (Summer Months)")
fig, ax = plt.subplots(figsize=(10, 6))
plot_data_ws_sr = yearly_avg[yearly_avg['city'].isin(CITIES.keys())].dropna(subset=['solar_rad', 'wind_speed'])

if not plot_data_ws_sr.empty:
    # Use the same city_colors mapping for consistency
    city_colors = {city: sns.color_palette()[i] for i, city in enumerate(CITIES.keys())}

    scatter_plot = sns.scatterplot(data=plot_data_ws_sr, x="solar_rad", y="wind_speed", hue="city", size="t2m_max", sizes=(50, 400), ax=ax, alpha=0.8)
    
    # Add data labels to the scatter plot
    for i, row in plot_data_ws_sr.iterrows():
        text_color = city_colors.get(row['city'], 'black') # Get color directly from our pre-defined map
        ax.text(row["solar_rad"], row["wind_speed"],
                f'({row["solar_rad"]:.0f}, {row["wind_speed"]:.1f})', # Format to (Solar, Wind)
                fontsize=8, ha='left', va='bottom', # Adjusted ha/va
                color=text_color)

    ax.set_xlabel("Average Solar Radiation (Wh/m²/day)")
    ax.set_ylabel("Average Wind Speed (m/s)")
    ax.set_title("Average Summer Wind Speed vs. Solar Radiation (2018-2024)")
    ax.legend(title="City", loc='center left', bbox_to_anchor=(1, 0.5))
else:
    ax.text(0.5, 0.5, "No historical Wind Speed or Solar Radiation data available to plot after cleaning.", transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='red')
fig.tight_layout(rect=[0, 0, 0.85, 1])
st.pyplot(fig)

# Separator for visual clarity in the app
st.markdown("---")

st.header("🔮 Forecasts for NCR (2025–2028) - Summer Months")
if not forecast_combined.empty:
    years_for_pivot = sorted(forecast_combined['year'].unique())
    cities_for_pivot = list(CITIES.keys())

    idx = pd.MultiIndex.from_product([years_for_pivot, cities_for_pivot], names=['year', 'city'])
    full_df_for_pivot = pd.DataFrame(index=idx).reset_index()

    pivot_df = pd.merge(full_df_for_pivot, forecast_combined, on=['year', 'city'], how='left')

    st.dataframe(
        pivot_df.pivot(index="year", columns="city", values=["Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"]),
        use_container_width=True
    )
else:
    st.warning("No forecasts generated. This might be due to insufficient historical data for Prophet to train or API issues. Check the warnings above.")

# Separator for visual clarity in the app
st.markdown("---")

st.header("📈 Forecast Trend Viewer")
metric = st.selectbox("Select a metric to view trend", ["Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"])
fig, ax = plt.subplots(figsize=(12, 6))

metric_map = {
    "Max Temp (°C)": "t2m_max",
    "Min Temp (°C)": "t2m_min",
    "Humidity (%)": "humidity",
    "Heat Index (°C)": "heat_index"
}

plot_generated = False
city_colors = {city: sns.color_palette()[i] for i, city in enumerate(CITIES.keys())} # Ensure consistent colors

for city in CITIES.keys(): # Iterate with index for consistent colors
    hist = yearly_avg[yearly_avg["city"] == city]
    fore = forecast_combined[forecast_combined["city"] == city]
    color = city_colors.get(city, 'gray')

    # Apply offsets for forecast trend viewer based on the metric
    current_offset = 0
    if metric in ["Max Temp (°C)", "Min Temp (°C)"]:
        current_offset = city_offsets.get(city, 0)
    elif metric == "Heat Index (°C)":
        current_offset = city_offsets_hi.get(city, 0)


    # Plot historical data
    if not hist.empty and metric_map[metric] in hist.columns and hist[metric_map[metric]].notna().any():
        y_hist_data = hist[metric_map[metric]]
        sns.lineplot(x=hist["year"], y=y_hist_data, color=color,
                     label=f"{city} (Historical)", ax=ax, marker="o", zorder=2)
        for x_val, y_val in zip(hist["year"], y_hist_data):
            ax.text(x_val, y_val + current_offset, f'{y_val:.1f}', color=color,
                    fontsize=9, ha='center', va='bottom' if current_offset >= 0 else 'top', zorder=3)
        plot_generated = True

    # Plot forecast data
    if not fore.empty and metric in fore.columns and fore[metric].notna().any():
        y_fore_data = fore[metric]
        sns.lineplot(x=fore["year"], y=y_fore_data, color=color,
                     label=f"{city} (Forecast)", ax=ax, linestyle="--", marker="o", zorder=2)
        for x_val, y_val in zip(fore["year"], y_fore_data):
            ax.text(x_val, y_val + current_offset, f'{y_val:.1f}', color=color,
                    fontsize=9, ha='center', va='bottom' if current_offset >= 0 else 'top', zorder=3)
        plot_generated = True

if plot_generated:
    ax.set_ylabel(metric)
    ax.set_title(f"Historical and Forecast Trend for {metric} (Summer Months)")
    ax.legend(title="City Data", loc='center left', bbox_to_anchor=(1, 0.5))
else:
    ax.text(0.5, 0.5, f"No sufficient data available to plot historical or forecast trends for {metric}.", transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='red')
fig.tight_layout(rect=[0, 0, 0.85, 1])
st.pyplot(fig)

# Separator for visual clarity in the app
st.markdown("---")

st.header("💡 HVAC-Specific Insights for NCR")
st.markdown("""
-   **High Heat Index**: Cities like Delhi and Faridabad might experience higher heat index values, indicating a strong need for efficient air conditioning systems.
-   **Humidity Control**: Given potentially rising dew points, particularly during the onset of monsoon, effective dehumidification systems will be crucial for indoor comfort and preventing mold growth.
-   **Cooling Load Management**: Consistent high temperatures in summer imply significant cooling loads. Prioritize energy-efficient HVAC equipment (e.g., higher SEER ratings) and potentially smart thermostats.
-   **Solar Radiation**: High solar radiation suggests opportunities for solar passive design (e.g., shading) and potential for integrating solar thermal or solar PV for HVAC systems, especially in areas like Gurgaon and Noida with newer infrastructure.
-   **Wind Speed**: While generally low, understanding wind patterns can help in natural ventilation strategies to reduce reliance on mechanical cooling during milder periods.
""")

# Separator for visual clarity in the app
st.markdown("---")

st.header("📥 Download Data")
with st.expander("Export Options"):
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Daily Data", index=False)
        yearly_avg.to_excel(writer, sheet_name="Yearly Averages", index=False)
        forecast_combined.to_excel(writer, sheet_name="Forecasts", index=False)

        combined = yearly_avg[["city", "year", "t2m_max", "t2m_min", "humidity", "heat_index"]].copy()
        combined.columns = ["city", "year", "Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"]
        forecast_export = forecast_combined[["city", "year", "Max Temp (°C)", "Min Temp (°C)", "Humidity (%)", "Heat Index (°C)"]]
        full_combined = pd.concat([combined, forecast_export], ignore_index=True).sort_values(["city", "year"])
        full_combined.to_excel(writer, sheet_name="Historical + Forecast", index=False)

    st.download_button(
        label="💾 Download Full Analysis (Excel)",
        data=excel_buffer.getvalue(),
        file_name="ncr_hvac_weather_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.success("✅ Analysis complete!")
