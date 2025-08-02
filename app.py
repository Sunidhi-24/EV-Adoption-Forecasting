import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Set Streamlit page config first thing
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Styling ===
st.markdown("""
    <style>
        body {
            background-color: #0f1419;
            color: #ffffff;
        }
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        .stMarkdown {
            color: #ffffff;
        }
        .stSelectbox > div > div {
            background-color: #2d3748;
            color: #ffffff;
        }
        .stButton > button {
            background-color: #4299e1;
            color: #ffffff;
            border: none;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #3182ce;
        }
    </style>
""", unsafe_allow_html=True)

# === Load data (must contain historical values, features, etc.) ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()



# === Navigation Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home/Dashboard", "📊 Single County Forecast", "🗺️ Interactive Map", "📈 Compare Counties", "💡 Suggestions & Submit"])

# === Tab 1: Home/Dashboard ===
with tab1:
    # Stylized title using markdown + HTML
    st.markdown("""
        <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
            🔮 EV Adoption Forecaster for Washington State
        </div>
    """, unsafe_allow_html=True)

    # Welcome subtitle
    st.markdown("""
        <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #FFFFFF;'>
            Welcome to the Electric Vehicle (EV) Adoption Forecast Dashboard
        </div>
    """, unsafe_allow_html=True)

    # Image
    st.image("ev-car-factory.jpg", use_container_width=True)
    
    # Dashboard Overview
    st.markdown("""
        <div style='text-align: left; font-size: 18px; padding-top: 20px; color: #FFFFFF;'>
            <h3>📊 Dashboard Overview</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_counties = df['County'].nunique()
        st.metric("Total Counties", total_counties)
    
    with col2:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col3:
        total_ev = df['Electric Vehicle (EV) Total'].sum()
        st.metric("Total EVs", f"{total_ev:,.0f}")
    
    with col4:
        avg_ev_per_county = total_ev / total_counties
        st.metric("Avg EVs per County", f"{avg_ev_per_county:.0f}")
    
    # Top Counties - Interactive Chart
    st.subheader("🏆 Top 10 Counties by EV Adoption")
    top_counties = df.groupby('County')['Electric Vehicle (EV) Total'].sum().sort_values(ascending=False).head(10)
    
    # Create interactive Plotly chart
    fig = px.bar(
        x=top_counties.values,
        y=top_counties.index,
        orientation='h',
        title="Top Counties by Total EV Count",
        labels={'x': 'Total EV Count', 'y': 'County'},
        color=top_counties.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_size=16,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

# === Tab 2: Single County Forecast ===
with tab2:
    st.markdown("""
        <div style='text-align: center; font-size: 28px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
            📊 Single County EV Forecast
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: left; font-size: 18px; padding-top: 10px; color: #FFFFFF;'>
            Select a county and see the forecasted EV adoption trend for the next 3 years.
        </div>
    """, unsafe_allow_html=True)
    
    # === County dropdown ===
    county_list = sorted(df['County'].dropna().unique().tolist())
    county = st.selectbox("Select a County", county_list)

    if county not in df['County'].unique():
        st.warning(f"County '{county}' not found in dataset.")
        st.stop()

    # Add modern progress indicator
    with st.spinner(f"🔄 Analyzing {county} County data..."):
        county_df = df[df['County'] == county].sort_values("Date")
        county_code = county_df['county_encoded'].iloc[0]

    # Display county info card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Historical Records", len(county_df))
    with col2:
        st.metric("Total Historical EVs", f"{county_df['Electric Vehicle (EV) Total'].sum():,.0f}")
    with col3:
        st.metric("Avg Monthly EVs", f"{county_df['Electric Vehicle (EV) Total'].mean():.1f}")

    # === Forecasting ===
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    future_rows = []
    forecast_horizon = 36

    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }
        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)

        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    # === Combine Historical + Forecast for Cumulative Plot ===
    historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_cum['Source'] = 'Historical'
    historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

    forecast_df = pd.DataFrame(future_rows)
    forecast_df['Source'] = 'Forecast'
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

    combined = pd.concat([
        historical_cum[['Date', 'Cumulative EV', 'Source']],
        forecast_df[['Date', 'Cumulative EV', 'Source']]
    ], ignore_index=True)

    # === Plot Interactive Cumulative Graph ===
    st.subheader(f"📊 Cumulative EV Forecast for {county} County")
    
    # Create interactive Plotly chart
    fig = go.Figure()
    
    # Add historical data
    historical_data = combined[combined['Source'] == 'Historical']
    fig.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['Cumulative EV'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6)
    ))
    
    # Add forecast data
    forecast_df = pd.DataFrame(future_rows)
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Cumulative EV'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#10b981', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Cumulative EV Trend - {county} (3 Years Forecast)",
        xaxis_title="Date",
        yaxis_title="Cumulative EV Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_size=16,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

    # === Compare historical and forecasted cumulative EVs ===
    historical_total = historical_cum['Cumulative EV'].iloc[-1]
    forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

    if historical_total > 0:
        forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
        trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
        
        st.markdown(f"""
        <div style='background-color: #1e3a8a; color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6; margin: 10px 0;'>
            <strong>Based on the graph, EV adoption in {county} is expected to show a {trend} of {forecast_growth_pct:.2f}% over the next 3 years.</strong>
        </div>
        """, unsafe_allow_html=True)
        

            
    else:
        st.warning("Historical EV total is zero, so percentage forecast change can't be computed.")

# === Tab 3: Interactive Map ===
with tab3:
    st.markdown("""
        <div style='text-align: center; font-size: 28px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
            🗺️ Interactive EV Adoption Map
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: left; font-size: 18px; padding-top: 10px; color: #FFFFFF;'>
            Explore EV adoption patterns across Washington State counties on an interactive map.
        </div>
    """, unsafe_allow_html=True)

    # Calculate county statistics for mapping
    county_stats = df.groupby('County').agg({
        'Electric Vehicle (EV) Total': ['sum', 'mean', 'count']
    }).round(2)
    county_stats.columns = ['Total EVs', 'Avg Monthly EVs', 'Data Points']
    county_stats = county_stats.reset_index()
    
    # Create color scale for EV adoption
    max_evs = county_stats['Total EVs'].max()
    county_stats['Color_Intensity'] = (county_stats['Total EVs'] / max_evs * 255).astype(int)
    
    # Display map statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Counties", len(county_stats))
    with col2:
        st.metric("Highest EV County", county_stats.loc[county_stats['Total EVs'].idxmax(), 'County'])
    with col3:
        st.metric("Total EVs Statewide", f"{county_stats['Total EVs'].sum():,.0f}")
    
    # Create interactive map visualization
    st.subheader("🗺️ Washington State EV Adoption Heatmap")
    
    # Create a heatmap-style visualization using bar chart
    fig = px.bar(
        county_stats.sort_values('Total EVs', ascending=True),
        x='Total EVs',
        y='County',
        orientation='h',
        color='Total EVs',
        color_continuous_scale='viridis',
        title="EV Adoption by County (Total EVs)",
        labels={'Total EVs': 'Total Electric Vehicles', 'County': 'Washington Counties'},
        hover_data=['Avg Monthly EVs', 'Data Points']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_size=16,
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add county ranking table
    st.subheader("🏆 County Rankings")
    ranking_data = county_stats.sort_values('Total EVs', ascending=False)[['County', 'Total EVs', 'Avg Monthly EVs', 'Data Points']].copy()
    ranking_data['Rank'] = range(1, len(ranking_data) + 1)
    ranking_data = ranking_data[['Rank', 'County', 'Total EVs', 'Avg Monthly EVs', 'Data Points']]
    
    # Display as a styled table
    st.dataframe(
        ranking_data,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Total EVs": st.column_config.NumberColumn("Total EVs", format="%d"),
            "Avg Monthly EVs": st.column_config.NumberColumn("Avg Monthly EVs", format="%.1f"),
            "Data Points": st.column_config.NumberColumn("Data Points", format="%d")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # County selection for detailed view
    st.subheader("📍 Select County for Detailed View")
    selected_county = st.selectbox("Choose a county to view detailed statistics:", county_stats['County'].tolist())
    
    if selected_county:
        county_data = county_stats[county_stats['County'] == selected_county].iloc[0]
        
        # Display detailed county information
        st.markdown(f"""
        <div style='background-color: #1e3a8a; color: #ffffff; padding: 20px; border-radius: 15px; border-left: 8px solid #3b82f6; margin: 15px 0;'>
            <h4 style='margin: 0 0 15px 0;'>📊 {selected_county} County Statistics</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                <div>
                    <strong>Total EVs:</strong> {county_data['Total EVs']:,.0f}
                </div>
                <div>
                    <strong>Average Monthly EVs:</strong> {county_data['Avg Monthly EVs']:.1f}
                </div>
                <div>
                    <strong>Data Points:</strong> {county_data['Data Points']}
                </div>
                <div>
                    <strong>State Ranking:</strong> #{county_stats[county_stats['Total EVs'] >= county_data['Total EVs']].shape[0]} of {len(county_stats)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show county's EV trend
        county_trend = df[df['County'] == selected_county].sort_values('Date')
        if len(county_trend) > 0:
            st.subheader(f"📈 {selected_county} County EV Trend")
            
            trend_fig = px.line(
                county_trend,
                x='Date',
                y='Electric Vehicle (EV) Total',
                title=f"Monthly EV Registrations - {selected_county} County",
                labels={'Electric Vehicle (EV) Total': 'EV Count', 'Date': 'Month'}
            )
            
            trend_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font_size=14
            )
            
            trend_fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            trend_fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(trend_fig, use_container_width=True)

# === Tab 4: Compare Counties ===
with tab4:
    st.markdown("""
        <div style='text-align: center; font-size: 28px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
            📈 Compare EV Adoption Trends
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: left; font-size: 18px; padding-top: 10px; color: #FFFFFF;'>
            Compare EV adoption trends for up to 3 counties.
        </div>
    """, unsafe_allow_html=True)

    multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

    if multi_counties:
        # Show progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        comparison_data = []
        comparison_horizon = 60  # 5 years forecast for comparison

        for idx, cty in enumerate(multi_counties):
            status_text.text(f"Processing {cty} County... ({idx+1}/{len(multi_counties)})")
            
            cty_df = df[df['County'] == cty].sort_values("Date")
            cty_code = cty_df['county_encoded'].iloc[0]

            hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
            cum_ev = list(np.cumsum(hist_ev))
            months_since = cty_df['months_since_start'].max()
            last_date = cty_df['Date'].max()

            future_rows_cty = []
            for i in range(1, comparison_horizon + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                months_since += 1
                lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                recent_cum = cum_ev[-6:]
                ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

                new_row = {
                    'months_since_start': months_since,
                    'county_encoded': cty_code,
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_slope
                }
                pred = model.predict(pd.DataFrame([new_row]))[0]
                future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

                hist_ev.append(pred)
                if len(hist_ev) > 6:
                    hist_ev.pop(0)

                cum_ev.append(cum_ev[-1] + pred)
                if len(cum_ev) > 6:
                    cum_ev.pop(0)

            hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

            fc_df = pd.DataFrame(future_rows_cty)
            fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

            combined_cty = pd.concat([
                hist_cum[['Date', 'Cumulative EV']],
                fc_df[['Date', 'Cumulative EV']]
            ], ignore_index=True)

            combined_cty['County'] = cty
            comparison_data.append(combined_cty)
            
            # Update progress
            progress_bar.progress((idx + 1) / len(multi_counties))
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Combine all counties data for plotting
        comp_df = pd.concat(comparison_data, ignore_index=True)

        # Plot with optimized settings
        st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
        
        # Use more efficient plotting
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#1c1c1c')
        
        colors = ['#3b82f6', '#ef4444', '#10b981']  # Blue, Red, Green
        for idx, (cty, group) in enumerate(comp_df.groupby('County')):
            color = colors[idx % len(colors)]
            ax.plot(group['Date'], group['Cumulative EV'], 
                   marker='o', label=cty, color=color, linewidth=2, markersize=4)
        
        ax.set_title("EV Adoption Trends: Historical + 5-Year Forecast", fontsize=16, color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Cumulative EV Count", color='white')
        ax.grid(True, alpha=0.2, color='#374151')
        ax.set_facecolor("#1c1c1c")
        ax.tick_params(colors='white')
        ax.legend(title="County", framealpha=0.8)
        
        # Optimize for faster rendering
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  # Close figure to free memory
        
        # Display % growth for selected counties ===
        growth_summaries = []
        for cty in multi_counties:
            cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
            historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - comparison_horizon - 1]
            forecasted_total = cty_df['Cumulative EV'].iloc[-1]

            if historical_total > 0:
                growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
            else:
                growth_summaries.append(f"{cty}: N/A (no historical data)")

        # Join all in one sentence and show with custom styling
        growth_sentence = " | ".join(growth_summaries)
        st.markdown(f"""
        <div style='background-color: #1e3a8a; color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6; margin: 10px 0;'>
            <strong>Forecasted EV adoption growth over next 5 years — {growth_sentence}</strong>
        </div>
        """, unsafe_allow_html=True)



# === Tab 5: Suggestions & Submit ===
with tab5:
    st.markdown("""
        <div style='text-align: center; font-size: 28px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
            💡 Suggestions & Submit
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: left; font-size: 18px; padding-top: 10px; color: #FFFFFF;'>
            Share your feedback and suggestions for improving the EV Forecast Dashboard.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📬 Feedback & Suggestions")
    with st.form("feedback_form"):
        email = st.text_input("Your Email")
        thoughts = st.text_area("Your Thoughts or Suggestions")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if email and thoughts:
                st.success("✅ Thank you for your feedback!")
                st.balloons()
                # Add a small delay and then refresh
                import time
                time.sleep(2)
                st.rerun()
            else:
                st.error("⚠️ Please fill in both email and thoughts.")




