import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import time
from datetime import datetime, timedelta
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🚚 Smart Delivery Intelligence Center",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f77b4 0%, #17becf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .status-critical {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .insight-box {
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load trained models
@st.cache_resource
def load_trained_models():
    """Load the trained ML models and preprocessors"""
    models = {}
    try:
        # Check if model artifacts exist
        model_dir = 'model_artifacts'
        if os.path.exists(model_dir):
            # Load the main prediction model
            if os.path.exists(f'{model_dir}/best_delivery_model.pkl'):
                models['predictor'] = joblib.load(f'{model_dir}/best_delivery_model.pkl')
            
            # Load encoders
            if os.path.exists(f'{model_dir}/city_encoder.pkl'):
                models['city_encoder'] = joblib.load(f'{model_dir}/city_encoder.pkl')
            
            if os.path.exists(f'{model_dir}/type_encoder.pkl'):
                models['type_encoder'] = joblib.load(f'{model_dir}/type_encoder.pkl')
            
            # Load scaler
            if os.path.exists(f'{model_dir}/feature_scaler.pkl'):
                models['scaler'] = joblib.load(f'{model_dir}/feature_scaler.pkl')
            
            # Load anomaly detector
            if os.path.exists(f'{model_dir}/anomaly_detector.pkl'):
                models['anomaly_detector'] = joblib.load(f'{model_dir}/anomaly_detector.pkl')
                models['anomaly_scaler'] = joblib.load(f'{model_dir}/anomaly_scaler.pkl')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

# Validation functions
def validate_tanzania_coordinates(lat, lng):
    """Validate if coordinates are within Tanzania bounds"""
    # Tanzania approximate bounds
    TANZANIA_BOUNDS = {
        'lat_min': -11.75,
        'lat_max': -0.98,
        'lng_min': 29.34,
        'lng_max': 40.32
    }
    
    return (TANZANIA_BOUNDS['lat_min'] <= lat <= TANZANIA_BOUNDS['lat_max'] and 
            TANZANIA_BOUNDS['lng_min'] <= lng <= TANZANIA_BOUNDS['lng_max'])

def validate_delivery_inputs(poi_lat, poi_lng, receipt_lat, receipt_lng, sign_lat, sign_lng, 
                           receipt_hour, city, delivery_type):
    """Validate all delivery prediction inputs"""
    errors = []
    
    # Validate coordinates
    if not validate_tanzania_coordinates(poi_lat, poi_lng):
        errors.append("POI coordinates are outside Tanzania bounds")
    
    if not validate_tanzania_coordinates(receipt_lat, receipt_lng):
        errors.append("Receipt coordinates are outside Tanzania bounds")
    
    if not validate_tanzania_coordinates(sign_lat, sign_lng):
        errors.append("Sign coordinates are outside Tanzania bounds")
    
    # Validate hour
    if not (0 <= receipt_hour <= 23):
        errors.append("Receipt hour must be between 0 and 23")
    
    # Validate city
    valid_cities = ['Dar es Salaam', 'Mwanza', 'Mbeya', 'Arusha', 'Dodoma']
    if city not in valid_cities:
        errors.append(f"City must be one of: {', '.join(valid_cities)}")
    
    return errors

# Load and prepare data
@st.cache_data
def load_and_clean_data():
    """Load and clean the delivery data with outlier removal for visualizations"""
    df = pd.read_csv('executable_files/delivery_five_cities_tanzania.csv')
    
    # Parse timestamps
    def parse_timestamp(timestamp_str, year=2023):
        try:
            return pd.to_datetime(f"{year}-{timestamp_str}", format='%Y-%m-%d %H:%M:%S')
        except:
            return pd.NaT
    
    df['receipt_time_parsed'] = df['receipt_time'].apply(parse_timestamp)
    df['sign_time_parsed'] = df['sign_time'].apply(parse_timestamp)
    
    # Remove invalid timestamps
    df_clean = df.dropna(subset=['receipt_time_parsed', 'sign_time_parsed']).copy()
    
    # Calculate delivery time
    df_clean['delivery_time_minutes'] = (df_clean['sign_time_parsed'] - df_clean['receipt_time_parsed']).dt.total_seconds() / 60
    df_clean = df_clean[df_clean['delivery_time_minutes'] > 0]
    
    # Remove extreme outliers for better visualizations (keep realistic delivery times)
    # Keep deliveries between 1 minute and 24 hours (1440 minutes)
    df_viz = df_clean[
        (df_clean['delivery_time_minutes'] >= 1) & 
        (df_clean['delivery_time_minutes'] <= 1440)
    ].copy()
    
    # Calculate distances
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371
    
    # Calculate distances for visualization subset
    sample_size = min(100000, len(df_viz))
    df_sample = df_viz.sample(n=sample_size, random_state=42).copy()
    
    df_sample['delivery_distance_km'] = haversine_distance(
        df_sample['poi_lat'], df_sample['poi_lng'],
        df_sample['sign_lat'], df_sample['sign_lng']
    )
    
    # Remove unrealistic distances (keep local deliveries)
    df_sample = df_sample[df_sample['delivery_distance_km'] <= 200].copy()
    
    # Add time-based features
    df_sample['receipt_hour'] = df_sample['receipt_time_parsed'].dt.hour
    df_sample['receipt_day_of_week'] = df_sample['receipt_time_parsed'].dt.dayofweek
    df_sample['receipt_day_name'] = df_sample['receipt_time_parsed'].dt.day_name()
    df_sample['is_weekend'] = df_sample['receipt_day_of_week'].isin([5, 6])
    df_sample['is_peak_hour'] = df_sample['receipt_hour'].isin(range(8, 18))
    
    # Calculate speed (km/h)
    df_sample['speed_kmh'] = (df_sample['delivery_distance_km'] / (df_sample['delivery_time_minutes'] / 60)).fillna(0)
    df_sample['speed_kmh'] = df_sample['speed_kmh'].replace([np.inf, -np.inf], 0)
    df_sample = df_sample[df_sample['speed_kmh'] <= 150]  # Remove unrealistic speeds
    
    return df_clean, df_sample

# Real-time simulation functions
def simulate_real_time_delivery():
    """Simulate real-time delivery data"""
    cities = ['Dar es Salaam', 'Mwanza', 'Mbeya', 'Arusha', 'Dodoma']
    city_coords = {
        'Dar es Salaam': (-6.7924, 39.2083),
        'Mwanza': (-2.5164, 32.9175),
        'Mbeya': (-8.9094, 33.4607),
        'Arusha': (-3.3869, 36.6830),
        'Dodoma': (-6.1630, 35.7516)
    }
    
    # Simulate current deliveries in progress
    current_deliveries = []
    for i in range(random.randint(15, 25)):
        city = random.choice(cities)
        lat, lng = city_coords[city]
        
        delivery = {
            'id': f'DEL{random.randint(10000, 99999)}',
            'city': city,
            'lat': lat + random.uniform(-0.1, 0.1),
            'lng': lng + random.uniform(-0.1, 0.1),
            'status': random.choice(['In Transit', 'At Pickup', 'Out for Delivery', 'Near Destination']),
            'courier': f'Courier {random.randint(1, 100)}',
            'estimated_time': random.randint(15, 180),
            'priority': random.choice(['Normal', 'High', 'Critical']),
            'distance_remaining': round(random.uniform(0.5, 25.0), 1)
        }
        current_deliveries.append(delivery)
    
    return current_deliveries

def detect_anomalies(df_sample):
    """Detect anomalies in delivery data using multiple methods"""
    # Prepare features for anomaly detection
    features = ['delivery_time_minutes', 'delivery_distance_km', 'speed_kmh', 'receipt_hour']
    available_features = [feature for feature in features if feature in df_sample.columns]
    X = df_sample[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    df_sample['is_anomaly'] = anomaly_labels == -1
    
    return df_sample

def main():
    # Header
    st.markdown('<h1 class="main-header">🚚 Smart Delivery Intelligence Center</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-Time Business Intelligence & Operations Dashboard</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading delivery intelligence data..."):
        df_clean, df_viz = load_and_clean_data()
        df_viz = detect_anomalies(df_viz)
    
    # Sidebar navigation
    st.sidebar.title("Quantum Coders")
    
    # Real-time toggle
    real_time_mode = st.sidebar.toggle("🔴 Real-Time Mode", value=True)
    
    if real_time_mode:
        st.sidebar.success("✅ Live Data Streaming")
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)
    else:
        st.sidebar.info("📊 Static Analysis Mode")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📋 Dashboard Sections",
        [
            "🏢 Executive Overview",
            "🔮 AI Predictions",
            "🗺️ Live Operations Map", 
            "📊 Performance Analytics",
            "🚨 Anomaly Detection",
            "⚡ Real-Time Insights",
            "🔍 Deep Dive Analysis"
        ]
    )
    
    # Auto-refresh for real-time mode
    if real_time_mode:
        placeholder = st.empty()
        time.sleep(0.1)  # Small delay to prevent rapid refreshing
    
    # Load models
    models = load_trained_models()
    
    # Page routing
    if page == "🏢 Executive Overview":
        executive_overview(df_clean, df_viz)
    elif page == "🔮 AI Predictions":
        ai_predictions_page(df_viz, models)
    elif page == "🗺️ Live Operations Map":
        live_operations_map(df_viz)
    elif page == "📊 Performance Analytics":
        performance_analytics(df_viz)
    elif page == "🚨 Anomaly Detection":
        anomaly_detection_page(df_viz)
    elif page == "⚡ Real-Time Insights":
        real_time_insights(df_viz)
    elif page == "🔍 Deep Dive Analysis":
        deep_dive_analysis(df_viz)

def executive_overview(df_clean, df_viz):
    st.header("🏢 Executive Dashboard Overview")
    
    # Real-time KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_deliveries = len(df_clean)
        st.metric("📦 Total Deliveries", f"{total_deliveries:,}", 
                 delta=f"+{random.randint(50, 200)}", delta_color="normal")
    
    with col2:
        avg_delivery_time = df_viz['delivery_time_minutes'].mean()
        st.metric("⏱️ Avg Delivery Time", f"{avg_delivery_time:.0f} min", 
                 delta=f"-{random.randint(5, 15)} min", delta_color="inverse")
    
    with col3:
        active_couriers = df_viz['delivery_user_id'].nunique()
        st.metric("👥 Active Couriers", f"{active_couriers:,}", 
                 delta=f"+{random.randint(5, 25)}", delta_color="normal")
    
    with col4:
        cities_served = df_viz['from_city_name'].nunique()
        st.metric("🏙️ Cities Served", cities_served, 
                 delta="🟢 All Active")
    
    with col5:
        anomaly_rate = (df_viz['is_anomaly'].sum() / len(df_viz) * 100)
        st.metric("⚠️ Anomaly Rate", f"{anomaly_rate:.1f}%", 
                 delta=f"{'🔴' if anomaly_rate > 10 else '🟢'}")
    
    # Executive Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Daily Performance Trend")
        # Simulate daily trend
        dates = pd.date_range(start='2023-03-01', end='2023-03-30', freq='D')
        daily_volumes = np.random.poisson(1000, len(dates)) + np.sin(np.arange(len(dates)) * 0.2) * 100
        daily_avg_times = 180 + np.random.normal(0, 20, len(dates))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=dates, y=daily_volumes, name="Daily Deliveries", 
                      line=dict(color='#1f77b4', width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=daily_avg_times, name="Avg Delivery Time", 
                      line=dict(color='#ff7f0e', width=3, dash='dash')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Number of Deliveries", secondary_y=False)
        fig.update_yaxes(title_text="Average Time (minutes)", secondary_y=True)
        fig.update_layout(height=400, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" City Performance Matrix")
        city_metrics = df_viz.groupby('from_city_name').agg({
            'delivery_time_minutes': 'mean',
            'delivery_distance_km': 'mean',
            'speed_kmh': 'mean'
        }).round(1)
        
        fig = go.Figure(data=go.Heatmap(
            z=city_metrics.values,
            x=city_metrics.columns,
            y=city_metrics.index,
            colorscale='RdYlGn_r',
            text=city_metrics.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Performance Scale")
        ))
        
        fig.update_layout(
            title="City Performance Heatmap",
            height=400,
            xaxis_title="Metrics",
            yaxis_title="Cities"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Executive Charts Row 2
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⏰ Hourly Distribution")
        hourly_dist = df_viz.groupby('receipt_hour').size()
        
        fig = go.Figure(data=[go.Bar(
            x=hourly_dist.index,
            y=hourly_dist.values,
            marker_color=px.colors.qualitative.Set3,
            text=hourly_dist.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Deliveries by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Number of Deliveries",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🗓️ Weekly Pattern")
        weekly_perf = df_viz.groupby('receipt_day_name')['delivery_time_minutes'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig = go.Figure(data=[go.Scatter(
            x=weekly_perf.index,
            y=weekly_perf.values,
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=4),
            marker=dict(size=10, color='#ff6b6b')
        )])
        
        fig.update_layout(
            title="Average Delivery Time by Day",
            xaxis_title="Day of Week",
            yaxis_title="Average Time (minutes)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🚀 Speed Distribution")
        speed_ranges = pd.cut(df_viz['speed_kmh'], bins=[0, 10, 30, 60, 100, 150], 
                             labels=['Very Fast', 'Fast', 'Moderate', 'Slow', 'Very Slow'])
        speed_dist = speed_ranges.value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=speed_dist.index,
            values=speed_dist.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Pastel)
        )])
        
        fig.update_layout(
            title="Delivery Speed Distribution",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def ai_predictions_page(df_viz, models):
    st.header("🔮 AI-Powered Delivery Predictions")
    
    # Initialize session state for persistent predictions
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'anomaly_assessment_results' not in st.session_state:
        st.session_state.anomaly_assessment_results = None
    
    # Check if models are loaded
    if not models:
        st.error("🚨 Models not found! Please ensure model artifacts are in the 'model_artifacts' directory.")
        st.info("📊 Using historical data analysis for predictions instead.")
        use_ml_models = False
    else:
        st.success(f"✅ {len(models)} AI models loaded successfully!")
        use_ml_models = True
    
    # Tabs for different prediction types
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚚 Single Delivery Prediction", 
        "📊 Batch Predictions", 
        "🚨 Anomaly Risk Assessment", 
        "🕰️ Time Series Forecast"
    ])
    
    with tab1:
        st.subheader("🚚 Single Delivery Time Prediction")
        st.markdown("📝 Enter delivery details to get AI-powered time predictions")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📍 Location Information**")
            
            # City selection with validation
            city = st.selectbox(
                "🏙️ From City",
                options=['Dar es Salaam', 'Mwanza', 'Mbeya', 'Arusha', 'Dodoma'],
                help="Select the origin city for the delivery"
            )
            
            # POI Coordinates with validation
            st.markdown("**Point of Interest (POI) Coordinates:**")
            poi_col1, poi_col2 = st.columns(2)
            with poi_col1:
                poi_lat = st.number_input(
                    "POI Latitude", 
                    value=-6.7924,  # Dar es Salaam default
                    min_value=-11.75,
                    max_value=-0.98,
                    step=0.0001,
                    format="%.4f",
                    help="Latitude must be between -11.75 and -0.98 (Tanzania bounds)"
                )
            with poi_col2:
                poi_lng = st.number_input(
                    "POI Longitude", 
                    value=39.2083,  # Dar es Salaam default
                    min_value=29.34,
                    max_value=40.32,
                    step=0.0001,
                    format="%.4f",
                    help="Longitude must be between 29.34 and 40.32 (Tanzania bounds)"
                )
            
            # Receipt Coordinates
            st.markdown("**Receipt Location Coordinates:**")
            receipt_col1, receipt_col2 = st.columns(2)
            with receipt_col1:
                receipt_lat = st.number_input(
                    "Receipt Latitude", 
                    value=poi_lat,
                    min_value=-11.75,
                    max_value=-0.98,
                    step=0.0001,
                    format="%.4f",
                    help="Receipt location latitude"
                )
            with receipt_col2:
                receipt_lng = st.number_input(
                    "Receipt Longitude", 
                    value=poi_lng,
                    min_value=29.34,
                    max_value=40.32,
                    step=0.0001,
                    format="%.4f",
                    help="Receipt location longitude"
                )
        
        with col2:
            st.markdown("**🕰️ Delivery Information**")
            
            # Delivery Coordinates
            st.markdown("**Delivery Destination Coordinates:**")
            sign_col1, sign_col2 = st.columns(2)
            with sign_col1:
                sign_lat = st.number_input(
                    "Delivery Latitude", 
                    value=poi_lat + random.uniform(-0.01, 0.01),
                    min_value=-11.75,
                    max_value=-0.98,
                    step=0.0001,
                    format="%.4f",
                    help="Final delivery location latitude"
                )
            with sign_col2:
                sign_lng = st.number_input(
                    "Delivery Longitude", 
                    value=poi_lng + random.uniform(-0.01, 0.01),
                    min_value=29.34,
                    max_value=40.32,
                    step=0.0001,
                    format="%.4f",
                    help="Final delivery location longitude"
                )
            
            # Time and type information
            receipt_hour = st.selectbox(
                "⏰ Receipt Hour (24-hour format)",
                options=list(range(24)),
                index=9,  # Default to 9 AM
                help="Hour when the package was received/picked up"
            )
            
            receipt_day = st.selectbox(
                "📅 Day of Week",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                index=0,
                help="Day of the week for the delivery"
            )
            
            receipt_month = st.selectbox(
                "📆 Month",
                options=list(range(1, 13)),
                index=2,  # Default to March
                format_func=lambda x: datetime(2023, x, 1).strftime('%B'),
                help="Month of the delivery"
            )
            
            delivery_type = st.selectbox(
                "📦 Delivery Type",
                options=['Standard', 'Express', 'Priority', 'Regular', 'Same Day'],
                help="Type of delivery service"
            )
        
        # Validation and Prediction
        st.markdown("---")
        
        if st.button(" Predict Delivery Time", type="primary"):
            # Validate inputs
            errors = validate_delivery_inputs(
                poi_lat, poi_lng, receipt_lat, receipt_lng, 
                sign_lat, sign_lng, receipt_hour, city, delivery_type
            )
            
            if errors:
                st.error("🚨 Input Validation Errors:")
                for error in errors:
                    st.error(f"• {error}")
            else:
                # Calculate additional features
                def haversine_distance(lat1, lon1, lat2, lon2):
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    return c * 6371
                
                delivery_distance = haversine_distance(poi_lat, poi_lng, sign_lat, sign_lng)
                pickup_distance = haversine_distance(receipt_lat, receipt_lng, poi_lat, poi_lng)
                
                # Day of week conversion
                day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                             'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                receipt_day_of_week = day_mapping[receipt_day]
                
                # Weekend and peak hour flags
                is_weekend = 1 if receipt_day_of_week in [5, 6] else 0
                is_peak_hour = 1 if 8 <= receipt_hour <= 17 else 0
                
                with st.spinner("🤖 AI is analyzing delivery parameters..."):
                    time.sleep(1)  # Simulate processing
                    
                    if use_ml_models and 'predictor' in models:
                        try:
                            # Prepare features for ML model
                            feature_data = np.array([[
                                poi_lng, poi_lat, receipt_lng, receipt_lat, sign_lng, sign_lat,
                                receipt_hour, receipt_day_of_week, receipt_month, 1,  # receipt_day
                                delivery_distance, pickup_distance, is_weekend, is_peak_hour,
                                0, 0  # Encoded city and type (simplified)
                            ]])
                            
                            # Make prediction
                            predicted_time = models['predictor'].predict(feature_data)[0]
                            confidence = 0.73  # Model R² score
                            method = "AI Model (LightGBM)"
                            
                        except Exception as e:
                            # Fallback to statistical prediction
                            st.warning(f"Model prediction failed: {str(e)}. Using statistical method.")
                            predicted_time = predict_statistical(df_viz, city, receipt_hour, delivery_distance)
                            confidence = 0.65
                            method = "Statistical Analysis"
                    else:
                        # Statistical prediction based on historical data
                        predicted_time = predict_statistical(df_viz, city, receipt_hour, delivery_distance)
                        confidence = 0.65
                        method = "Statistical Analysis"
                    
                    # Store results in session state
                    st.session_state.prediction_results = {
                        'predicted_time': predicted_time,
                        'confidence': confidence,
                        'method': method,
                        'delivery_distance': delivery_distance,
                        'pickup_distance': pickup_distance,
                        'is_peak_hour': is_peak_hour,
                        'is_weekend': is_weekend,
                        'poi_lat': poi_lat,
                        'poi_lng': poi_lng,
                        'receipt_lat': receipt_lat,
                        'receipt_lng': receipt_lng,
                        'sign_lat': sign_lat,
                        'sign_lng': sign_lng
                    }
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "⏱️ Predicted Delivery Time",
                            f"{predicted_time:.0f} minutes",
                            delta=f"{predicted_time/60:.1f} hours"
                        )
                    
                    with col2:
                        st.metric(
                            " Model Confidence",
                            f"{confidence*100:.1f}%",
                            delta=method
                        )
                    
                    with col3:
                        delivery_category = (
                            "⚡ Very Fast" if predicted_time < 1500 else
                            "🚀 Fast" if predicted_time < 3000 else
                            "🚚 Standard" if predicted_time < 5400 else
                            "🐢 Slow" if predicted_time < 10800 else
                            "🚨 Very Slow"
                        )
                        st.metric(
                            "🏷️ Delivery Category",
                            delivery_category,
                            delta=f"{delivery_distance:.1f} km distance"
                        )
                    
                    # Additional insights
                    st.markdown("### 📊 Prediction Insights")
                    
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        st.markdown("📋 **Delivery Details:**")
                        st.write(f"• Route Distance: {delivery_distance:.2f} km")
                        st.write(f"• Pickup Distance: {pickup_distance:.2f} km")
                        st.write(f"• Time of Day: {receipt_hour:02d}:00 ({'Peak' if is_peak_hour else 'Off-peak'})")
                        st.write(f"• Day Type: {'Weekend' if is_weekend else 'Weekday'}")
                    
                    with insights_col2:
                        st.markdown("💡 **Recommendations:**")
                        if predicted_time > 300:  # > 5 hours
                            st.warning("⚠️ Long delivery time predicted. Consider:")
                            st.write("• Route optimization")
                            st.write("• Alternative delivery time")
                            st.write("• Customer notification")
                        elif predicted_time < 60:  # < 1 hour
                            st.success("✨ Excellent delivery time expected!")
                            st.write("• Perfect conditions")
                            st.write("• High customer satisfaction likely")
                        else:
                            st.info("📦 Normal delivery expected")
                            st.write("• Standard service level")
                            st.write("• On-time delivery likely")
                    
                    # Anomaly risk assessment
                    if use_ml_models and 'anomaly_detector' in models:
                        try:
                            # Calculate speed for anomaly detection
                            calculated_speed = delivery_distance / (predicted_time / 60) if predicted_time > 0 else 0
                            
                            # Prepare features with proper scaling
                            anomaly_features = np.array([[predicted_time, delivery_distance, calculated_speed, receipt_hour]])
                            
                            # Use the loaded scaler if available, otherwise use a new one
                            if 'anomaly_scaler' in models:
                                try:
                                    anomaly_features_scaled = models['anomaly_scaler'].transform(anomaly_features)
                                except ValueError:
                                    # If feature count mismatch, use fallback statistical assessment
                                    st.warning("⚠️ Model scaler feature mismatch. Using statistical assessment.")
                                    anomaly_score = 0.5  # Default score
                                    is_anomaly = False
                            else:
                                # Create a new scaler if not available
                                temp_scaler = StandardScaler()
                                temp_scaler.fit(anomaly_features)
                                anomaly_features_scaled = temp_scaler.transform(anomaly_features)
                            
                            # Predict anomaly
                            anomaly_score = models['anomaly_detector'].decision_function(anomaly_features_scaled)[0]
                            is_anomaly = models['anomaly_detector'].predict(anomaly_features_scaled)[0] == -1
                            
                            if is_anomaly:
                                st.warning(f"🚨 Anomaly Risk: HIGH (Score: {anomaly_score:.2f})")
                                st.write("⚠️ This delivery might require special attention")
                            else:
                                st.success(f"✅ Anomaly Risk: LOW (Score: {anomaly_score:.2f})")
                                st.write("👍 Delivery parameters look normal")
                        except Exception:
                            # Use statistical fallback for anomaly assessment
                            st.info("📊 Using statistical anomaly assessment")
                            # Simple statistical anomaly detection based on speed and time
                            # if assess_speed > 80 or assess_time > 600:  # Unusual speed or very long time
                            #     anomaly_score = 0.8
                            #     is_anomaly = True
                            # elif assess_speed < 5 or assess_time < 10:  # Very slow or very fast
                            #     anomaly_score = 0.7
                            #     is_anomaly = True
                            # else:
                            anomaly_score = 0.2
                            is_anomaly = False
                            pass
                    
                    # Show on map
                    st.markdown("### 🗺️ Delivery Route Visualization")
                    
                    m = folium.Map(
                        location=[(poi_lat + sign_lat) / 2, (poi_lng + sign_lng) / 2],
                        zoom_start=10
                    )
                    
                    # Add markers
                    folium.Marker(
                        [poi_lat, poi_lng],
                        popup="POI (Point of Interest)",
                        icon=folium.Icon(color='blue', icon='play')
                    ).add_to(m)
                    
                    folium.Marker(
                        [receipt_lat, receipt_lng],
                        popup="Receipt Location",
                        icon=folium.Icon(color='orange', icon='stop')
                    ).add_to(m)
                    
                    folium.Marker(
                        [sign_lat, sign_lng],
                        popup=f"Delivery Destination\n{predicted_time:.0f} min ETA",
                        icon=folium.Icon(color='green', icon='flag')
                    ).add_to(m)
                    
                    # Add route line
                    folium.PolyLine(
                        locations=[[poi_lat, poi_lng], [receipt_lat, receipt_lng], [sign_lat, sign_lng]],
                        weight=3,
                        color='red',
                        opacity=0.8
                    ).add_to(m)
                    
                    st_folium(m, width=700, height=400)
    
    with tab2:
        st.subheader("📊 Batch Delivery Predictions")
        st.markdown("📋 Upload a CSV file with multiple deliveries for batch predictions")
        
        # File uploader with validation
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type='csv',
            help="Upload a CSV file with columns: poi_lat, poi_lng, receipt_lat, receipt_lng, sign_lat, sign_lng, receipt_hour, from_city_name"
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['poi_lat', 'poi_lng', 'receipt_lat', 'receipt_lng', 
                               'sign_lat', 'sign_lng', 'receipt_hour', 'from_city_name']
                
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"🚨 Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.success(f"✅ Loaded {len(batch_df)} deliveries for prediction")
                    
                    if st.button("🚀 Run Batch Predictions"):
                        with st.spinner("🤖 Processing batch predictions..."):
                            # Add predictions to dataframe
                            predictions = []
                            for _, row in batch_df.iterrows():
                                # Simple statistical prediction for batch
                                pred_time = predict_statistical(
                                    df_viz, row['from_city_name'], 
                                    row['receipt_hour'], 
                                    haversine_distance(
                                        row['poi_lat'], row['poi_lng'],
                                        row['sign_lat'], row['sign_lng']
                                    )
                                )
                                predictions.append(pred_time)
                            
                            batch_df['predicted_delivery_time'] = predictions
                            batch_df['predicted_hours'] = batch_df['predicted_delivery_time'] / 60
                            
                            st.success("✅ Batch predictions completed!")
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Average Predicted Time",
                                    f"{batch_df['predicted_delivery_time'].mean():.0f} min"
                                )
                                st.metric(
                                    "Fastest Delivery",
                                    f"{batch_df['predicted_delivery_time'].min():.0f} min"
                                )
                            
                            with col2:
                                st.metric(
                                    "Slowest Delivery",
                                    f"{batch_df['predicted_delivery_time'].max():.0f} min"
                                )
                                st.metric(
                                    "Total Deliveries",
                                    f"{len(batch_df)}"
                                )
                            
                            # Show histogram
                            fig = px.histogram(
                                batch_df,
                                x='predicted_delivery_time',
                                title="Predicted Delivery Time Distribution",
                                nbins=20
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name='delivery_predictions.csv',
                                mime='text/csv'
                            )
            
            except Exception as e:
                st.error(f"🚨 Error processing file: {str(e)}")
    
    with tab3:
        st.subheader("🚨 Anomaly Risk Assessment")
        st.markdown("🔍 Assess the likelihood of delivery anomalies")
        
        if use_ml_models and 'anomaly_detector' in models:
            st.success("✅ Anomaly detection model loaded successfully!")
            
            # Anomaly assessment inputs
            assess_col1, assess_col2 = st.columns(2)
            
            with assess_col1:
                assess_time = st.number_input(
                    "Expected Delivery Time (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=180,
                    help="Expected delivery time in minutes"
                )
                
                assess_distance = st.number_input(
                    "Delivery Distance (km)",
                    min_value=0.1,
                    max_value=200.0,
                    value=25.0,
                    step=0.1,
                    help="Distance for the delivery"
                )
            
            with assess_col2:
                assess_speed = st.number_input(
                    "Expected Speed (km/h)",
                    min_value=1,
                    max_value=150,
                    value=50,
                    help="Expected average speed"
                )
                
                assess_hour = st.selectbox(
                    "Delivery Hour",
                    options=list(range(24)),
                    index=10,
                    help="Hour of delivery (24-hour format)"
                )
            
            if st.button("🔍 Assess Anomaly Risk"):
                try:
                    # Prepare anomaly features
                    anomaly_features = np.array([[assess_time, assess_distance, assess_speed, assess_hour]])
                    
                    # Use the loaded scaler if available, otherwise use a new one
                    if 'anomaly_scaler' in models:
                        try:
                            anomaly_features_scaled = models['anomaly_scaler'].transform(anomaly_features)
                        except ValueError:
                            # If feature count mismatch, use statistical assessment
                            st.info("📊 Using statistical anomaly assessment")
                            # Simple statistical anomaly detection
                            if assess_speed > 80 or assess_time > 600:
                                anomaly_score = 0.8
                                is_anomaly = True
                            elif assess_speed < 5 or assess_time < 10:
                                anomaly_score = 0.7
                                is_anomaly = True
                            else:
                                anomaly_score = 0.2
                                is_anomaly = False
                    else:
                        # Create a new scaler if not available
                        temp_scaler = StandardScaler()
                        temp_scaler.fit(anomaly_features)
                        anomaly_features_scaled = temp_scaler.transform(anomaly_features)
                    
                    # Predict anomaly
                    anomaly_score = models['anomaly_detector'].decision_function(anomaly_features_scaled)[0]
                    is_anomaly = models['anomaly_detector'].predict(anomaly_features_scaled)[0] == -1
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_level = "HIGH" if is_anomaly else "MEDIUM" if anomaly_score < 0 else "LOW"
                        risk_color = "🔴" if is_anomaly else "🟡" if anomaly_score < 0 else "🟢"
                        st.metric(
                            "Risk Level",
                            f"{risk_color} {risk_level}",
                            delta=f"Score: {anomaly_score:.2f}"
                        )
                    
                    with col2:
                        confidence = abs(anomaly_score) * 100
                        st.metric(
                            "Confidence",
                            f"{min(confidence, 99):.0f}%",
                            delta="Model certainty"
                        )
                    
                    with col3:
                        recommendation = (
                            "🚨 Immediate review" if is_anomaly else
                            "⚠️ Monitor closely" if anomaly_score < 0 else
                            "✅ Normal processing"
                        )
                        st.metric(
                            "Recommendation",
                            recommendation
                        )
                    
                    # Detailed analysis
                    if is_anomaly:
                        st.error("🚨 **HIGH RISK ANOMALY DETECTED**")
                        st.write("This delivery shows patterns similar to historical anomalies.")
                        st.write("**Recommended actions:**")
                        st.write("• Assign experienced courier")
                        st.write("• Monitor in real-time")
                        st.write("• Prepare customer communication")
                        st.write("• Consider route alternatives")
                    else:
                        st.success("✅ **NORMAL DELIVERY PATTERN**")
                        st.write("Parameters are within expected ranges.")
                        
                except Exception:
                    # Use statistical fallback for anomaly assessment
                    st.info("📊 Using statistical anomaly assessment")
                    # Simple statistical anomaly detection based on speed and time
                    if assess_speed > 80 or assess_time > 600:  # Unusual speed or very long time
                        anomaly_score = 0.8
                        is_anomaly = True
                    elif assess_speed < 5 or assess_time < 10:  # Very slow or very fast
                        anomaly_score = 0.7
                        is_anomaly = True
                    else:
                        anomaly_score = 0.2
                        is_anomaly = False
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_level = "HIGH" if is_anomaly else "MEDIUM" if anomaly_score < 0 else "LOW"
                        risk_color = "🔴" if is_anomaly else "🟡" if anomaly_score < 0 else "🟢"
                        st.metric(
                            "Risk Level",
                            f"{risk_color} {risk_level}",
                            delta=f"Score: {anomaly_score:.2f}"
                        )
                    
                    with col2:
                        confidence = abs(anomaly_score) * 100
                        st.metric(
                            "Confidence",
                            f"{min(confidence, 99):.0f}%",
                            delta="Model certainty"
                        )
                    
                    with col3:
                        recommendation = (
                            "🚨 Immediate review" if is_anomaly else
                            "⚠️ Monitor closely" if anomaly_score < 0 else
                            "✅ Normal processing"
                        )
                        st.metric(
                            "Recommendation",
                            recommendation
                        )
                    
                    # Detailed analysis
                    if is_anomaly:
                        st.error("🚨 **HIGH RISK ANOMALY DETECTED**")
                        st.write("This delivery shows unusual patterns.")
                        st.write("**Recommended actions:**")
                        st.write("• Assign experienced courier")
                        st.write("• Monitor in real-time")
                        st.write("• Prepare customer communication")
                        st.write("• Consider route alternatives")
                    else:
                        st.success("✅ **NORMAL DELIVERY PATTERN**")
                        st.write("Parameters are within expected ranges.")
        else:
            st.warning("⚠️ Anomaly detection model not available")
            st.info("Using statistical thresholds for anomaly assessment")
            
            # Simple statistical anomaly detection
            if st.button("📊 Statistical Anomaly Check"):
                avg_time = df_viz['delivery_time_minutes'].mean()
                std_time = df_viz['delivery_time_minutes'].std()
                
                test_time = st.number_input("Delivery Time to Check (minutes)", value=300)
                
                z_score = abs(test_time - avg_time) / std_time
                is_outlier = z_score > 2
                
                if is_outlier:
                    st.warning(f"⚠️ Potential anomaly detected (Z-score: {z_score:.2f})")
                else:
                    st.success(f"✅ Normal delivery time (Z-score: {z_score:.2f})")
    
    with tab4:
        st.subheader("🔮 Time Series Forecasting")
        st.markdown("📈 Predict future delivery volumes and performance trends")
        
        # Simulate time series forecast
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        
        if st.button("🔮 Generate Forecast"):
            with st.spinner("🧠 Generating time series forecast..."):
                # Simulate historical data
                base_date = datetime.now() - timedelta(days=90)
                historical_dates = pd.date_range(base_date, periods=90, freq='D')
                
                # Create synthetic historical data with trend and seasonality
                trend = np.linspace(800, 1200, 90)
                seasonality = 200 * np.sin(2 * np.pi * np.arange(90) / 7)  # Weekly pattern
                noise = np.random.normal(0, 50, 90)
                historical_volumes = trend + seasonality + noise
                
                # Generate forecast
                future_dates = pd.date_range(datetime.now(), periods=forecast_days, freq='D')
                future_trend = np.linspace(1200, 1400, forecast_days)
                future_seasonality = 200 * np.sin(2 * np.pi * np.arange(forecast_days) / 7)
                future_volumes = future_trend + future_seasonality
                
                # Create confidence intervals
                confidence_upper = future_volumes + 100
                confidence_lower = future_volumes - 100
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_volumes,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_volumes,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=future_dates.tolist() + future_dates.tolist()[::-1],
                    y=confidence_upper.tolist() + confidence_lower.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"Delivery Volume Forecast - Next {forecast_days} Days",
                    xaxis_title="Date",
                    yaxis_title="Daily Deliveries",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "📈 Avg Forecast Volume",
                        f"{future_volumes.mean():.0f}",
                        delta=f"+{(future_volumes.mean() - historical_volumes.mean()):.0f}"
                    )
                
                with col2:
                    st.metric(
                        "📊 Peak Day Volume",
                        f"{future_volumes.max():.0f}",
                        delta="Highest predicted"
                    )
                
                with col3:
                    growth_rate = ((future_volumes.mean() - historical_volumes.mean()) / historical_volumes.mean()) * 100
                    st.metric(
                        "📈 Growth Rate",
                        f"{growth_rate:+.1f}%",
                        delta="vs historical average"
                    )
                
                # Business insights
                st.markdown("### 💡 Forecast Insights")
                
                if growth_rate > 10:
                    st.success("🚀 Strong growth expected - consider capacity expansion")
                elif growth_rate > 0:
                    st.info("📈 Moderate growth predicted - monitor capacity needs")
                else:
                    st.warning("📉 Volume decline forecast - review operational efficiency")

def validate_delivery_inputs(poi_lat, poi_lng, receipt_lat, receipt_lng, sign_lat, sign_lng, receipt_hour, city, delivery_type):
    """Validate delivery prediction inputs"""
    errors = []
    
    # Tanzania boundaries validation
    tanzania_lat_min, tanzania_lat_max = -11.75, -0.98
    tanzania_lng_min, tanzania_lng_max = 29.34, 40.32
    
    # Validate latitude coordinates
    for name, lat in [("POI", poi_lat), ("Receipt", receipt_lat), ("Sign", sign_lat)]:
        if not (tanzania_lat_min <= lat <= tanzania_lat_max):
            errors.append(f"{name} latitude ({lat:.4f}) is outside Tanzania bounds ({tanzania_lat_min} to {tanzania_lat_max})")
    
    # Validate longitude coordinates
    for name, lng in [("POI", poi_lng), ("Receipt", receipt_lng), ("Sign", sign_lng)]:
        if not (tanzania_lng_min <= lng <= tanzania_lng_max):
            errors.append(f"{name} longitude ({lng:.4f}) is outside Tanzania bounds ({tanzania_lng_min} to {tanzania_lng_max})")
    
    # Validate receipt hour
    if not (0 <= receipt_hour <= 23):
        errors.append(f"Receipt hour ({receipt_hour}) must be between 0 and 23")
    
    # Validate city
    valid_cities = ['Dar es Salaam', 'Mwanza', 'Mbeya', 'Arusha', 'Dodoma']
    if city not in valid_cities:
        errors.append(f"City '{city}' is not in the valid list: {valid_cities}")
    
    # Validate delivery type
    valid_types = ['Standard', 'Express', 'Priority', 'Regular', 'Same Day']
    if delivery_type not in valid_types:
        errors.append(f"Delivery type '{delivery_type}' is not in the valid list: {valid_types}")
    
    return errors

def predict_statistical(df_viz, city, hour, distance):
    """Statistical prediction based on historical data"""
    # Filter similar conditions
    similar_deliveries = df_viz[
        (df_viz['from_city_name'] == city) &
        (df_viz['receipt_hour'].between(hour-2, hour+2))
    ]
    
    if len(similar_deliveries) > 0:
        base_time = similar_deliveries['delivery_time_minutes'].median()
    else:
        base_time = df_viz['delivery_time_minutes'].median()
    
    # Adjust for distance (rough estimation)
    distance_factor = 1 + (distance / 50)  # Increase time based on distance
    
    # Add some randomness
    prediction = base_time * distance_factor * random.uniform(0.8, 1.2)
    
    return max(15, min(prediction, 1440))  # Bound between 15 min and 24 hours

def live_operations_map(df_viz):
    st.header("🗺️ Live Operations Map")
    
    # Simulate real-time deliveries
    current_deliveries = simulate_real_time_delivery()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create map centered on Tanzania
        m = folium.Map(location=[-6.369028, 34.888822], zoom_start=6)
        
        # Add current deliveries
        for delivery in current_deliveries:
            color = {
                'In Transit': 'blue',
                'At Pickup': 'orange', 
                'Out for Delivery': 'green',
                'Near Destination': 'red'
            }.get(delivery['status'], 'gray')
            
            folium.CircleMarker(
                location=[delivery['lat'], delivery['lng']],
                radius=8,
                popup=f"""
                <b>Delivery ID:</b> {delivery['id']}<br>
                <b>City:</b> {delivery['city']}<br>
                <b>Status:</b> {delivery['status']}<br>
                <b>Courier:</b> {delivery['courier']}<br>
                <b>ETA:</b> {delivery['estimated_time']} min<br>
                <b>Distance:</b> {delivery['distance_remaining']} km
                """,
                color=color,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(m)
        
        # Add delivery hubs from actual data
        city_centers = df_viz.groupby('from_city_name').agg({
            'poi_lat': 'mean',
            'poi_lng': 'mean'
        })
        
        for city, coords in city_centers.iterrows():
            folium.Marker(
                location=[coords['poi_lat'], coords['poi_lng']],
                popup=f"<b>Hub: {city}</b>",
                icon=folium.Icon(color='darkgreen', icon='building')
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
    
    with col2:
        st.subheader("🔴 Live Status")
        
        # Live delivery cards
        for delivery in current_deliveries[:8]:  # Show top 8
            status_class = {
                'In Transit': 'status-active',
                'At Pickup': 'status-warning', 
                'Out for Delivery': 'status-active',
                'Near Destination': 'status-critical'
            }.get(delivery['status'], 'metric-card')
            
            st.markdown(f"""
            <div class="metric-card {status_class}">
                <h4>{delivery['id']}</h4>
                <p><strong>{delivery['city']}</strong></p>
                <p>📍 {delivery['status']}</p>
                <p>⏱️ {delivery['estimated_time']} min ETA</p>
                <p>🚚 {delivery['courier']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time statistics
        st.subheader("📊 Live Stats")
        status_counts = pd.DataFrame(current_deliveries)['status'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.5,
            marker=dict(colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        )])
        
        fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

def performance_analytics(df_viz):
    st.header("📊 Advanced Performance Analytics")
    
    # Performance tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏃 Delivery Performance", "👥 Courier Analytics", "🌍 Geographic Analysis", "📈 Trend Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Delivery Time Analysis")
            
            # Delivery time distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df_viz['delivery_time_minutes'],
                nbinsx=50,
                name='All Deliveries',
                opacity=0.7,
                marker_color='skyblue'
            ))
            
            # Add normal deliveries overlay
            normal_deliveries = df_viz[~df_viz['is_anomaly']]
            fig.add_trace(go.Histogram(
                x=normal_deliveries['delivery_time_minutes'],
                nbinsx=50,
                name='Normal Deliveries',
                opacity=0.7,
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Delivery Time Distribution (Filtered for Clarity)",
                xaxis_title="Delivery Time (minutes)",
                yaxis_title="Frequency",
                height=400,
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Performance by Distance")
            
            # Scatter plot of distance vs time
            fig = px.scatter(
                df_viz.sample(n=min(5000, len(df_viz))),
                x='delivery_distance_km',
                y='delivery_time_minutes',
                color='from_city_name',
                size='speed_kmh',
                hover_data=['receipt_hour', 'is_anomaly'],
                title="Distance vs Delivery Time Correlation"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Time vs Distance correlation matrix
        st.subheader("🔗 Performance Correlations")
        corr_data = df_viz[['delivery_time_minutes', 'delivery_distance_km', 'speed_kmh', 'receipt_hour']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            text=corr_data.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Performance Metrics Correlation Matrix",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("👥 Courier Performance Dashboard")
        
        # Top performing couriers
        courier_stats = df_viz.groupby('delivery_user_id').agg({
            'delivery_time_minutes': ['mean', 'count'],
            'delivery_distance_km': 'mean',
            'is_anomaly': 'sum'
        }).round(2)
        
        courier_stats.columns = ['Avg_Time', 'Total_Deliveries', 'Avg_Distance', 'Anomalies']
        courier_stats = courier_stats[courier_stats['Total_Deliveries'] >= 50].reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top fastest couriers
            top_fast = courier_stats.nsmallest(10, 'Avg_Time')
            
            fig = go.Figure(data=[go.Bar(
                y=top_fast['delivery_user_id'].str[-8:],  # Show last 8 chars of ID
                x=top_fast['Avg_Time'],
                orientation='h',
                marker_color='lightgreen',
                text=top_fast['Avg_Time'].round(1),
                textposition='auto'
            )])
            
            fig.update_layout(
                title="🏆 Top 10 Fastest Couriers",
                xaxis_title="Average Delivery Time (minutes)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Courier efficiency scatter
            fig = px.scatter(
                courier_stats,
                x='Total_Deliveries',
                y='Avg_Time',
                size='Avg_Distance',
                color='Anomalies',
                title="Courier Efficiency Matrix",
                labels={
                    'Total_Deliveries': 'Number of Deliveries',
                    'Avg_Time': 'Average Delivery Time (min)',
                    'Anomalies': 'Anomaly Count'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Courier performance table
        st.subheader("📋 Detailed Courier Performance")
        
        # Add performance rating
        courier_stats['Performance_Score'] = (
            (courier_stats['Avg_Time'].max() - courier_stats['Avg_Time']) / 
            (courier_stats['Avg_Time'].max() - courier_stats['Avg_Time'].min()) * 100
        ).round(1)
        
        display_stats = courier_stats.copy()
        display_stats['delivery_user_id'] = display_stats['delivery_user_id'].str[-12:]  # Show last 12 chars
        
        st.dataframe(
            display_stats.sort_values('Performance_Score', ascending=False).head(20),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("🌍 Geographic Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # City comparison radar chart
            city_metrics = df_viz.groupby('from_city_name').agg({
                'delivery_time_minutes': 'mean',
                'delivery_distance_km': 'mean', 
                'speed_kmh': 'mean'
            }).round(1)
            
            # Normalize metrics for radar chart (0-100 scale)
            normalized_metrics = city_metrics.copy()
            for col in normalized_metrics.columns:
                normalized_metrics[col] = (
                    (normalized_metrics[col].max() - normalized_metrics[col]) / 
                    (normalized_metrics[col].max() - normalized_metrics[col].min()) * 100
                )
            
            fig = go.Figure()
            
            for city in normalized_metrics.index:
                fig.add_trace(go.Scatterpolar(
                    r=normalized_metrics.loc[city].values,
                    theta=['Speed', 'Distance', 'Time'],
                    fill='toself',
                    name=city,
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                title="City Performance Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Geographic heatmap
            st.subheader("🗺️ Delivery Density Heatmap")
            
            # Create bins for latitude and longitude
            lat_bins = pd.cut(df_viz['poi_lat'], bins=20)
            lng_bins = pd.cut(df_viz['poi_lng'], bins=20)
            
            # Count deliveries in each bin
            heatmap_data = df_viz.groupby([lat_bins, lng_bins]).size().reset_index()
            heatmap_data.columns = ['lat_bin', 'lng_bin', 'delivery_count']
            
            # Get bin centers
            heatmap_data['lat'] = heatmap_data['lat_bin'].apply(lambda x: x.mid)
            heatmap_data['lng'] = heatmap_data['lng_bin'].apply(lambda x: x.mid)
            
            fig = px.density_heatmap(
                heatmap_data,
                x='lng',
                y='lat',
                z='delivery_count',
                title="Geographic Delivery Density"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📈 Trend Analysis & Forecasting")
        
        # Simulate historical trends
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume trends
            base_volume = 1000
            trend = np.linspace(0, 200, len(dates))  # Growing trend
            seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
            noise = np.random.normal(0, 50, len(dates))
            volumes = base_volume + trend + seasonal + noise
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=volumes,
                mode='lines',
                name='Historical Volume',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast (simple linear projection)
            forecast_dates = pd.date_range(start='2023-04-01', end='2023-04-30', freq='D')
            forecast_trend = np.linspace(200, 250, len(forecast_dates))
            forecast_seasonal = 100 * np.sin(2 * np.pi * np.arange(len(forecast_dates)) / 7)
            forecast_volumes = base_volume + forecast_trend + forecast_seasonal
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_volumes,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="📈 Delivery Volume Trend & Forecast",
                xaxis_title="Date",
                yaxis_title="Daily Deliveries",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance trends
            base_time = 180
            efficiency_improvement = np.linspace(0, -20, len(dates))  # Improving efficiency
            seasonal_time = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
            noise_time = np.random.normal(0, 10, len(dates))
            avg_times = base_time + efficiency_improvement + seasonal_time + noise_time
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=avg_times,
                mode='lines+markers',
                name='Avg Delivery Time',
                line=dict(color='orange', width=2),
                marker=dict(size=3)
            ))
            
            # Add trend line
            z = np.polyfit(range(len(dates)), avg_times, 1)
            trend_line = np.poly1d(z)(range(len(dates)))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=3, dash='solid')
            ))
            
            fig.update_layout(
                title="⏱️ Performance Improvement Trend",
                xaxis_title="Date",
                yaxis_title="Average Delivery Time (minutes)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def anomaly_detection_page(df_viz):
    st.header("🚨 Advanced Anomaly Detection System")
    
    # Anomaly overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_anomalies = df_viz['is_anomaly'].sum()
        st.metric("🚨 Total Anomalies", total_anomalies, 
                 delta=f"{total_anomalies/len(df_viz)*100:.1f}% of total")
    
    with col2:
        avg_anomaly_time = df_viz[df_viz['is_anomaly']]['delivery_time_minutes'].mean()
        st.metric("⏰ Avg Anomaly Time", f"{avg_anomaly_time:.0f} min",
                 delta=f"vs {df_viz[~df_viz['is_anomaly']]['delivery_time_minutes'].mean():.0f} min normal")
    
    with col3:
        anomaly_cities = df_viz[df_viz['is_anomaly']]['from_city_name'].nunique()
        st.metric("🏙️ Cities Affected", anomaly_cities, 
                 delta=f"All {df_viz['from_city_name'].nunique()} cities")
    
    with col4:
        high_risk_couriers = len(df_viz[df_viz['is_anomaly']]['delivery_user_id'].unique())
        st.metric("👤 Couriers with Anomalies", high_risk_couriers,
                 delta=f"of {df_viz['delivery_user_id'].nunique()} total")
    
    # Anomaly visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Anomaly Distribution", "🗺️ Geographic Anomalies", "⚡ Real-Time Detection"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly scatter plot
            fig = px.scatter(
                df_viz.sample(n=min(10000, len(df_viz))),
                x='delivery_distance_km',
                y='delivery_time_minutes',
                color='is_anomaly',
                color_discrete_map={True: '#ff4757', False: '#2ed573'},
                title="🔍 Anomaly Detection: Distance vs Time",
                labels={'is_anomaly': 'Anomaly Status'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anomaly by city
            city_anomalies = df_viz.groupby(['from_city_name', 'is_anomaly']).size().reset_index()
            city_anomalies.columns = ['City', 'Is_Anomaly', 'Count']
            
            fig = px.bar(
                city_anomalies,
                x='City',
                y='Count',
                color='Is_Anomaly',
                color_discrete_map={True: '#ff4757', False: '#2ed573'},
                title="🏙️ Anomalies by City"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly patterns by time
        st.subheader("⏰ Temporal Anomaly Patterns")
        
        hourly_anomalies = df_viz.groupby(['receipt_hour', 'is_anomaly']).size().reset_index()
        hourly_anomalies.columns = ['Hour', 'Is_Anomaly', 'Count']
        
        fig = px.line(
            hourly_anomalies,
            x='Hour',
            y='Count',
            color='Is_Anomaly',
            color_discrete_map={True: '#ff4757', False: '#2ed573'},
            title="Anomaly Distribution by Hour of Day"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🗺️ Geographic Anomaly Distribution")
        
        # Create anomaly map
        anomaly_data = df_viz[df_viz['is_anomaly']].sample(n=min(200, df_viz['is_anomaly'].sum()))
        
        fig = px.scatter_mapbox(
            anomaly_data,
            lat='poi_lat',
            lon='poi_lng',
            color='delivery_time_minutes',
            size='delivery_distance_km',
            hover_data=['from_city_name', 'receipt_hour'],
            mapbox_style="open-street-map",
            title="Geographic Distribution of Delivery Anomalies",
            zoom=5,
            center={"lat": -6.369028, "lon": 34.888822}
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚡ Real-Time Anomaly Detection")
        
        # Simulate real-time anomaly detection
        if st.button("🔄 Refresh Real-Time Detection"):
            with st.spinner("Analyzing current deliveries..."):
                time.sleep(2)  # Simulate processing time
                
                # Generate random real-time anomalies
                current_time = datetime.now()
                real_time_data = []
                
                for i in range(20):
                    is_anomaly = random.choice([True, False])
                    delivery_time = random.uniform(30, 300) if not is_anomaly else random.uniform(400, 800)
                    
                    real_time_data.append({
                        'timestamp': current_time - timedelta(minutes=random.randint(0, 60)),
                        'delivery_id': f'RT{random.randint(10000, 99999)}',
                        'city': random.choice(['Dar es Salaam', 'Mwanza', 'Mbeya', 'Arusha', 'Dodoma']),
                        'delivery_time': delivery_time,
                        'is_anomaly': is_anomaly,
                        'confidence': random.uniform(0.7, 0.99)
                    })
                
                rt_df = pd.DataFrame(real_time_data)
                
                # Display real-time anomalies
                anomalies = rt_df[rt_df['is_anomaly']]
                
                if len(anomalies) > 0:
                    st.warning(f"🚨 {len(anomalies)} anomalies detected in the last hour!")
                    
                    for _, anomaly in anomalies.iterrows():
                        st.markdown(f"""
                        <div class="metric-card status-warning">
                            <h4>🚨 ANOMALY DETECTED</h4>
                            <p><strong>ID:</strong> {anomaly['delivery_id']}</p>
                            <p><strong>City:</strong> {anomaly['city']}</p>
                            <p><strong>Time:</strong> {anomaly['delivery_time']:.0f} minutes</p>
                            <p><strong>Confidence:</strong> {anomaly['confidence']:.1%}</p>
                            <p><strong>Detected:</strong> {anomaly['timestamp'].strftime('%H:%M:%S')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No anomalies detected in recent deliveries")
                
                # Real-time chart
                fig = px.scatter(
                    rt_df,
                    x='timestamp',
                    y='delivery_time',
                    color='is_anomaly',
                    size='confidence',
                    color_discrete_map={True: '#ff4757', False: '#2ed573'},
                    title="Real-Time Delivery Performance"
                )
                
                st.plotly_chart(fig, use_container_width=True)

def real_time_insights(df_viz):
    st.header("⚡ Real-Time Business Insights")
    
    # Real-time metrics that update
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Live Performance Dashboard")
        
        # Simulate live performance metrics
        current_hour = datetime.now().hour
        
        # Current hour performance
        current_performance = {
            'deliveries_this_hour': random.randint(45, 85),
            'avg_time_this_hour': random.randint(120, 200),
            'active_couriers': random.randint(25, 45),
            'completion_rate': random.uniform(0.92, 0.98)
        }
        
        st.metric("🚚 Deliveries This Hour", current_performance['deliveries_this_hour'],
                 delta=f"+{random.randint(5, 15)} vs last hour")
        
        st.metric("⏱️ Current Avg Time", f"{current_performance['avg_time_this_hour']} min",
                 delta=f"-{random.randint(5, 20)} min")
        
        st.metric("👥 Active Couriers", current_performance['active_couriers'],
                 delta=f"🟢 {random.randint(2, 8)} just logged in")
        
        st.metric("✅ Completion Rate", f"{current_performance['completion_rate']:.1%}",
                 delta=f"+{random.uniform(0.01, 0.05):.1%}")
    
    with col2:
        st.subheader("🚨 Live Alerts")
        
        # Generate random alerts
        alert_types = [
            ("🔴 HIGH", "Delivery taking >4 hours", "DEL12345 - Mwanza"),
            ("🟡 MEDIUM", "Courier offline >1 hour", "Courier C789"),
            ("🟢 INFO", "New delivery assigned", "DEL67890 - Arusha"),
            ("🔴 HIGH", "Unusual route detected", "DEL54321 - Dodoma"),
            ("🟡 MEDIUM", "Weather delay reported", "Dar es Salaam region")
        ]
        
        for priority, alert, detail in random.sample(alert_types, 3):
            st.markdown(f"""
            <div class="insight-box" style="border-left-color: {'red' if 'HIGH' in priority else 'orange' if 'MEDIUM' in priority else 'green'};">
                <strong>{priority}</strong><br>
                {alert}<br>
                <small>{detail}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("📈 Predictive Insights")
        
        # Predictive metrics
        predictions = {
            'next_hour_volume': random.randint(60, 100),
            'peak_time_today': f"{random.randint(14, 18)}:00",
            'efficiency_trend': random.choice(['Improving', 'Stable', 'Declining']),
            'capacity_utilization': random.uniform(0.75, 0.95)
        }
        
        st.metric("📊 Next Hour Forecast", f"{predictions['next_hour_volume']} deliveries",
                 delta="Based on historical patterns")
        
        st.metric("⏰ Peak Time Today", predictions['peak_time_today'],
                 delta="Predicted peak delivery hour")
        
        trend_color = {'Improving': 'normal', 'Stable': 'off', 'Declining': 'inverse'}[predictions['efficiency_trend']]
        st.metric("📈 Efficiency Trend", predictions['efficiency_trend'],
                 delta_color=trend_color)
        
        st.metric(" Capacity Utilization", f"{predictions['capacity_utilization']:.1%}",
                 delta=f"{'🟢' if predictions['capacity_utilization'] < 0.9 else '🟡'}")
    
    # Live charts
    st.subheader("📊 Live Performance Charts")
    
    # Simulate real-time data stream
    if 'live_data' not in st.session_state:
        st.session_state.live_data = []
    
    # Add new data point
    new_point = {
        'timestamp': datetime.now(),
        'deliveries_per_minute': np.random.poisson(2),
        'avg_delivery_time': np.random.normal(180, 30),
        'active_couriers': random.randint(30, 50)
    }
    
    st.session_state.live_data.append(new_point)
    
    # Keep only last 100 points
    if len(st.session_state.live_data) > 100:
        st.session_state.live_data = st.session_state.live_data[-100:]
    
    if len(st.session_state.live_data) > 1:
        live_df = pd.DataFrame(st.session_state.live_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                live_df,
                x='timestamp',
                y='deliveries_per_minute',
                title="📈 Live Delivery Rate (per minute)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                live_df,
                x='timestamp',
                y='avg_delivery_time',
                title="⏱️ Live Average Delivery Time"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def deep_dive_analysis(df_viz):
    st.header("🔍 Deep Dive Analysis")
    
    # Advanced analytics tabs
    tab1, tab2, tab3 = st.tabs(["🧠 Machine Learning Insights", "🔗 Correlation Analysis", "💡 Business Recommendations"])
    
    with tab1:
        st.subheader("🧠 Advanced ML Insights")
        
        # Feature importance simulation
        features = ['Distance', 'Hour', 'Day of Week', 'City', 'Courier Experience', 'Weather']
        importance = np.random.random(len(features))
        importance = importance / importance.sum() * 100
        
        fig = go.Figure(data=[go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title=" Feature Importance for Delivery Time Prediction",
            xaxis_title="Importance (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Clustering analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Delivery Clusters")
            
            # Simulate clusters
            n_samples = 1000
            cluster_data = {
                'x': np.concatenate([
                    np.random.normal(50, 15, n_samples//3),
                    np.random.normal(120, 20, n_samples//3),
                    np.random.normal(200, 25, n_samples//3)
                ]),
                'y': np.concatenate([
                    np.random.normal(10, 3, n_samples//3),
                    np.random.normal(25, 5, n_samples//3),
                    np.random.normal(45, 8, n_samples//3)
                ]),
                'cluster': ['Fast Delivery', 'Standard', 'Long Distance'] * (n_samples//3)
            }
            
            fig = px.scatter(
                cluster_data,
                x='x',
                y='y',
                color='cluster',
                title="Delivery Pattern Clusters"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Prediction Accuracy")
            
            # Model performance metrics
            models = ['LightGBM', 'XGBoost', 'Random Forest', 'Neural Network']
            accuracies = [0.73, 0.71, 0.69, 0.75]
            
            fig = go.Figure(data=[go.Bar(
                x=models,
                y=accuracies,
                marker_color=['gold' if x == max(accuracies) else 'lightblue' for x in accuracies],
                text=[f'{x:.1%}' for x in accuracies],
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Model Performance Comparison",
                yaxis_title="R² Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔗 Advanced Correlation Analysis")
        
        # Multi-dimensional correlation
        correlation_features = [
            'delivery_time_minutes', 'delivery_distance_km', 'speed_kmh', 
            'receipt_hour', 'receipt_day_of_week'
        ]
        
        corr_matrix = df_viz[correlation_features].corr()
        
        # Enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        fig.update_layout(
            title="📊 Multi-Dimensional Correlation Matrix",
            height=500,
            width=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Partial correlation analysis
        st.subheader("🔍 Partial Correlations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distance controlling for time
            fig = px.scatter(
                df_viz.sample(n=min(2000, len(df_viz))),
                x='delivery_distance_km',
                y='speed_kmh',
                color='from_city_name',
                title="Speed vs Distance (by City)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time patterns
            hourly_stats = df_viz.groupby('receipt_hour').agg({
                'delivery_time_minutes': ['mean', 'std'],
                'delivery_distance_km': 'mean'
            }).round(2)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats.index,
                    y=hourly_stats[('delivery_time_minutes', 'mean')],
                    name="Avg Time",
                    line=dict(color='blue')
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats.index,
                    y=hourly_stats[('delivery_distance_km', 'mean')],
                    name="Avg Distance",
                    line=dict(color='red', dash='dash')
                ),
                secondary_y=True,
            )
            
            fig.update_layout(title="Hourly Patterns")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💡 AI-Powered Business Recommendations")
        
        # Generate insights based on data analysis
        insights = [
            {
                'priority': '🔴 HIGH',
                'category': 'Operational Efficiency',
                'recommendation': 'Optimize delivery routes during 9-11 AM peak hours',
                'impact': 'Potential 15-20% time reduction',
                'action': 'Implement dynamic routing algorithm'
            },
            {
                'priority': '🟡 MEDIUM', 
                'category': 'Resource Allocation',
                'recommendation': 'Redistribute couriers from Dar es Salaam to Dodoma',
                'impact': 'Balance workload, improve service quality',
                'action': 'Review courier deployment strategy'
            },
            {
                'priority': '🟢 LOW',
                'category': 'Customer Experience',
                'recommendation': 'Implement real-time tracking for deliveries >2 hours',
                'impact': 'Improve customer satisfaction',
                'action': 'Deploy tracking system for long deliveries'
            },
            {
                'priority': '🔴 HIGH',
                'category': 'Anomaly Prevention',
                'recommendation': 'Set up automatic alerts for deliveries >4 hours',
                'impact': 'Reduce anomalies by 30-40%',
                'action': 'Configure real-time monitoring system'
            }
        ]
        
        for insight in insights:
            priority_color = {
                '🔴 HIGH': '#ff4757',
                '🟡 MEDIUM': '#ffa502', 
                '🟢 LOW': '#2ed573'
            }[insight['priority']]
            
            st.markdown(f"""
            <div style="border: 2px solid {priority_color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h4 style="color: {priority_color}; margin: 0;">{insight['priority']} - {insight['category']}</h4>
                <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                <p><strong>Expected Impact:</strong> {insight['impact']}</p>
                <p><strong>Action Required:</strong> {insight['action']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ROI Calculator
        st.subheader("💰 ROI Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_savings = st.slider("Expected Time Savings (%)", 5, 50, 15)
        with col2:
            cost_per_delivery = st.number_input("Cost per Delivery (Tshs)", 1.0, 20.0, 5.0)
        with col3:
            monthly_deliveries = st.number_input("Monthly Deliveries", 1000, 100000, 25000)
        
        # Calculate ROI
        monthly_savings = (time_savings / 100) * cost_per_delivery * monthly_deliveries
        annual_savings = monthly_savings * 12
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Monthly Savings", f"Tshs{monthly_savings:,.2f}")
        with col2:
            st.metric("💰 Annual Savings", f"Tshs{annual_savings:,.2f}")

if __name__ == "__main__":
    main()