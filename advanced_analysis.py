import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
df = pd.read_csv('executable_files/delivery_five_cities_tanzania.csv')

print("=" * 60)
print("ADVANCED BUSINESS INTELLIGENCE ANALYSIS")
print("=" * 60)

# Parse timestamps properly
def parse_timestamp(timestamp_str, year=2023):
    try:
        return pd.to_datetime(f"{year}-{timestamp_str}", format='%Y-%m-%d %H:%M:%S')
    except:
        return pd.NaT

print('\n=== TEMPORAL ANALYSIS ===')
df['receipt_time_parsed'] = df['receipt_time'].apply(parse_timestamp)
df['sign_time_parsed'] = df['sign_time'].apply(parse_timestamp)

# Remove invalid timestamps
df_clean = df.dropna(subset=['receipt_time_parsed', 'sign_time_parsed']).copy()
print(f'Valid timestamp records: {len(df_clean):,} out of {len(df):,}')

# Calculate delivery time
df_clean['delivery_time_minutes'] = (df_clean['sign_time_parsed'] - df_clean['receipt_time_parsed']).dt.total_seconds() / 60
df_clean = df_clean[df_clean['delivery_time_minutes'] > 0]  # Remove negative delivery times

print(f'Records with valid delivery times: {len(df_clean):,}')
print(f'Delivery time statistics:')
print(f'  - Mean: {df_clean["delivery_time_minutes"].mean():.1f} minutes')
print(f'  - Median: {df_clean["delivery_time_minutes"].median():.1f} minutes')
print(f'  - Std Dev: {df_clean["delivery_time_minutes"].std():.1f} minutes')
print(f'  - Min: {df_clean["delivery_time_minutes"].min():.1f} minutes')
print(f'  - Max: {df_clean["delivery_time_minutes"].max():.1f} minutes')

# Time-based features
df_clean['receipt_hour'] = df_clean['receipt_time_parsed'].dt.hour
df_clean['receipt_day_of_week'] = df_clean['receipt_time_parsed'].dt.dayofweek
df_clean['receipt_month'] = df_clean['receipt_time_parsed'].dt.month

print('\n=== DELIVERY TIME BY CITY ===')
city_performance = df_clean.groupby('from_city_name')['delivery_time_minutes'].agg(['mean', 'median', 'std', 'count']).round(2)
print(city_performance)

print('\n=== DELIVERY TIME BY HOUR OF DAY ===')
hourly_performance = df_clean.groupby('receipt_hour')['delivery_time_minutes'].agg(['mean', 'count']).round(2)
print('Top 5 hours with fastest deliveries:')
print(hourly_performance.sort_values('mean').head())
print('\nTop 5 hours with slowest deliveries:')
print(hourly_performance.sort_values('mean').tail())

print('\n=== DELIVERY TIME BY DAY OF WEEK ===')
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_performance = df_clean.groupby('receipt_day_of_week')['delivery_time_minutes'].agg(['mean', 'count']).round(2)
daily_performance.index = [day_names[i] for i in daily_performance.index]
print(daily_performance)

print('\n=== DELIVERY PERSONNEL PERFORMANCE ===')
courier_performance = df_clean.groupby('delivery_user_id')['delivery_time_minutes'].agg(['mean', 'count']).round(2)
courier_performance = courier_performance[courier_performance['count'] >= 100]  # At least 100 deliveries
print(f'Couriers with 100+ deliveries: {len(courier_performance)}')
print('Top 10 fastest couriers (by average delivery time):')
print(courier_performance.sort_values('mean').head(10))

print('\n=== DISPATCH LOCATION PERFORMANCE ===')
dispatch_performance = df_clean.groupby('from_dipan_id')['delivery_time_minutes'].agg(['mean', 'count']).round(2)
dispatch_performance = dispatch_performance[dispatch_performance['count'] >= 1000]  # At least 1000 deliveries
print(f'Dispatch locations with 1000+ deliveries: {len(dispatch_performance)}')
print('Top 10 fastest dispatch locations:')
print(dispatch_performance.sort_values('mean').head(10))

print('\n=== DELIVERY TYPE ANALYSIS ===')
type_performance = df_clean.groupby('typecode')['delivery_time_minutes'].agg(['mean', 'median', 'count']).round(2)
type_performance = type_performance.sort_values('count', ascending=False)
print('Delivery performance by type (top 10 by volume):')
print(type_performance.head(10))

# Calculate distances for all records (using sample for memory efficiency)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371  # Earth's radius in km

# Sample for distance analysis to avoid memory issues
sample_size = 50000
if len(df_clean) > sample_size:
    df_sample = df_clean.sample(n=sample_size, random_state=42)
else:
    df_sample = df_clean.copy()

print(f'\n=== DISTANCE ANALYSIS (Sample of {len(df_sample):,} records) ===')
df_sample['delivery_distance_km'] = haversine_distance(
    df_sample['poi_lat'], df_sample['poi_lng'],
    df_sample['sign_lat'], df_sample['sign_lng']
)

df_sample['pickup_distance_km'] = haversine_distance(
    df_sample['receipt_lat'], df_sample['receipt_lng'],
    df_sample['poi_lat'], df_sample['poi_lng']
)

# Calculate speed
df_sample['average_speed_kmh'] = (
    df_sample['delivery_distance_km'] / (df_sample['delivery_time_minutes'] / 60)
).fillna(0)
df_sample['average_speed_kmh'] = df_sample['average_speed_kmh'].replace([np.inf, -np.inf], 0)

print(f'Delivery distance statistics:')
print(f'  - Mean: {df_sample["delivery_distance_km"].mean():.2f} km')
print(f'  - Median: {df_sample["delivery_distance_km"].median():.2f} km')
print(f'  - Min: {df_sample["delivery_distance_km"].min():.2f} km')
print(f'  - Max: {df_sample["delivery_distance_km"].max():.2f} km')

print(f'\nPickup distance statistics:')
print(f'  - Mean: {df_sample["pickup_distance_km"].mean():.2f} km')
print(f'  - Median: {df_sample["pickup_distance_km"].median():.2f} km')

print(f'\nDelivery speed statistics (km/h):')
speed_stats = df_sample[df_sample['average_speed_kmh'] > 0]['average_speed_kmh']
print(f'  - Mean: {speed_stats.mean():.2f} km/h')
print(f'  - Median: {speed_stats.median():.2f} km/h')
print(f'  - Min: {speed_stats.min():.2f} km/h')
print(f'  - Max: {speed_stats.max():.2f} km/h')

print('\n=== DISTANCE VS DELIVERY TIME CORRELATION ===')
correlation = df_sample[['delivery_distance_km', 'delivery_time_minutes', 'average_speed_kmh']].corr()
print(correlation)

print('\n=== BUSINESS INSIGHTS SUMMARY ===')
print('1. OPERATIONAL EFFICIENCY:')
avg_delivery_time = df_clean['delivery_time_minutes'].mean()
print(f'   - Average delivery time: {avg_delivery_time:.1f} minutes ({avg_delivery_time/60:.1f} hours)')

fastest_city = city_performance['mean'].idxmin()
slowest_city = city_performance['mean'].idxmax()
print(f'   - Fastest city: {fastest_city} ({city_performance.loc[fastest_city, "mean"]:.1f} min avg)')
print(f'   - Slowest city: {slowest_city} ({city_performance.loc[slowest_city, "mean"]:.1f} min avg)')

print('\n2. WORKFORCE INSIGHTS:')
print(f'   - Total delivery personnel: {df_clean["delivery_user_id"].nunique():,}')
print(f'   - Average deliveries per courier: {len(df_clean) / df_clean["delivery_user_id"].nunique():.1f}')
active_couriers = len(courier_performance)
print(f'   - Active couriers (100+ deliveries): {active_couriers}')

print('\n3. NETWORK COMPLEXITY:')
print(f'   - Dispatch locations: {df_clean["from_dipan_id"].nunique():,}')
print(f'   - Service areas (AOI): {df_clean["aoi_id"].nunique():,}')
print(f'   - Delivery types: {df_clean["typecode"].nunique():,}')

print('\n4. GEOGRAPHIC SCOPE:')
print(f'   - Average delivery distance: {df_sample["delivery_distance_km"].mean():.1f} km')
print(f'   - Coverage area: Tanzania ({df_clean["poi_lat"].min():.2f}° to {df_clean["poi_lat"].max():.2f}° latitude)')

print('\n5. PERFORMANCE PATTERNS:')
best_hour = hourly_performance['mean'].idxmin()
worst_hour = hourly_performance['mean'].idxmax()
print(f'   - Best delivery hour: {best_hour}:00 ({hourly_performance.loc[best_hour, "mean"]:.1f} min avg)')
print(f'   - Worst delivery hour: {worst_hour}:00 ({hourly_performance.loc[worst_hour, "mean"]:.1f} min avg)')

best_day = daily_performance['mean'].idxmin()
worst_day = daily_performance['mean'].idxmax()
print(f'   - Best delivery day: {best_day} ({daily_performance.loc[best_day, "mean"]:.1f} min avg)')
print(f'   - Worst delivery day: {worst_day} ({daily_performance.loc[worst_day, "mean"]:.1f} min avg)')

print('\n6. DATA QUALITY FOR ML:')
print(f'   - Clean records for training: {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)')
print(f'   - Features available: Geographic coordinates, temporal patterns, personnel IDs')
print(f'   - Target variable: Delivery time (continuous, {df_clean["delivery_time_minutes"].min():.1f} to {df_clean["delivery_time_minutes"].max():.1f} minutes)')

print(f'\n7. ANOMALY DETECTION POTENTIAL:')
# Calculate IQR for outlier detection
Q1 = df_clean['delivery_time_minutes'].quantile(0.25)
Q3 = df_clean['delivery_time_minutes'].quantile(0.75)
IQR = Q3 - Q1
outliers = len(df_clean[(df_clean['delivery_time_minutes'] < Q1 - 1.5*IQR) | 
                       (df_clean['delivery_time_minutes'] > Q3 + 1.5*IQR)])
print(f'   - Potential outliers (IQR method): {outliers:,} ({outliers/len(df_clean)*100:.1f}%)')
print(f'   - Normal delivery range: {Q1:.1f} to {Q3:.1f} minutes')

print("\n" + "=" * 60)
print("ADVANCED ANALYSIS COMPLETE")
print("=" * 60)