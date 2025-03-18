import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, bds
from datetime import datetime
import os
import io
import base64
from scipy.stats import shapiro, jarque_bera
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Agricultural Commodity Price Analysis",
    page_icon="🌾",
    layout="wide"
)

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'date_processed' not in st.session_state:
    st.session_state.date_processed = False
if 'selected_markets' not in st.session_state:
    st.session_state.selected_markets = []
if 'best_cvgs' not in st.session_state:
    st.session_state.best_cvgs = None
if 'selected_cvg' not in st.session_state:
    st.session_state.selected_cvg = None
if 'markets_processed' not in st.session_state:
    st.session_state.markets_processed = False
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'min_price_threshold' not in st.session_state:
    st.session_state.min_price_threshold = None

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .analysis-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def create_cvg_identifier(row):
    """Create CVG identifier using full names."""
    return f"{row['Commodity']}_{row['Variety']}_{row['Grade']}_{row['Market']}"

def select_best_cvgs(df, n=3, preferred_markets=None):
    """Select best CVG combinations based on data completeness and number of observations."""
    # First, get unique commodities
    unique_commodities = df['Commodity'].unique()
    
    # Initialize list to store best CVGs
    best_cvgs = []
    
    # For each commodity, find the best CVG combinations
    for commodity in unique_commodities:
        # Filter data for current commodity
        commodity_df = df[df['Commodity'] == commodity]
        
        cvg_stats = commodity_df.groupby('CVG').agg({
            'Date': ['min', 'max', 'count'],
            'Modal': 'count'
        }).reset_index()
        
        cvg_stats.columns = ['CVG', 'start_date', 'end_date', 'total_count', 'non_null_count']
        cvg_stats['span'] = (cvg_stats['end_date'] - cvg_stats['start_date']).dt.days
        
        # Calculate records per year to normalize across different time spans
        cvg_stats['years_span'] = cvg_stats['span'] / 365.25
        cvg_stats['records_per_year'] = cvg_stats['non_null_count'] / cvg_stats['years_span']
        
        # Filter for last 5 years
        five_years_ago = pd.Timestamp.now() - pd.DateOffset(years=5)
        cvg_stats = cvg_stats[cvg_stats['end_date'] >= five_years_ago]
        
        # Set minimum threshold for records
        min_records = 500  # Minimum number of records required
        cvg_stats = cvg_stats[cvg_stats['non_null_count'] >= min_records]
        
        # Sort by number of records and records per year
        cvg_stats = cvg_stats.sort_values(['non_null_count', 'records_per_year'], ascending=[False, False])
        
        if preferred_markets:
            # Filter CVGs for preferred markets within this commodity
            commodity_preferred_cvgs = []
            for market in preferred_markets:
                market_cvgs = cvg_stats[cvg_stats['CVG'].str.contains(market, case=False)]
                if not market_cvgs.empty:
                    # Sort market CVGs by number of records
                    market_cvgs = market_cvgs.sort_values('non_null_count', ascending=False)
                    commodity_preferred_cvgs.append(market_cvgs.iloc[0]['CVG'])
            
            # Add remaining CVGs based on highest number of records
            remaining_slots = n - len(commodity_preferred_cvgs)
            if remaining_slots > 0:
                remaining_cvgs = cvg_stats[~cvg_stats['CVG'].isin(commodity_preferred_cvgs)]
                remaining_cvgs = remaining_cvgs.sort_values('non_null_count', ascending=False)
                commodity_preferred_cvgs.extend(remaining_cvgs.head(remaining_slots)['CVG'].tolist())
            
            best_cvgs.extend(commodity_preferred_cvgs)
        else:
            best_cvgs.extend(cvg_stats.head(n)['CVG'].tolist())
    
    return best_cvgs

def validate_prices(df, min_price_threshold=100):
    """
    Validate and clean price data by removing unrealistic values.
    Args:
        df: DataFrame with price data
        min_price_threshold: Minimum acceptable price (default 100 Rs./Quintal)
    Returns:
        cleaned_df: DataFrame with validated prices
        stats: Dictionary with validation statistics
    """
    original_count = len(df)
    
    # Create a copy to avoid modifying original data
    cleaned_df = df.copy()
    
    # Identify unrealistic prices
    unrealistic_mask = cleaned_df['Modal'] < min_price_threshold
    unrealistic_count = unrealistic_mask.sum()
    
    # Remove unrealistic prices
    cleaned_df = cleaned_df[~unrealistic_mask]
    
    # Calculate validation statistics
    stats = {
        'original_records': original_count,
        'unrealistic_records': unrealistic_count,
        'remaining_records': len(cleaned_df),
        'removed_percentage': (unrealistic_count / original_count * 100) if original_count > 0 else 0
    }
    
    return cleaned_df, stats

def interpolate_data(df):
    """
    Interpolate missing values using linear interpolation.
    Args:
        df: DataFrame with numeric columns to interpolate
    Returns:
        DataFrame with interpolated values
    """
    # Sort by date first
    df = df.sort_values('Date')
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    numeric_cols = ['Arrivals', 'Min', 'Max', 'Modal']
    for col in numeric_cols:
        if col in df_copy.columns:
            # Forward fill first to handle leading NAs
            df_copy[col] = df_copy[col].fillna(method='ffill')
            # Then interpolate remaining gaps
            df_copy[col] = df_copy[col].interpolate(method='linear')
            # Back fill any remaining trailing NAs
            df_copy[col] = df_copy[col].fillna(method='bfill')
    
    return df_copy

def process_weather_data(weather_df):
    """Process weather data and create date column."""
    # Create date column from YEAR, MO, DY columns
    weather_df['Date'] = pd.to_datetime({
        'year': weather_df['YEAR'],
        'month': weather_df['MO'],
        'day': weather_df['DY']
    })
    
    # Select all columns except YEAR, MO, DY for processing
    weather_cols = [col for col in weather_df.columns if col not in ['YEAR', 'MO', 'DY']]
    
    # Create processed dataframe with all weather variables
    processed_df = pd.DataFrame({'Date': weather_df['Date']})
    for col in weather_cols:
        if col != 'Date':
            processed_df[col] = weather_df[col]
    
    # Handle missing values (replace -999 with NaN)
    processed_df = processed_df.replace(-999, np.nan)
    
    # Sort by date
    processed_df = processed_df.sort_values('Date')
    
    return processed_df

def analyze_data(cvg_data, weather_data=None):
    """Perform basic analysis on the data."""
    analysis_results = {}
    
    # Basic Statistics
    analysis_results['basic_stats'] = {
        "Maximum Price": cvg_data['Modal'].max(),
        "Mean Price": cvg_data['Modal'].mean(),
        "Minimum Price": cvg_data['Modal'].min(),
        "Standard Deviation": cvg_data['Modal'].std(),
        "Coefficient of Variation": (cvg_data['Modal'].std() / cvg_data['Modal'].mean()) * 100,
        "Skewness": cvg_data['Modal'].skew(),
        "Kurtosis": cvg_data['Modal'].kurtosis()
    }
    
    return analysis_results

def generate_report(cvg_data, selected_cvg, analysis_results):
    """Generate a basic report for the analyzed crop."""
    report = []
    
    # Add header
    report.append(f"# Agricultural Commodity Price Analysis Report")
    report.append(f"## CVG Identifier: {selected_cvg}")
    report.append(f"## Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add basic statistics
    if 'basic_stats' in analysis_results:
        report.append("### Basic Statistics")
        report.append(pd.DataFrame(analysis_results['basic_stats'], index=[0]).to_string())
    
    return "\n".join(report)

def get_download_link(text, filename):
    """Generate a download link for the report."""
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Report</a>'

def parse_date(date_str):
    """Parse date string in various formats."""
    if pd.isna(date_str):
        return pd.NaT
        
    try:
        # Try different date formats
        formats = [
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%d.%m.%Y',
            '%Y.%m.%d',
            '%d-%b-%Y',  # For dates like '15-Jan-2023'
            '%d/%b/%Y',  # For dates like '15/Jan/2023'
            '%Y-%b-%d',  # For dates like '2023-Jan-15'
            '%b-%d-%Y',  # For dates like 'Jan-15-2023'
            '%b/%d/%Y'   # For dates like 'Jan/15/2023'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # If none of the specific formats work, try pandas' default parser
        parsed_date = pd.to_datetime(date_str)
        
        # Validate the parsed date is within a reasonable range
        if parsed_date.year < 1900 or parsed_date.year > 2100:
            return pd.NaT
            
        return parsed_date
    except:
        return pd.NaT

def resample_data(df, freq='W'):
    """
    Resample data to specified frequency.
    Args:
        df: DataFrame with Date and Modal columns
        freq: Frequency for resampling ('W' for weekly, 'M' for monthly)
    Returns:
        DataFrame with resampled data
    """
    # Sort data by date first
    df = df.sort_values('Date')
    
    # Set date as index and resample
    df_resampled = df.set_index('Date')
    
    # Resample using mean and handle missing values
    resampled = df_resampled['Modal'].resample(freq).mean()
    
    # Convert back to DataFrame and reset index
    result = pd.DataFrame({'Date': resampled.index, 'Modal': resampled.values})
    
    return result

def get_descriptive_stats(data, frequency="Daily"):
    """Calculate descriptive statistics for the given data."""
    shapiro_stat, shapiro_p = shapiro(data['Modal'])
    jb_stat, jb_p = jarque_bera(data['Modal'])
    
    stats_dict = {
        'Descriptive statistics': [
            'Maximum (Rs./Quintal)',
            'Mean (Rs./Quintal)',
            'Minimum (Rs./Quintal)',
            'Standard deviation (Rs./Quintal)',
            'Coefficient of variation (%)',
            'Skewness',
            'Kurtosis',
            "Shapiro–Wilk's test",
            'Jarque–Bera test'
        ],
        frequency: [
            f"{data['Modal'].max():.2f}",
            f"{data['Modal'].mean():.2f}",
            f"{data['Modal'].min():.2f}",
            f"{data['Modal'].std():.2f}",
            f"{(data['Modal'].std() / data['Modal'].mean() * 100):.2f}",
            f"{data['Modal'].skew():.2f}",
            f"{data['Modal'].kurtosis():.2f}",
            f"{shapiro_stat:.2f} ({'< 0.0001' if shapiro_p < 0.0001 else f'{shapiro_p:.4f}'})",
            f"{jb_stat:.2f} ({'< 0.0001' if jb_p < 0.0001 else f'{jb_p:.4f}'})"
        ]
    }
    return pd.DataFrame(stats_dict)

def auto_process_dates(df):
    """Automatically process dates and handle invalid entries."""
    # Create a copy of original dates for reporting
    original_dates = df['Date'].copy()
    
    # Try to parse dates
    df['Date'] = df['Date'].apply(parse_date)
    
    # Check for invalid dates
    invalid_mask = df['Date'].isna()
    invalid_dates = invalid_mask.sum()
    
    # Create validation report
    validation_report = {
        'total_records': len(df),
        'invalid_dates': invalid_dates,
        'valid_dates': len(df) - invalid_dates,
        'invalid_rows': pd.DataFrame({
            'Row Number': np.where(invalid_mask)[0] + 1,
            'Original Date': original_dates[invalid_mask],
            'Market': df.loc[invalid_mask, 'Market'],
            'Commodity': df.loc[invalid_mask, 'Commodity']
        }) if invalid_dates > 0 else None
    }
    
    # Remove rows with invalid dates
    df = df[~invalid_mask].copy()
    
    return df, validation_report

def merge_commodity_files(uploaded_files):
    """Merge multiple commodity files into a single DataFrame."""
    merged_df = pd.DataFrame()
    processing_results = []
    
    for file in uploaded_files:
        try:
            # Read the file
            df = pd.read_csv(file)
            
            # Store original row count
            original_rows = len(df)
            
            # Check required columns
            required_cols = ['Date', 'Commodity', 'Market', 'Modal']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                processing_results.append({
                    'file': file.name,
                    'status': 'Failed',
                    'message': f"Missing required columns: {', '.join(missing_cols)}",
                    'rows_added': 0
                })
                continue
            
            # Append to merged DataFrame
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            
            processing_results.append({
                'file': file.name,
                'status': 'Success',
                'message': 'File processed successfully',
                'rows_added': original_rows
            })
            
        except Exception as e:
            processing_results.append({
                'file': file.name,
                'status': 'Failed',
                'message': str(e),
                'rows_added': 0
            })
    
    return merged_df, processing_results

def perform_seasonal_decomposition(data, freq):
    """
    Perform seasonal decomposition on time series data.
    
    Args:
        data: DataFrame with Date and Modal columns
        freq: Frequency for decomposition ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        Decomposition result object
    """
    # Set date as index
    ts_data = data.set_index('Date')['Modal']
    
    # Determine period based on frequency
    if freq == 'D':
        period = 30  # Monthly seasonality for daily data
    elif freq == 'W':
        period = 52  # Yearly seasonality for weekly data
    else:  # Monthly
        period = 12  # Yearly seasonality for monthly data
    
    # Perform decomposition
    decomposition = seasonal_decompose(
        ts_data,
        period=period,
        extrapolate_trend='freq'
    )
    
    return decomposition

def plot_decomposition(decomposition, title):
    """
    Create an interactive plot for seasonal decomposition results.
    
    Args:
        decomposition: Seasonal decomposition result object
        title: Title for the plot
    
    Returns:
        Plotly figure object
    """
    # Create subplot figure
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # Add traces for each component
    fig.add_trace(
        go.Scatter(x=decomposition.observed.index, y=decomposition.observed,
                  mode='lines', name='Observed', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                  mode='lines', name='Trend', line=dict(color='#2ca02c')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                  mode='lines', name='Seasonal', line=dict(color='#ff7f0e')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                  mode='lines', name='Residual', line=dict(color='#d62728')),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=title,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Trend', row=2, col=1)
    fig.update_yaxes(title_text='Seasonal', row=3, col=1)
    fig.update_yaxes(title_text='Residual', row=4, col=1)
    
    return fig

def perform_adf_test(data, commodity_name):
    """
    Perform Augmented Dickey-Fuller test and format results.
    
    Args:
        data: DataFrame with Modal price column
        commodity_name: Name of the commodity/CVG
    
    Returns:
        DataFrame with formatted ADF test results
    """
    # Perform ADF test
    adf_result = adfuller(data['Modal'].dropna(), autolag='AIC')
    
    # Format p-value
    p_value = adf_result[1]
    p_value_formatted = '< 0.0001' if p_value < 0.0001 else f'{p_value:.4f}'
    
    # Determine stationarity
    is_stationary = p_value < 0.05
    remark = 'Stationary' if is_stationary else 'Non-Stationary'
    
    # Create results DataFrame
    results_dict = {
        'Data': [commodity_name],
        'Test Statistic': [f'{adf_result[0]:.4f}'],
        'p-value': [p_value_formatted],
        'Lags Used': [adf_result[2]],
        'Remarks': [remark]
    }
    
    return pd.DataFrame(results_dict)

def perform_bds_test(data, commodity_name):
    """
    Perform BDS test for independence of a time series.
    
    Args:
        data: DataFrame with Modal price column
        commodity_name: Name of the commodity/CVG
    
    Returns:
        DataFrame with formatted BDS test results
    """
    # Prepare data
    price_data = data['Modal'].dropna().values
    
    # Define distance multipliers for epsilon calculation
    distances = [0.5, 1.0, 1.5, 2.0]
    
    results = []
    for dist in distances:
        try:
            # Perform BDS test with max_dim=3 to get results for dimensions 2 and 3
            bds_stat, p_value = bds(price_data, max_dim=3, distance=dist)
            
            # Process results for each dimension (results start from dimension 2)
            for dim in range(2, 4):  # dimensions 2 and 3
                idx = dim - 2  # index in the results array
                
                # Get statistics for current dimension
                stat = bds_stat[idx] if isinstance(bds_stat, np.ndarray) else bds_stat
                p_val = p_value[idx] if isinstance(p_value, np.ndarray) else p_value
                
                # Format p-value
                p_value_formatted = '< 0.0001' if p_val < 0.0001 else f'{p_val:.4f}'
                
                # Determine independence based on p-value
                if p_val < 0.0001:
                    interpretation = 'Strong evidence against i.i.d.'
                elif p_val < 0.05:
                    interpretation = 'Evidence against i.i.d.'
                else:
                    interpretation = 'No evidence against i.i.d.'
                
                results.append({
                    'Agricultural Crops': commodity_name,
                    'Distance Multiplier': f'{dist:.1f}',
                    'Dimension (m)': dim,
                    'BDS Statistic': f'{stat:.4f}',
                    'p-value': p_value_formatted,
                    'Interpretation': interpretation
                })
        except Exception as e:
            print(f"Error in BDS test for distance {dist}: {str(e)}")
            results.append({
                'Agricultural Crops': commodity_name,
                'Distance Multiplier': f'{dist:.1f}',
                'Dimension (m)': 'Error',
                'BDS Statistic': 'Error',
                'p-value': 'Error',
                'Interpretation': str(e)
            })
    
    return pd.DataFrame(results)

def resample_weather_data(data, freq='W'):
    """
    Resample weather data to specified frequency.
    
    Args:
        data: DataFrame with Date and weather variables
        freq: Frequency for resampling ('W' for weekly, 'M' for monthly)
    
    Returns:
        DataFrame with resampled data
    """
    # Set date as index
    df = data.copy()
    df = df.set_index('Date')
    
    # Resample all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    resampled = df[numeric_cols].resample(freq).mean()
    
    # Reset index to get Date back as column
    resampled = resampled.reset_index()
    
    return resampled

def main():
    st.title("🌾 Agricultural Commodity Price Analysis")
    
    # File uploads
    st.markdown("## Data Upload")
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Commodity Price Data")
        price_files = st.file_uploader(
            "Upload commodity price data files (CSV format)",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if price_files:
            st.markdown(f"**{len(price_files)} files selected**")
    
    with col2:
        st.markdown("### Weather Data")
        weather_file = st.file_uploader(
            "Upload weather data (CSV format)",
            type=['csv']
        )
        if weather_file is not None:
            try:
                weather_df = pd.read_csv(weather_file)
                st.session_state.weather_data = process_weather_data(weather_df)
                st.success("Weather data processed successfully!")
                
                # Display weather data summary
                st.markdown("#### Weather Data Summary")
                weather_summary = {
                    'Date Range': f"{st.session_state.weather_data['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.weather_data['Date'].max().strftime('%Y-%m-%d')}",
                    'Total Records': len(st.session_state.weather_data),
                    'Variables Available': ', '.join([col for col in st.session_state.weather_data.columns if col != 'Date'])
                }
                st.table(pd.DataFrame([weather_summary]).T.rename(columns={0: 'Value'}))
            except Exception as e:
                st.error(f"Error processing weather file: {str(e)}")
                st.session_state.weather_data = None

    if price_files:
        try:
            # Process and merge files if not already done
            if st.session_state.processed_df is None:
                # Merge uploaded files
                df, processing_results = merge_commodity_files(price_files)
                
                # Display processing results
                st.markdown("### File Processing Results")
                
                # Create a DataFrame for results
                results_df = pd.DataFrame(processing_results)
                
                # Display successful files
                successful_files = results_df[results_df['status'] == 'Success']
                if not successful_files.empty:
                    st.markdown("#### ✅ Successfully Processed Files")
                    st.table(successful_files[['file', 'rows_added']])
                    st.markdown(f"Total rows added: {successful_files['rows_added'].sum():,}")
                
                # Display failed files
                failed_files = results_df[results_df['status'] == 'Failed']
                if not failed_files.empty:
                    st.markdown("#### ❌ Failed Files")
                    st.table(failed_files[['file', 'message']])
                
                if df.empty:
                    st.error("No valid data to process. Please check your files and try again.")
                    return
                
                # Automatically process dates
                df, date_validation = auto_process_dates(df)
                
                # Display date validation results
                st.markdown("### Date Processing Results")
                st.markdown(f"""
                - Total records: {date_validation['total_records']:,}
                - Valid dates: {date_validation['valid_dates']:,}
                - Invalid dates removed: {date_validation['invalid_dates']:,}
                """)
                
                if date_validation['invalid_dates'] > 0:
                    with st.expander("View Invalid Date Details"):
                        st.markdown("### Removed Invalid Date Entries")
                        st.dataframe(date_validation['invalid_rows'])
                
                st.session_state.processed_df = df
                st.session_state.date_processed = True
            else:
                df = st.session_state.processed_df

            # Weather Data Analysis Section
            if st.session_state.weather_data is not None:
                st.markdown("## Weather Data Analysis")
                st.markdown("### Correlation Analysis with Commodity Prices")
                
                # Select commodity and CVG for correlation analysis
                unique_commodities = sorted(df['Commodity'].unique())
                selected_commodity = st.selectbox(
                    "Select Commodity for Weather Analysis",
                    unique_commodities,
                    key="weather_commodity"
                )
                
                # Filter data for selected commodity
                commodity_df = df[df['Commodity'] == selected_commodity]
                
                # Create CVG identifier
                commodity_df['CVG'] = commodity_df.apply(create_cvg_identifier, axis=1)
                cvg_options = sorted(commodity_df['CVG'].unique())
                
                selected_cvg = st.selectbox(
                    "Select CVG for Weather Analysis",
                    cvg_options,
                    key="weather_cvg"
                )
                
                if st.button("Analyze Weather Correlations"):
                    with st.spinner("Calculating correlations..."):
                        # Filter data for selected CVG
                        cvg_data = commodity_df[commodity_df['CVG'] == selected_cvg].copy()
                        
                        # Create tabs for different frequencies
                        daily_tab, weekly_tab, monthly_tab = st.tabs(["Daily Data", "Weekly Data", "Monthly Data"])
                        
                        # Daily Analysis
                        with daily_tab:
                            st.markdown("#### Daily Weather Correlations")
                            
                            # Merge price and weather data
                            merged_daily = pd.merge(
                                cvg_data[['Date', 'Modal']],
                                st.session_state.weather_data,
                                on='Date',
                                how='inner'
                            )
                            
                            if len(merged_daily) > 0:
                                # Calculate correlations for daily data
                                weather_vars = [col for col in merged_daily.columns if col not in ['Date', 'Modal']]
                                daily_correlations = []
                                
                                for var in weather_vars:
                                    correlation = merged_daily['Modal'].corr(pd.to_numeric(merged_daily[var], errors='coerce'))
                                    _, p_value = stats.pearsonr(
                                        pd.to_numeric(merged_daily['Modal'], errors='coerce'),
                                        pd.to_numeric(merged_daily[var], errors='coerce')
                                    )
                                    
                                    daily_correlations.append({
                                        'Weather Variable': var,
                                        'Correlation': f"{correlation:.4f}",
                                        'p-value': '< 0.0001' if p_value < 0.0001 else f"{p_value:.4f}",
                                        'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                                    })
                                
                                # Display daily correlation results
                                daily_corr_df = pd.DataFrame(daily_correlations)
                                st.markdown("##### Correlation Results (Daily)")
                                st.markdown("""
                                Significance levels:
                                - *** : p < 0.001
                                - ** : p < 0.01
                                - * : p < 0.05
                                - ns : not significant
                                """)
                                st.table(daily_corr_df.style.set_properties(**{
                                    'text-align': 'center',
                                    'font-size': '1em'
                                }))
                                
                                # Daily correlation visualization
                                st.markdown("##### Correlation Visualization (Daily)")
                                fig = px.bar(
                                    daily_corr_df,
                                    x='Weather Variable',
                                    y='Correlation',
                                    title=f'Daily Weather Variable Correlations with {selected_cvg} Prices',
                                    labels={'Correlation': 'Correlation Coefficient'}
                                )
                                fig.update_layout(
                                    xaxis_tickangle=-45,
                                    showlegend=False,
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Daily time series plots
                                st.markdown("##### Time Series Visualization (Daily)")
                                for var in weather_vars:
                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=merged_daily['Date'],
                                            y=merged_daily['Modal'],
                                            name='Price',
                                            line=dict(color='blue')
                                        ),
                                        secondary_y=False
                                    )
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=merged_daily['Date'],
                                            y=pd.to_numeric(merged_daily[var], errors='coerce'),
                                            name=var,
                                            line=dict(color='red')
                                        ),
                                        secondary_y=True
                                    )
                                    
                                    fig.update_layout(
                                        title=f'{var} vs Price Over Time (Daily)',
                                        xaxis_title='Date',
                                        height=400
                                    )
                                    fig.update_yaxes(title_text="Price (Rs./Quintal)", secondary_y=False)
                                    fig.update_yaxes(title_text=var, secondary_y=True)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No matching dates found between daily weather and price data.")
                        
                        # Weekly Analysis
                        with weekly_tab:
                            st.markdown("#### Weekly Weather Correlations")
                            
                            # Resample price data to weekly
                            weekly_price = resample_data(cvg_data[['Date', 'Modal']], 'W')
                            
                            # Resample weather data to weekly
                            weekly_weather = resample_weather_data(st.session_state.weather_data, 'W')
                            
                            # Merge weekly data
                            merged_weekly = pd.merge(
                                weekly_price,
                                weekly_weather,
                                on='Date',
                                how='inner'
                            )
                            
                            if len(merged_weekly) > 0:
                                # Calculate correlations for weekly data
                                weather_vars = [col for col in merged_weekly.columns if col not in ['Date', 'Modal']]
                                weekly_correlations = []
                                
                                for var in weather_vars:
                                    correlation = merged_weekly['Modal'].corr(pd.to_numeric(merged_weekly[var], errors='coerce'))
                                    _, p_value = stats.pearsonr(
                                        pd.to_numeric(merged_weekly['Modal'], errors='coerce'),
                                        pd.to_numeric(merged_weekly[var], errors='coerce')
                                    )
                                    
                                    weekly_correlations.append({
                                        'Weather Variable': var,
                                        'Correlation': f"{correlation:.4f}",
                                        'p-value': '< 0.0001' if p_value < 0.0001 else f"{p_value:.4f}",
                                        'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                                    })
                                
                                # Display weekly correlation results
                                weekly_corr_df = pd.DataFrame(weekly_correlations)
                                st.markdown("##### Correlation Results (Weekly)")
                                st.markdown("""
                                Significance levels:
                                - *** : p < 0.001
                                - ** : p < 0.01
                                - * : p < 0.05
                                - ns : not significant
                                """)
                                st.table(weekly_corr_df.style.set_properties(**{
                                    'text-align': 'center',
                                    'font-size': '1em'
                                }))
                                
                                # Weekly correlation visualization
                                st.markdown("##### Correlation Visualization (Weekly)")
                                fig = px.bar(
                                    weekly_corr_df,
                                    x='Weather Variable',
                                    y='Correlation',
                                    title=f'Weekly Weather Variable Correlations with {selected_cvg} Prices',
                                    labels={'Correlation': 'Correlation Coefficient'}
                                )
                                fig.update_layout(
                                    xaxis_tickangle=-45,
                                    showlegend=False,
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Weekly time series plots
                                st.markdown("##### Time Series Visualization (Weekly)")
                                for var in weather_vars:
                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=merged_weekly['Date'],
                                            y=merged_weekly['Modal'],
                                            name='Price',
                                            line=dict(color='blue')
                                        ),
                                        secondary_y=False
                                    )
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=merged_weekly['Date'],
                                            y=pd.to_numeric(merged_weekly[var], errors='coerce'),
                                            name=var,
                                            line=dict(color='red')
                                        ),
                                        secondary_y=True
                                    )
                                    
                                    fig.update_layout(
                                        title=f'{var} vs Price Over Time (Weekly)',
                                        xaxis_title='Date',
                                        height=400
                                    )
                                    fig.update_yaxes(title_text="Price (Rs./Quintal)", secondary_y=False)
                                    fig.update_yaxes(title_text=var, secondary_y=True)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No matching dates found between weekly weather and price data.")
                        
                        # Monthly Analysis
                        with monthly_tab:
                            st.markdown("#### Monthly Weather Correlations")
                            
                            # Resample price data to monthly
                            monthly_price = resample_data(cvg_data[['Date', 'Modal']], 'M')
                            
                            # Resample weather data to monthly
                            monthly_weather = resample_weather_data(st.session_state.weather_data, 'M')
                            
                            # Merge monthly data
                            merged_monthly = pd.merge(
                                monthly_price,
                                monthly_weather,
                                on='Date',
                                how='inner'
                            )
                            
                            if len(merged_monthly) > 0:
                                # Calculate correlations for monthly data
                                weather_vars = [col for col in merged_monthly.columns if col not in ['Date', 'Modal']]
                                monthly_correlations = []
                                
                                for var in weather_vars:
                                    correlation = merged_monthly['Modal'].corr(pd.to_numeric(merged_monthly[var], errors='coerce'))
                                    _, p_value = stats.pearsonr(
                                        pd.to_numeric(merged_monthly['Modal'], errors='coerce'),
                                        pd.to_numeric(merged_monthly[var], errors='coerce')
                                    )
                                    
                                    monthly_correlations.append({
                                        'Weather Variable': var,
                                        'Correlation': f"{correlation:.4f}",
                                        'p-value': '< 0.0001' if p_value < 0.0001 else f"{p_value:.4f}",
                                        'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                                    })
                                
                                # Display monthly correlation results
                                monthly_corr_df = pd.DataFrame(monthly_correlations)
                                st.markdown("##### Correlation Results (Monthly)")
                                st.markdown("""
                                Significance levels:
                                - *** : p < 0.001
                                - ** : p < 0.01
                                - * : p < 0.05
                                - ns : not significant
                                """)
                                st.table(monthly_corr_df.style.set_properties(**{
                                    'text-align': 'center',
                                    'font-size': '1em'
                                }))
                                
                                # Monthly correlation visualization
                                st.markdown("##### Correlation Visualization (Monthly)")
                                fig = px.bar(
                                    monthly_corr_df,
                                    x='Weather Variable',
                                    y='Correlation',
                                    title=f'Monthly Weather Variable Correlations with {selected_cvg} Prices',
                                    labels={'Correlation': 'Correlation Coefficient'}
                                )
                                fig.update_layout(
                                    xaxis_tickangle=-45,
                                    showlegend=False,
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Monthly time series plots
                                st.markdown("##### Time Series Visualization (Monthly)")
                                for var in weather_vars:
                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=merged_monthly['Date'],
                                            y=merged_monthly['Modal'],
                                            name='Price',
                                            line=dict(color='blue')
                                        ),
                                        secondary_y=False
                                    )
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=merged_monthly['Date'],
                                            y=pd.to_numeric(merged_monthly[var], errors='coerce'),
                                            name=var,
                                            line=dict(color='red')
                                        ),
                                        secondary_y=True
                                    )
                                    
                                    fig.update_layout(
                                        title=f'{var} vs Price Over Time (Monthly)',
                                        xaxis_title='Date',
                                        height=400
                                    )
                                    fig.update_yaxes(title_text="Price (Rs./Quintal)", secondary_y=False)
                                    fig.update_yaxes(title_text=var, secondary_y=True)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No matching dates found between monthly weather and price data.")
            
            # Display data summary
            with st.expander("View Data Summary"):
                st.markdown("### Data Summary")
                summary_stats = {
                    'Total Records': len(df),
                    'Unique Commodities': len(df['Commodity'].unique()),
                    'Unique Markets': len(df['Market'].unique()),
                    'Date Range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
                }
                st.table(pd.DataFrame([summary_stats]).T.rename(columns={0: 'Value'}))
                
                st.markdown("#### Records by Commodity")
                commodity_counts = df.groupby('Commodity').size().reset_index()
                commodity_counts.columns = ['Commodity', 'Records']
                st.table(commodity_counts.sort_values('Records', ascending=False))
            
            if st.button("Reset Data Processing", use_container_width=True):
                st.session_state.date_processed = False
                st.session_state.processed_df = None
                st.session_state.markets_processed = False
                st.session_state.selected_markets = []
                st.session_state.best_cvgs = None
                st.session_state.selected_cvg = None
                st.session_state.min_price_threshold = None
                st.experimental_rerun()
            
            # Proceed directly to Step 2: Market Selection
            if st.session_state.date_processed:
                # Step 2: Market Selection
                st.markdown("## Step 2: Market Selection")
                
                # Get unique commodities for selection
                unique_commodities = sorted(df['Commodity'].unique())
                selected_commodity = st.selectbox(
                    "Select Commodity",
                    unique_commodities
                )
                
                # Filter markets based on selected commodity
                commodity_markets = sorted(df[df['Commodity'] == selected_commodity]['Market'].unique())
                st.write(f"Total available markets for {selected_commodity}: {len(commodity_markets)}")
                
                st.markdown("Select up to 3 preferred markets for analysis. If less than 3 are selected, the remaining will be filled based on data completeness.")
                
                # Load selected markets from session state or create new selections
                selected_markets = []
                for i in range(3):
                    default_index = 0
                    if i < len(st.session_state.selected_markets):
                        try:
                            market_index = commodity_markets.index(st.session_state.selected_markets[i])
                            default_index = market_index + 1  # Add 1 because "None" is at index 0
                        except ValueError:
                            default_index = 0
                    
                    market = st.selectbox(
                        f"Market {i+1}",
                        ["None"] + commodity_markets,
                        index=default_index,
                        key=f"market_{i}"
                    )
                    if market != "None":
                        selected_markets.append(market)
                
                process_markets = st.button("🏪 Process Markets", use_container_width=True)
                
                if process_markets or st.session_state.markets_processed:
                    with st.spinner("Processing markets and creating CVGs..."):
                        # Store selected markets in session state
                        st.session_state.selected_markets = selected_markets
                        
                        # Create CVG identifier
                        df['CVG'] = df.apply(create_cvg_identifier, axis=1)
                        
                        # Filter data for selected commodity
                        commodity_df = df[df['Commodity'] == selected_commodity]
                        
                        # Select best CVGs based on preferred markets for the selected commodity
                        if not st.session_state.best_cvgs:
                            st.session_state.best_cvgs = select_best_cvgs(commodity_df, n=3, preferred_markets=selected_markets)
                        
                        st.session_state.markets_processed = True
                        
                        # Display selected CVGs with details
                        st.markdown("### Selected CVGs")
                        cvg_details = []
                        for cvg in st.session_state.best_cvgs:
                            cvg_data = commodity_df[commodity_df['CVG'] == cvg]
                            details = {
                                'CVG': cvg,
                                'Start Date': cvg_data['Date'].min().strftime('%Y-%m-%d'),
                                'End Date': cvg_data['Date'].max().strftime('%Y-%m-%d'),
                                'Total Records': len(cvg_data),
                                'Market': cvg.split('_')[-1]
                            }
                            cvg_details.append(details)
                        
                        st.table(pd.DataFrame(cvg_details))
                        
                        # Select CVG for analysis
                        default_cvg_index = 0
                        if st.session_state.selected_cvg in st.session_state.best_cvgs:
                            default_cvg_index = st.session_state.best_cvgs.index(st.session_state.selected_cvg)
                        
                        selected_cvg = st.selectbox(
                            "Select CVG for Analysis",
                            st.session_state.best_cvgs,
                            index=default_cvg_index
                        )
                        st.session_state.selected_cvg = selected_cvg
                        
                        if st.button("Reset Market Selection", use_container_width=True):
                            st.session_state.markets_processed = False
                            st.session_state.selected_markets = []
                            st.session_state.best_cvgs = None
                            st.session_state.selected_cvg = None
                            st.session_state.min_price_threshold = None
                            st.experimental_rerun()
                        
                        # Add Start Analysis button
                        st.markdown("## Step 3: Analysis")
                        
                        # Add minimum price threshold selection before analysis starts
                        st.markdown("### Set Minimum Price Threshold")
                        st.markdown("Set the minimum acceptable price for detecting unrealistic values.")
                        
                        # Get current minimum price for reference
                        cvg_data = commodity_df[commodity_df['CVG'] == selected_cvg].copy()
                        current_min = cvg_data['Modal'].min()
                        suggested_min = max(50, round(current_min * 0.5))  # Suggest 50% of current minimum or 50, whichever is higher
                        
                        # Use session state for min_price_threshold
                        if st.session_state.min_price_threshold is None:
                            st.session_state.min_price_threshold = suggested_min
                        
                        min_price_threshold = st.number_input(
                            "Minimum Price Threshold (Rs./Quintal)",
                            min_value=1,
                            max_value=1000,
                            value=st.session_state.min_price_threshold,
                            help="Values below this threshold will be considered unrealistic and removed. Default is suggested based on the data.",
                            key="min_price_input"
                        )
                        
                        # Update session state with new threshold
                        st.session_state.min_price_threshold = min_price_threshold
                        
                        start_analysis = st.button("🚀 Start Analysis", use_container_width=True)
                        
                        if start_analysis:
                            with st.spinner("Performing analysis..."):
                                # Validate and clean price data with user-defined threshold
                                cvg_data, validation_stats = validate_prices(cvg_data, min_price_threshold=st.session_state.min_price_threshold)
                                
                                # Display validation results
                                st.markdown("### Data Validation Results")
                                st.markdown(f"""
                                - Original records: {validation_stats['original_records']:,}
                                - Unrealistic records removed: {validation_stats['unrealistic_records']:,} ({validation_stats['removed_percentage']:.2f}%)
                                - Remaining records: {validation_stats['remaining_records']:,}
                                """)
                                
                                if validation_stats['unrealistic_records'] > 0:
                                    st.warning(f"⚠️ Removed {validation_stats['unrealistic_records']} records with prices below {min_price_threshold} Rs./Quintal")
                                
                                # Proceed only if we have enough data after validation
                                if len(cvg_data) < 100:
                                    st.error("❌ Insufficient data remaining after removing unrealistic values. Please adjust the minimum price threshold or select a different CVG.")
                                    return
                                
                                # First interpolate the daily data
                                cvg_data = interpolate_data(cvg_data)
                                
                                # Then create resampled versions
                                weekly_data = resample_data(cvg_data, 'W')
                                monthly_data = resample_data(cvg_data, 'ME')
                                
                                # Ensure no missing values in resampled data
                                weekly_data = interpolate_data(weekly_data)
                                monthly_data = interpolate_data(monthly_data)
                                
                                # Perform basic analysis
                                analysis_results = analyze_data(cvg_data)
                                
                                # Display analyses
                                st.header(f"Analysis Results for {selected_cvg}")
                                
                                # Data Overview Section
                                with st.container():
                                    st.markdown("### 📊 Data Overview")
                                    
                                    # Show date range separately
                                    start_date = cvg_data['Date'].min().strftime('%Y-%m-%d')
                                    end_date = cvg_data['Date'].max().strftime('%Y-%m-%d')
                                    st.markdown(f"**Time Period**: {start_date} to {end_date}")
                                    st.markdown(f"**Total Records**: {len(cvg_data):,}")
                                    
                                    # Create descriptive statistics table
                                    commodity_name = selected_cvg.split('_')[0]
                                    
                                    stats_dict = {
                                        'Descriptive statistics': [
                                            'Maximum (Rs./Quintal)',
                                            'Mean (Rs./Quintal)',
                                            'Minimum (Rs./Quintal)',
                                            'Standard deviation (Rs./Quintal)',
                                            'Coefficient of variation (%)',
                                            'Skewness',
                                            'Kurtosis'
                                        ],
                                        commodity_name: [
                                            f"{cvg_data['Modal'].max():.2f}",
                                            f"{cvg_data['Modal'].mean():.2f}",
                                            f"{cvg_data['Modal'].min():.2f}",
                                            f"{cvg_data['Modal'].std():.2f}",
                                            f"{(cvg_data['Modal'].std() / cvg_data['Modal'].mean() * 100):.2f}",
                                            f"{cvg_data['Modal'].skew():.2f}",
                                            f"{cvg_data['Modal'].kurtosis():.2f}"
                                        ]
                                    }
                                    
                                    stats_df = pd.DataFrame(stats_dict)
                                    stats_df = stats_df.set_index('Descriptive statistics')
                                    
                                    st.table(stats_df.style.set_properties(**{
                                        'text-align': 'center',
                                        'font-size': '1em'
                                    }))
                                
                                # Time Series Data Section
                                with st.container():
                                    st.markdown("### 📈 Time Series Data")
                                    
                                    # Create tabs for different frequencies
                                    daily_tab, weekly_tab, monthly_tab = st.tabs(["Daily Data", "Weekly Data", "Monthly Data"])
                                    
                                    # Daily Data Tab
                                    with daily_tab:
                                        st.markdown("#### Daily Price Data")
                                        
                                        # Display daily statistics
                                        daily_stats = get_descriptive_stats(cvg_data, "Daily")
                                        daily_stats = daily_stats.set_index('Descriptive statistics')
                                        st.table(daily_stats.style.set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '1em'
                                        }))
                                        
                                        # Display ADF test results
                                        st.markdown("#### Stationarity Test (Daily Data)")
                                        daily_adf = perform_adf_test(cvg_data, selected_cvg)
                                        st.table(daily_adf.style.set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '1em'
                                        }))
                                        
                                        # Display BDS test results
                                        st.markdown("#### Nonlinearity Test (Daily Data)")
                                        try:
                                            daily_bds = perform_bds_test(cvg_data, selected_cvg)
                                            st.markdown("BDS test results for different epsilon values (multiples of standard deviation σ):")
                                            st.table(daily_bds.style.set_properties(**{
                                                'text-align': 'center',
                                                'font-size': '1em'
                                            }))
                                        except Exception as e:
                                            st.error(f"Error in BDS test: {str(e)}")
                                        
                                        # Interactive line chart for daily data
                                        fig_daily = px.line(
                                            cvg_data, x='Date', y='Modal',
                                            title='Daily Price Data',
                                            labels={'Modal': 'Price (Rs./Quintal)', 'Date': 'Date'},
                                            markers=True
                                        )
                                        fig_daily.update_traces(connectgaps=False)
                                        fig_daily.update_layout(hovermode='x unified')
                                        st.plotly_chart(fig_daily, use_container_width=True)
                                        
                                        # Seasonal Decomposition for Daily Data
                                        st.markdown("#### Time Series Decomposition (Daily)")
                                        try:
                                            daily_decomposition = perform_seasonal_decomposition(cvg_data, 'D')
                                            fig_daily_decomp = plot_decomposition(
                                                daily_decomposition,
                                                'Daily Price Time Series Decomposition'
                                            )
                                            st.plotly_chart(fig_daily_decomp, use_container_width=True)
                                        except Exception as e:
                                            st.warning(f"Could not perform daily decomposition: {str(e)}")
                                        
                                        # Download link for daily data
                                        csv_df = cvg_data[['Date', 'Modal']].copy()
                                        csv_df['Date'] = csv_df['Date'].dt.strftime('%Y-%m-%d')
                                        csv = csv_df.to_csv(index=False)
                                        b64 = base64.b64encode(csv.encode()).decode()
                                        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="daily_price_data.csv">Download Daily Data (CSV)</a>', unsafe_allow_html=True)
                                    
                                    # Weekly Data Tab
                                    with weekly_tab:
                                        st.markdown("#### Weekly Price Data")
                                        
                                        # Display weekly statistics
                                        weekly_stats = get_descriptive_stats(weekly_data, "Weekly")
                                        weekly_stats = weekly_stats.set_index('Descriptive statistics')
                                        st.table(weekly_stats.style.set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '1em'
                                        }))
                                        
                                        # Display ADF test results
                                        st.markdown("#### Stationarity Test (Weekly Data)")
                                        weekly_adf = perform_adf_test(weekly_data, selected_cvg)
                                        st.table(weekly_adf.style.set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '1em'
                                        }))
                                        
                                        # Display BDS test results
                                        st.markdown("#### Nonlinearity Test (Weekly Data)")
                                        try:
                                            weekly_bds = perform_bds_test(weekly_data, selected_cvg)
                                            st.markdown("BDS test results for different epsilon values (multiples of standard deviation σ):")
                                            st.table(weekly_bds.style.set_properties(**{
                                                'text-align': 'center',
                                                'font-size': '1em'
                                            }))
                                        except Exception as e:
                                            st.error(f"Error in BDS test: {str(e)}")
                                        
                                        # Interactive line chart for weekly data
                                        fig_weekly = px.line(
                                            weekly_data, x='Date', y='Modal',
                                            title='Weekly Price Data',
                                            labels={'Modal': 'Price (Rs./Quintal)', 'Date': 'Date'},
                                            markers=True
                                        )
                                        fig_weekly.update_traces(connectgaps=False)
                                        fig_weekly.update_layout(hovermode='x unified')
                                        st.plotly_chart(fig_weekly, use_container_width=True)
                                        
                                        # Seasonal Decomposition for Weekly Data
                                        st.markdown("#### Time Series Decomposition (Weekly)")
                                        try:
                                            weekly_decomposition = perform_seasonal_decomposition(weekly_data, 'W')
                                            fig_weekly_decomp = plot_decomposition(
                                                weekly_decomposition,
                                                'Weekly Price Time Series Decomposition'
                                            )
                                            st.plotly_chart(fig_weekly_decomp, use_container_width=True)
                                        except Exception as e:
                                            st.warning(f"Could not perform weekly decomposition: {str(e)}")
                                        
                                        # Download link for weekly data
                                        weekly_csv = weekly_data.copy()
                                        weekly_csv['Date'] = weekly_csv['Date'].dt.strftime('%Y-%m-%d')
                                        csv = weekly_csv.to_csv(index=False)
                                        b64 = base64.b64encode(csv.encode()).decode()
                                        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="weekly_price_data.csv">Download Weekly Data (CSV)</a>', unsafe_allow_html=True)
                                    
                                    # Monthly Data Tab
                                    with monthly_tab:
                                        st.markdown("#### Monthly Price Data")
                                        
                                        # Display monthly statistics
                                        monthly_stats = get_descriptive_stats(monthly_data, "Monthly")
                                        monthly_stats = monthly_stats.set_index('Descriptive statistics')
                                        st.table(monthly_stats.style.set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '1em'
                                        }))
                                        
                                        # Display ADF test results
                                        st.markdown("#### Stationarity Test (Monthly Data)")
                                        monthly_adf = perform_adf_test(monthly_data, selected_cvg)
                                        st.table(monthly_adf.style.set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '1em'
                                        }))
                                        
                                        # Display BDS test results
                                        st.markdown("#### Nonlinearity Test (Monthly Data)")
                                        try:
                                            monthly_bds = perform_bds_test(monthly_data, selected_cvg)
                                            st.markdown("BDS test results for different epsilon values (multiples of standard deviation σ):")
                                            st.table(monthly_bds.style.set_properties(**{
                                                'text-align': 'center',
                                                'font-size': '1em'
                                            }))
                                        except Exception as e:
                                            st.error(f"Error in BDS test: {str(e)}")
                                        
                                        # Interactive line chart for monthly data
                                        fig_monthly = px.line(
                                            monthly_data, x='Date', y='Modal',
                                            title='Monthly Price Data',
                                            labels={'Modal': 'Price (Rs./Quintal)', 'Date': 'Date'},
                                            markers=True
                                        )
                                        fig_monthly.update_traces(connectgaps=False)
                                        fig_monthly.update_layout(hovermode='x unified')
                                        st.plotly_chart(fig_monthly, use_container_width=True)
                                        
                                        # Seasonal Decomposition for Monthly Data
                                        st.markdown("#### Time Series Decomposition (Monthly)")
                                        try:
                                            monthly_decomposition = perform_seasonal_decomposition(monthly_data, 'M')
                                            fig_monthly_decomp = plot_decomposition(
                                                monthly_decomposition,
                                                'Monthly Price Time Series Decomposition'
                                            )
                                            st.plotly_chart(fig_monthly_decomp, use_container_width=True)
                                        except Exception as e:
                                            st.warning(f"Could not perform monthly decomposition: {str(e)}")
                                        
                                        # Download link for monthly data
                                        monthly_csv = monthly_data.copy()
                                        monthly_csv['Date'] = monthly_csv['Date'].dt.strftime('%Y-%m-%d')
                                        csv = monthly_csv.to_csv(index=False)
                                        b64 = base64.b64encode(csv.encode()).decode()
                                        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="monthly_price_data.csv">Download Monthly Data (CSV)</a>', unsafe_allow_html=True)

                                # Generate and download report
                                report_text = generate_report(cvg_data, selected_cvg, analysis_results)
                                st.markdown(get_download_link(report_text, f"{selected_cvg}_analysis_report.txt"), unsafe_allow_html=True)
                            
                            st.success("Analysis completed successfully!")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please check your data format and try again.")
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main() 
