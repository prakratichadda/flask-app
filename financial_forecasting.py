# financial-analysis-suite-web/backend/api/financial_forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore
import datetime
import io
import base64
from tensorflow.keras.models import Sequential # Using tensorflow.keras
from tensorflow.keras.layers import LSTM, Dense # Using tensorflow.keras

def finance_forecasting(filepath_or_bytes_obj: any, contamination: float = 0.01, forecast_months: int = 12, 
                        target_col: str = 'target_sales', date_col: str = 'Date'):
    """
    Main function to perform financial forecasting, anomaly detection, and visualization.

    Args:
        filepath_or_bytes_obj (str or io.BytesIO): Path to the CSV data file or a BytesIO object of the file.
        contamination (float): The proportion of outliers in the data set for IsolationForest.
        forecast_months (int): Number of future periods (steps/months) to forecast.
        target_col (str): The name of the column to forecast.
        date_col (str): The name of the date column in the CSV. If not found or invalid,
                        a numerical time step index will be used.
    
    Returns:
        tuple: (df_anomalies, forecast_df, plotly_forecast_fig, plot_images)
            - df_anomalies (pd.DataFrame): DataFrame with detected anomalies.
            - forecast_df (pd.DataFrame): DataFrame with forecasted values.
            - plotly_forecast_fig (go.Figure): Plotly figure object for interactive forecast visualization.
            - plot_images (dict): Dictionary of base64 encoded Matplotlib plot images and Plotly figure objects.
    """
    plot_images = {}

    # ------------------ Data Loading and Preparation ------------------ #
    def load_and_prepare_data(fp: any, dc: str) -> pd.DataFrame: 
        """
        Loads CSV data, cleans it, handles missing values, duplicates, and sets up time index.
        Can load from a file path string or an io.BytesIO object.
        """
        if isinstance(fp, io.BytesIO):
            df = pd.read_csv(fp)
        else:
            df = pd.read_csv(fp)
            
        df.columns = df.columns.str.strip()

        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors='coerce')
            df = df.dropna(subset=[dc])
            if not df.empty:
                df = df.sort_values(dc)
                df = df.set_index(dc)
            else:
                df['time_step'] = range(len(df))
                df = df.set_index('time_step')
        else:
            df['time_step'] = range(len(df))
            df = df.set_index('time_step')
        
        if df.empty:
            raise ValueError("No valid data rows found after date processing.")

        # Update: Use obj.ffill() or obj.bfill() directly to avoid FutureWarning
        df = df.ffill().bfill().fillna(0) # Chain ffill, bfill, then 0 for any remaining NaNs

        df.drop_duplicates(inplace=True)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the uploaded CSV.")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].mean() if pd.api.types.is_numeric_dtype(df[target_col]) else 0)

        return df

    # ------------------ Anomaly Detection ------------------ #
    def detect_anomalies(df: pd.DataFrame, col='target_sales', contam=0.01) -> pd.DataFrame:
        """
        Detects anomalies in the specified target column (and potentially other numeric features)
        using IsolationForest.
        """
        numeric_cols_for_anomaly = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols_for_anomaly:
            df_copy = df.copy()
            df_copy['is_anomaly'] = False # Default to no anomalies if no numeric data for IF
            return df_copy # Return copy to avoid SettingWithCopyWarning
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols_for_anomaly])

        model = IsolationForest(contamination=contam, random_state=42)
        df_copy = df.copy()
        df_copy['anomaly'] = model.fit_predict(X_scaled)
        df_copy['is_anomaly'] = df_copy['anomaly'] == -1
        return df_copy

    # ------------------ Forecasting ------------------ #
    def forecast_target(df: pd.DataFrame, col='target_sales', f_months=12) -> pd.DataFrame:
        """
        Forecasts future values of the target column using an LSTM model.
        Handles both DatetimeIndex and numerical index for future periods.
        """
        data_to_scale = df[[col]].dropna()

        if data_to_scale.empty:
            raise ValueError(f"No valid data in target column '{col}' for forecasting after dropping NaNs. Cannot perform forecast.")
            
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_to_scale)

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        SEQ_LEN = 12 # Sequence length for LSTM
        # Ensure enough data for sequences AND for the input_seq for initial prediction
        if len(scaled_data) < SEQ_LEN + 1: # Need SEQ_LEN + 1 points to create at least one sequence (X[0], y[0])
            raise ValueError(f"Not enough data to create sequences for forecasting. Need at least {SEQ_LEN + 1} data points for LSTM (current: {len(scaled_data)}).")
        
        # This is where the sequences are created and reshaped
        X, y = create_sequences(scaled_data, SEQ_LEN)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and compile LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(SEQ_LEN, 1)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        try:
            model.fit(X, y, epochs=30, batch_size=16, verbose=0)
        except Exception as e:
            raise RuntimeError(f"Error during LSTM model training: {e}.")

        # Forecast future values
        input_seq = scaled_data[-SEQ_LEN:] # Start with the last SEQ_LEN data points
        # Ensure input_seq has enough elements, even if create_sequences passed due to just enough data
        if len(input_seq) < SEQ_LEN:
             raise ValueError(f"Insufficient data for initial forecast sequence. Need at least {SEQ_LEN} points for input_seq (current: {len(input_seq)}).")

        forecast = []
        for _ in range(f_months):
            input_reshaped = input_seq.reshape((1, SEQ_LEN, 1))
            pred = model.predict(input_reshaped, verbose=0)
            forecast.append(pred[0, 0])
            input_seq = np.append(input_seq[1:], pred)

        # Inverse transform the forecast to original scale
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        # Create future index based on original df's index type
        if isinstance(df.index, pd.DatetimeIndex):
            last_date_in_df = df.index[-1]
            future_index = pd.date_range(start=last_date_in_df + pd.DateOffset(months=1), periods=f_months, freq='MS') # Changed to MS for Monthly Start
        else: # Numerical time step index
            last_time_step_in_df = df.index[-1]
            future_index = range(last_time_step_in_df + 1, last_time_step_in_df + 1 + f_months)
        
        forecast_df = pd.DataFrame(forecast, index=future_index, columns=[f'Forecast_{col}'])
        return forecast_df

    # ------------------ Plotting Functions ------------------ #
    def get_base64_image(plt_figure):
        """Converts a Matplotlib figure to a base64 encoded PNG string. Handles None input."""
        if plt_figure is None:
            return None # Return None if no figure provided
        buf = io.BytesIO()
        try:
            plt_figure.savefig(buf, format='png', bbox_inches='tight')
        except Exception as e:
            # Log this error but don't crash
            print(f"Warning: Failed to save Matplotlib figure to buffer: {e}")
            return None
        finally:
            plt.close(plt_figure) # Always close the figure
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    def plot_numeric_trends(df_input, numeric_cols_to_plot):
        if df_input.empty or not numeric_cols_to_plot:
            return None
        
        fig, ax = plt.subplots(figsize=(15, 8))
        df_input[numeric_cols_to_plot].plot(ax=ax)
        ax.set_title("Numeric Trends After Cleaning")
        ax.set_xlabel("Date" if isinstance(df_input.index, pd.DatetimeIndex) else "Time Step")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        return fig

    def plot_sales_vs_target_sales(df_input, sales_col, target_sales_col):
        if sales_col not in df_input.columns or target_sales_col not in df_input.columns:
            return go.Figure() # Return empty Plotly figure if columns are missing
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_input.index, y=df_input[sales_col], mode='lines+markers', name='Sales',
            line=dict(color='teal', width=2), marker=dict(symbol="circle", size=4)
        ))
        fig.add_trace(go.Scatter(
            x=df_input.index, y=df_input[target_sales_col], mode='lines+markers', name='Target Sales',
            line=dict(color='orange', width=2, dash='dot'), marker=dict(symbol="star", size=4)
        ))
        fig.update_layout(
            title="Sales vs Target Sales (Interactive)",
            xaxis_title="Date" if isinstance(df_input.index, pd.DatetimeIndex) else "Time Step",
            yaxis_title="Amount",
            hovermode="x unified",
            plot_bgcolor='white',
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig

    def plot_market_indicators_treemap(df_input):
        market_indicators = [col for col in ['gdp_growth', 'unemployment_rate', 'inflation_rate'] if col in df_input.columns]
        
        if not market_indicators:
            return {} # Return empty dict if no indicators to plot
        
        indicator_figs = {}
        for indicator in market_indicators:
            fig = px.line(df_input, x=df_input.index, y=indicator, title=f"{indicator.replace('_', ' ').title()} Over Time")
            fig.update_layout(
                xaxis_title="Date" if isinstance(df_input.index, pd.DatetimeIndex) else "Time Step",
                yaxis_title="Value",
                hovermode='x unified',
                plot_bgcolor='whitesmoke',
                margin=dict(l=40, r=40, t=80, b=40)
            )
            indicator_figs[indicator] = fig
        return indicator_figs
        
    def plot_correlation_heatmap(df_input, features_used_for_corr):
        numeric_df = df_input[features_used_for_corr].select_dtypes(include=np.number)

        if numeric_df.empty:
            return go.Figure() # Return empty Plotly figure
        
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Correlation Heatmap of Financial Indicators"
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        return fig


    # Main execution logic
    try:
        df_cleaned = load_and_prepare_data(filepath_or_bytes_obj, date_col)
        numeric_cols_for_general_plots = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        if 'sales' in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned['sales']):
            if 'sales' not in numeric_cols_for_general_plots:
                numeric_cols_for_general_plots.append('sales')
        if target_col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[target_col]):
            if target_col not in numeric_cols_for_general_plots:
                numeric_cols_for_general_plots.append(target_col)
        
        numeric_cols_for_general_plots = [col for col in numeric_cols_for_general_plots if col not in ['anomaly', 'is_anomaly']]

    except ValueError as e:
        raise ValueError(f"Data loading and preparation failed: {e}")

    df_anomalies = None
    try:
        df_anomalies = detect_anomalies(df_cleaned, col=target_col, contam=contamination)
    except Exception as e:
        # If anomaly detection fails, proceed with original df but no anomaly flags
        print(f"Warning: Anomaly detection for financial forecasting failed: {e}. Proceeding with forecasting without anomaly flags.")
        df_anomalies = df_cleaned.copy()
        df_anomalies['is_anomaly'] = False


    forecast_df = pd.DataFrame()
    plotly_forecast_fig = go.Figure()

    try:
        forecast_df = forecast_target(df_anomalies, col=target_col, f_months=forecast_months)

        plotly_forecast_fig = go.Figure()
        plotly_forecast_fig.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies[target_col], 
                                                name='Historical Sales', mode='lines+markers', line=dict(color='blue')))
        if 'is_anomaly' in df_anomalies.columns and df_anomalies['is_anomaly'].any():
            anomalies_to_plot = df_anomalies[df_anomalies['is_anomaly']]
            plotly_forecast_fig.add_trace(go.Scatter(x=anomalies_to_plot.index, y=anomalies_to_plot[target_col], 
                                                     mode='markers', name='Anomalies', 
                                                     marker=dict(color='red', size=8, symbol='x')))

        plotly_forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[forecast_df.columns[0]], 
                                                name='Forecasted Sales', mode='lines+markers', 
                                                line=dict(color='orange', dash='dash')))

        plotly_forecast_fig.update_layout(
            title=f'Historical and Forecasted {target_col} with Anomalies',
            xaxis_title="Date" if isinstance(df_anomalies.index, pd.DatetimeIndex) else "Time Step",
            yaxis_title=target_col,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )

    except (ValueError, RuntimeError) as e:
        # --- CRITICAL CHANGE HERE: Re-raise the error ---
        # This makes the Flask API return an error, which the React frontend can then display clearly.
        # This prevents silently returning an empty forecast_df
        raise ValueError(f"Forecasting calculation failed: {e}. Please check your data and parameters (e.g., ensure enough historical data).")


    # --- Generate Additional Plots ---
    # Wrap Matplotlib plot generation in try-except to catch GUI errors
    try:
        fig_numeric_trends = plot_numeric_trends(df_cleaned, numeric_cols_for_general_plots)
        if fig_numeric_trends: plot_images['numeric_trends'] = get_base64_image(fig_numeric_trends)
    except Exception as e:
        print(f"Warning: Failed to generate numeric trends plot (Matplotlib): {e}")
        plot_images['numeric_trends'] = None # Ensure it's explicitly None if it fails


    if 'sales' in df_cleaned.columns and target_col in df_cleaned.columns:
        plotly_sales_chart = plot_sales_vs_target_sales(df_cleaned, 'sales', target_col)
        plot_images['sales_vs_target_sales_plotly'] = plotly_sales_chart
    else:
        plot_images['sales_vs_target_sales_plotly'] = go.Figure() # Ensure it's an empty figure if data is missing
    
    plotly_market_indicator_figs = plot_market_indicators_treemap(df_cleaned)
    if plotly_market_indicator_figs: plot_images['market_indicators_plotly_figs'] = plotly_market_indicator_figs

    all_numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if 'anomaly' in all_numeric_cols: all_numeric_cols.remove('anomaly')
    if 'is_anomaly' in all_numeric_cols: all_numeric_cols.remove('is_anomaly')
    plotly_correlation_heatmap_fig = plot_correlation_heatmap(df_cleaned, all_numeric_cols)
    if plotly_correlation_heatmap_fig: plot_images['correlation_heatmap_plotly'] = plotly_correlation_heatmap_fig


    return df_anomalies, forecast_df, plotly_forecast_fig, plot_images