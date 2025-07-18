# financial-analysis-suite-web/backend/api/index.py
# (Or backend/api/__init__.py)

from flask import Flask, request, jsonify
from flask_cors import CORS # Important for allowing your React frontend to talk to your Flask backend
import io
import json
import os
import pandas as pd
import plotly.graph_objects as go # For Plotly figure type checking

# Import your adapted core logic functions
# These imports assume that financial_forecasting.py, fraud_detection.py,
# tax_compliance.py, and invoice_processing.py are in the SAME directory
# (backend/api/) as this index.py file.
from .financial_forecasting import finance_forecasting
from .fraud_detection import fraud_detection_analysis
from .tax_compliance import calculate_tax_liability
from .invoice_processing import process_invoices

app = Flask(__name__)
CORS(app) # Enable CORS for all routes - necessary for React frontend to access API

# Basic route for testing if the API is alive
@app.route('/', methods=['GET'])
def home():
    """Returns a simple message indicating the API is running."""
    return jsonify({"message": "Financial Analysis Suite API is running!"})

# --- Financial Forecasting Endpoint ---
@app.route('/api/forecast', methods=['POST'])
def forecast_endpoint():
    """
    Handles financial forecasting requests. Expects a CSV file and parameters.
    Returns forecasted data, anomalies, and plot data.
    """
    # File upload via FormData from React frontend
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Pass file content as BytesIO object to your core function
        file_bytes_io = io.BytesIO(file.read())

        # Extract other parameters from form data (e.g., from a FormData object in JS)
        target_col = request.form.get('target_column', 'target_sales')
        date_col = request.form.get('date_column', 'Date')
        forecast_months = int(request.form.get('forecast_months', 12))
        contamination = float(request.form.get('contamination', 0.01))

        # Call your core logic (already adapted not to use Streamlit's st_object)
        df_anomalies, forecast_df, plotly_forecast_fig, plot_images = finance_forecasting(
            file_bytes_io,
            contamination=contamination,
            forecast_months=forecast_months,
            target_col=target_col,
            date_col=date_col
        )

        # Prepare results for JSON response
        # DataFrames to JSON (orient='split' is good for re-creating in JavaScript)
        # Plotly figures to JSON (Plotly.js can render this directly in the frontend)
        # Matplotlib base64 strings are already strings
        
        response_data = {
            "anomalies_data": df_anomalies.to_json(orient='split', date_format='iso'),
            "forecast_data": forecast_df.to_json(orient='split', date_format='iso'),
            "main_forecast_plot_json": plotly_forecast_fig.to_json(),
            "additional_plots": {}
        }
        
        # Process the 'plot_images' dictionary returned by finance_forecasting
        for k, v in plot_images.items():
            if isinstance(v, str): # This would be a Matplotlib base64 string
                response_data["additional_plots"][k] = v
            elif isinstance(v, go.Figure): # This would be a Plotly figure object
                response_data["additional_plots"][k] = v.to_json()
            elif isinstance(v, dict) and all(isinstance(val, go.Figure) for val in v.values()):
                # Handle the specific case of 'market_indicators_plotly_figs' which is a dict of Plotly figures
                response_data["additional_plots"][k] = {inner_k: inner_v.to_json() for inner_k, inner_v in v.items()}
            # Add handling for other potential return types if necessary
            # else:
            #     response_data["additional_plots"][k] = str(v) # Fallback for unexpected types

        return jsonify(response_data)

    except Exception as e:
        # It's good practice to log the full traceback for debugging in production environments
        # On Vercel, these logs would appear in your Vercel dashboard for the function.
        app.logger.error(f"Error in /api/forecast: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500 # Return 500 for internal server error

# --- Fraud Detection Endpoint ---
@app.route('/api/fraud', methods=['POST'])
def fraud_endpoint():
    """
    Handles fraud detection requests. Expects a CSV file and parameters.
    Returns fraud analysis results and plot data.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes_io = io.BytesIO(file.read())
        contamination = float(request.form.get('contamination', 0.01))
        date_col_name = request.form.get('date_column_name', 'TransactionDate')
        
        # Call your core logic (already adapted)
        df_full, anomalies_df, anomaly_summary_list, top_anom_df, amount_col_name, plot_images = fraud_detection_analysis(
            file_bytes_io,
            contamination=contamination,
            date_col_name=date_col_name
        )

        response_data = {
            "full_data_json": df_full.to_json(orient='split', date_format='iso'),
            "anomalies_data_json": anomalies_df.to_json(orient='split', date_format='iso'),
            "anomaly_summary": anomaly_summary_list,
            "top_anomalies_data_json": top_anom_df.to_json(orient='split', date_format='iso') if top_anom_df is not None else None,
            "amount_col_name": amount_col_name,
            "plot_images": {k: v for k,v in plot_images.items()} # These are already base64 strings from fraud_detection.py
        }
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /api/fraud: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Tax Compliance Endpoint ---
@app.route('/api/tax_calculate', methods=['POST'])
def tax_calculate_endpoint():
    """
    Calculates tax liability based on provided income, deductions, and year.
    Expects JSON body.
    """
    try:
        data = request.get_json() # Frontend sends JSON body for this endpoint
        income = float(data.get('income'))
        deductions = float(data.get('deductions'))
        year = int(data.get('year'))

        result = calculate_tax_liability(income, deductions, year)
        return jsonify(result) # Result is already a dictionary, so directly jsonify

    except Exception as e:
        app.logger.error(f"Error in /api/tax_calculate: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400 # Return 400 for bad request

# --- Invoice Processing Endpoint ---
@app.route('/api/invoice_process', methods=['POST'])
def invoice_process_endpoint():
    """
    Handles invoice processing requests. Expects a CSV file and returns various analysis results.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes_io = io.BytesIO(file.read())
        
        # Call your core logic (already adapted)
        df_original, top_segments_df, city_revenue_fig, revenue_trend_fig, \
        suspicious_invoices_df, extracted_entities_df, actual_vs_budget_df, audit_flags_df = process_invoices(file_bytes_io)

        # Prepare results for JSON response
        response_data = {
            "summary": {
                "total_invoices": len(df_original),
                "total_revenue": df_original['total_value'].sum()
            },
            "top_segments_json": top_segments_df.to_json(orient='split'),
            "city_revenue_fig_json": city_revenue_fig.to_json(),
            "revenue_trend_fig_json": revenue_trend_fig.to_json(),
            "suspicious_invoices_json": suspicious_invoices_df.to_json(orient='split'),
            "extracted_entities_json": extracted_entities_df.to_json(orient='split'),
            "actual_vs_budget_json": actual_vs_budget_df.to_json(orient='split'),
            "audit_flags_json": audit_flags_df.to_json(orient='split')
        }
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /api/invoice_process: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Main entry point for Vercel ---
# Vercel will look for an 'app' object in this file if 'src' in vercel.json points here.
# This makes 'app' the WSGI application that Vercel will serve.