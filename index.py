from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import json
import os
import pandas as pd
import plotly.graph_objects as go

from financial_forecasting import finance_forecasting
from fraud_detection import fraud_detection_analysis
from tax_compliance import calculate_tax_liability
from invoice_processing import process_invoices

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Financial Analysis Suite API is running!"})


@app.route('/api/forecast', methods=['POST'])
def forecast_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes_io = io.BytesIO(file.read())
        target_col = request.form.get('target_column', 'target_sales')
        date_col = request.form.get('date_column', 'Date')
        forecast_months = int(request.form.get('forecast_months', 12))
        contamination = float(request.form.get('contamination', 0.01))

        df_anomalies, forecast_df, plotly_forecast_fig, plot_images = finance_forecasting(
            file_bytes_io,
            contamination=contamination,
            forecast_months=forecast_months,
            target_col=target_col,
            date_col=date_col
        )

        response_data = {
            "anomalies_data": df_anomalies.to_json(orient='split', date_format='iso'),
            "forecast_data": forecast_df.to_json(orient='split', date_format='iso'),
            "main_forecast_plot_json": plotly_forecast_fig.to_json(),
            "additional_plots": {}
        }

        for k, v in plot_images.items():
            if isinstance(v, str):
                response_data["additional_plots"][k] = v
            elif isinstance(v, go.Figure):
                response_data["additional_plots"][k] = v.to_json()
            elif isinstance(v, dict) and all(isinstance(val, go.Figure) for val in v.values()):
                response_data["additional_plots"][k] = {inner_k: inner_v.to_json() for inner_k, inner_v in v.items()}

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error in /api/forecast: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/fraud', methods=['POST'])
def fraud_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes_io = io.BytesIO(file.read())
        contamination = float(request.form.get('contamination', 0.01))
        date_col_name = request.form.get('date_column_name', 'TransactionDate')

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
            "plot_images": {k: v for k, v in plot_images.items()}
        }
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /api/fraud: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/tax_calculate', methods=['POST'])
def tax_calculate_endpoint():
    try:
        data = request.get_json()
        income = float(data.get('income'))
        deductions = float(data.get('deductions'))
        year = int(data.get('year'))

        result = calculate_tax_liability(income, deductions, year)
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in /api/tax_calculate: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route('/api/invoice_process', methods=['POST'])
def invoice_process_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes_io = io.BytesIO(file.read())

        df_original, top_segments_df, city_revenue_fig, revenue_trend_fig, \
        suspicious_invoices_df, extracted_entities_df, actual_vs_budget_df, audit_flags_df = process_invoices(file_bytes_io)

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
