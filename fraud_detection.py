# financial-analysis-suite-web/backend/api/fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Removed st_object from the main function definition
def fraud_detection_analysis(file_path_or_bytes_obj: any, contamination: float = 0.01, date_col_name: str = 'TransactionDate'):
    """
    Main function for fraud detection analysis.

    Args:
        file_path_or_bytes_obj (any): Path to the CSV data file or an io.BytesIO object of the file.
        contamination (float): The proportion of outliers in the data set for IsolationForest.
        date_col_name (str): The name of the primary date column for time-based features (e.g., 'TransactionDate').

    Returns:
        tuple: (df, anomalies_df, anomaly_summary, top_anom_df, amount_col_name, plot_base64_images)
            - df (pd.DataFrame): Original DataFrame with anomaly flags.
            - anomalies_df (pd.DataFrame): DataFrame containing only detected anomalies.
            - anomaly_summary (list): List of summary strings for anomalies.
            - top_anom_df (pd.DataFrame or None): Top anomalies by value.
            - amount_col_name (str or None): The name of the identified amount column.
            - plot_base64_images (dict): Dictionary of base64 encoded plot images.
    """

    # Adapted fp to accept BytesIO object
    def load_data(fp: any):
        """Loads CSV data from a file path or io.BytesIO object."""
        if isinstance(fp, io.BytesIO):
            df = pd.read_csv(fp)
        else: # Assume it's a string path if not BytesIO
            df = pd.read_csv(fp)
        df.columns = df.columns.str.strip() # Clean column names
        return df

    def feature_engineer_fraud_data(df, primary_date_col): # Removed st_object from here
        """
        Engineers new features relevant for fraud detection from raw transaction data.
        """
        df_copy = df.copy()

        for col in [primary_date_col, 'PreviousTransactionDate']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

        if primary_date_col in df_copy.columns and 'PreviousTransactionDate' in df_copy.columns:
            df_copy['TimeSinceLastTransaction'] = (df_copy[primary_date_col] - df_copy['PreviousTransactionDate']).dt.total_seconds()
            df_copy['TimeSinceLastTransaction'].fillna(-1, inplace=True)
        else:
            df_copy['TimeSinceLastTransaction'] = -1

        if primary_date_col in df_copy.columns:
            df_copy['TransactionHour'] = df_copy[primary_date_col].dt.hour.fillna(-1)
            df_copy['TransactionWeekday'] = df_copy[primary_date_col].dt.weekday.fillna(-1)
        else:
            df_copy['TransactionHour'] = -1
            df_copy['TransactionWeekday'] = -1

        df_copy['IsNightTransaction'] = df_copy['TransactionHour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

        if 'TransactionAmount' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['TransactionAmount']):
            amount_threshold = df_copy['TransactionAmount'].quantile(0.95)
            df_copy['HighTransactionAmount'] = (df_copy['TransactionAmount'] > amount_threshold).astype(int)
        else:
            df_copy['HighTransactionAmount'] = 0

        if 'LoginAttempts' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['LoginAttempts']):
            df_copy['HighLoginAttempts'] = (df_copy['LoginAttempts'] > 3).astype(int)
        else:
            df_copy['HighLoginAttempts'] = 0

        if 'AccountBalance' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['AccountBalance']):
            df_copy['LowAccountBalance'] = (df_copy['AccountBalance'] < 100).astype(int)
        else:
            df_copy['LowAccountBalance'] = 0

        if 'TransactionAmount' in df_copy.columns and 'AccountBalance' in df_copy.columns \
           and pd.api.types.is_numeric_dtype(df_copy['TransactionAmount']) and pd.api.types.is_numeric_dtype(df_copy['AccountBalance']):
            df_copy['TransactionAmountToBalanceRatio'] = df_copy['TransactionAmount'] / (df_copy['AccountBalance'].replace(0, np.nan) + 1).fillna(1)
        else:
            df_copy['TransactionAmountToBalanceRatio'] = 0

        categorical_cols_to_encode = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
        encoded_cols_names = []
        for col in categorical_cols_to_encode:
            if col in df_copy.columns:
                try:
                    df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))
                    encoded_cols_names.append(col)
                except Exception as e:
                    pass # Removed st_object.warning

        base_features = [
            'TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance'
        ]
        
        engineered_features = [
            'TimeSinceLastTransaction', 'TransactionHour',
            'TransactionWeekday', 'IsNightTransaction', 'HighTransactionAmount',
            'HighLoginAttempts', 'LowAccountBalance', 'TransactionAmountToBalanceRatio'
        ]
        
        final_features = []
        for feature in base_features + engineered_features + encoded_cols_names:
            if feature in df_copy.columns:
                final_features.append(feature)

        for f_col in final_features:
            if df_copy[f_col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_copy[f_col]):
                    df_copy[f_col].fillna(df_copy[f_col].mean(), inplace=True)
                else:
                    df_copy[f_col].fillna(0, inplace=True)

        return df_copy, final_features

    def detect_anomalies(df_input, features_for_model, contam=0.01): # Removed st_object from here
        if not features_for_model:
            raise ValueError("No valid features available for anomaly detection. Please check your data columns.")

        missing_features = [f for f in features_for_model if f not in df_input.columns]
        if missing_features:
            raise ValueError(f"Missing features required for anomaly detection: {missing_features}. This should not happen if feature_engineer_fraud_data works correctly.")
            
        for col in features_for_model:
            if not pd.api.types.is_numeric_dtype(df_input[col]):
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)
                # Removed st_object.warning

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_input[features_for_model])

        model = IsolationForest(n_estimators=100, contamination=contam, random_state=42)
        
        df_input.loc[:, 'anomaly'] = model.fit_predict(X_scaled)
        df_input.loc[:, 'is_anomaly'] = df_input['anomaly'].apply(lambda x: 1 if x == -1 else 0)

        anomalies_df = df_input[df_input['is_anomaly'] == 1].copy()
        
        return df_input, anomalies_df, features_for_model


    def summarize_anomalies(anomalies_df, dc_name): # Removed st_object from here
        summary = []
        if dc_name in anomalies_df.columns and pd.api.types.is_datetime64_any_dtype(anomalies_df[dc_name]) and not anomalies_df[dc_name].isnull().all():
            anomalies_df_copy = anomalies_df.copy()
            anomalies_df_copy["Year"] = anomalies_df_copy[dc_name].dt.year
            anomalies_df_copy["Month"] = anomalies_df_copy[dc_name].dt.month_name()
            grouped = anomalies_df_copy.groupby(["Year", "Month"]).size().reset_index(name="Count")

            if not grouped.empty:
                summary.append("Anomalies by Month:")
                for _, row in grouped.iterrows():
                    summary.append(f"- {int(row['Count'])} anomalies in {row['Month']} {int(row['Year'])}")
            else:
                summary.append("No anomalies detected by month.")
        else:
            summary.append(f"Total anomalies detected: {len(anomalies_df)}")
            # Removed st_object.warning

        return summary


    def top_anomalies(anomalies_df, value_col_hint=['TransactionAmount', 'amount', 'value', 'transaction', 'balance', 'sales']): # Removed st_object from here
        amount_col = None
        df_cols_lower = {col.lower(): col for col in anomalies_df.columns}

        for hint in value_col_hint:
            hint_lower = hint.lower()
            if hint_lower in df_cols_lower:
                amount_col = df_cols_lower[hint_lower]
                break
            for col_lower, original_col in df_cols_lower.items():
                if hint_lower in col_lower:
                    amount_col = original_col
                    break
            if amount_col:
                break

        if amount_col and pd.api.types.is_numeric_dtype(anomalies_df[amount_col]) and not anomalies_df.empty:
            return anomalies_df.sort_values(by=amount_col, ascending=False).head(5), amount_col
        
        # Removed st_object.warning
        return None, None

    # --- Plotting Functions ---
    def get_base64_image(plt_figure):
        """Converts a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        plt_figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(plt_figure)
        return img_base64

    def plot_anomaly_count(df):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='is_anomaly', data=df, palette=['green', 'red'], ax=ax)
        ax.set_title("Fraud vs Non-Fraud Predictions")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Fraud', 'Fraud'])
        ax.set_ylabel("Number of Transactions")
        return fig

    def plot_fraud_by_transaction_type(df):
        if 'TransactionType' not in df.columns:
            # Removed st_object.warning
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fraud_by_type = df.groupby(['TransactionType', 'is_anomaly']).size().unstack(fill_value=0)
        fraud_by_type.plot(kind='bar', stacked=True, color=['green', 'red'], ax=ax)
        ax.set_title("Fraud by Transaction Type")
        ax.set_ylabel("Number of Transactions")
        ax.set_xlabel("Transaction Type (Encoded)")
        ax.legend(["Not Fraud", "Fraud"])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def plot_fraud_over_time(df, date_col_name):
        if date_col_name not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col_name]) or df[date_col_name].isnull().all():
            # Removed st_object.warning
            return None
        
        fig, ax = plt.subplots(figsize=(12, 5))
        fraud_daily = df[df['is_anomaly'] == 1].groupby(df[date_col_name].dt.date).size()
        if not fraud_daily.empty:
            fraud_daily.plot(kind='line', marker='o', color='red', ax=ax)
            ax.set_title("Fraud Predictions Over Time")
            ax.set_ylabel("Fraudulent Transactions")
            ax.set_xlabel("Date")
            ax.grid(True)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        else:
            # Removed st_object.info
            return None

    def plot_top_fraudulent_accounts(df):
        if 'AccountID' not in df.columns:
            # Removed st_object.warning
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        top_accounts = df[df['is_anomaly'] == 1]['AccountID'].value_counts().head(10)
        
        if not top_accounts.empty:
            top_accounts.plot(kind='barh', color='darkred', ax=ax)
            ax.set_title("Top 10 Fraudulent Accounts")
            ax.set_xlabel("Number of Fraudulent Transactions")
            ax.invert_yaxis()
            plt.tight_layout()
            return fig
        else:
            # Removed st_object.info
            return None

    def plot_correlation_heatmap(df, features_used):
        cols_for_corr = [f for f in features_used if f in df.columns]
        if 'is_anomaly' in df.columns and 'is_anomaly' not in cols_for_corr:
            cols_for_corr.append('is_anomaly')
        
        numeric_df = df[cols_for_corr].select_dtypes(include=np.number)

        if numeric_df.empty:
            # Removed st_object.warning
            return None

        corr = numeric_df.corr()
        
        if 'is_anomaly' in corr.columns:
            sorted_corr = corr[['is_anomaly']].sort_values(by='is_anomaly', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, max(6, len(sorted_corr) * 0.5)))
            sns.heatmap(sorted_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True)
            ax.set_title("Correlation of Features with Fraud Prediction")
            plt.tight_layout()
            return fig
        else:
            # Removed st_object.warning
            return None

    # Main execution flow for fraud_detection_analysis
    plot_images_b64 = {}

    try:
        # Removed st_object.info
        df_raw = load_data(file_path_or_bytes_obj)
    except Exception as e:
        raise ValueError(f"Data loading failed for fraud detection: {e}")

    try:
        # Removed st_object.info
        df_featured, features_for_model = feature_engineer_fraud_data(df_raw, primary_date_col=date_col_name)
    except Exception as e:
        raise ValueError(f"Feature engineering failed for fraud detection: {e}")

    try:
        # Removed st_object.info
        df_with_anomalies, anomalies_df, used_features = detect_anomalies(df_featured.copy(), features_for_model, contam=contamination)
    except ValueError as e:
        raise ValueError(f"Anomaly detection failed: {e}")
    except RuntimeError as e:
        raise RuntimeError(f"An unexpected error occurred during anomaly detection: {e}")
    except Exception as e:
        raise RuntimeError(f"A general error occurred during anomaly detection: {e}")


    anomaly_summary_list = summarize_anomalies(anomalies_df, dc_name=date_col_name)
    top_anomalies_df, amount_col_identified = top_anomalies(anomalies_df)

    # --- Generate Plots ---
    # Removed st_object.info
    
    # Plot 1: Anomaly Count Plot
    fig_count = plot_anomaly_count(df_with_anomalies)
    if fig_count: plot_images_b64['anomaly_count'] = get_base64_image(fig_count)
    
    # Plot 2: Fraud by Transaction Type
    fig_type = plot_fraud_by_transaction_type(df_with_anomalies)
    if fig_type: plot_images_b64['fraud_by_type'] = get_base64_image(fig_type)

    # Plot 3: Fraud Over Time (requires valid date column)
    fig_time = plot_fraud_over_time(df_with_anomalies, date_col_name)
    if fig_time: plot_images_b64['fraud_over_time'] = get_base64_image(fig_time)

    # Plot 4: Top Fraudulent Accounts (requires 'AccountID')
    fig_accounts = plot_top_fraudulent_accounts(df_with_anomalies)
    if fig_accounts: plot_images_b64['top_fraud_accounts'] = get_base64_image(fig_accounts)

    # Plot 5: Correlation Heatmap
    fig_corr = plot_correlation_heatmap(df_with_anomalies, used_features)
    if fig_corr: plot_images_b64['correlation_heatmap'] = get_base64_image(fig_corr)


    return df_with_anomalies, anomalies_df, anomaly_summary_list, top_anomalies_df, amount_col_identified, plot_images_b64