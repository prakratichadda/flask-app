# financial-analysis-suite-web/backend/api/invoice_processing.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io
import plotly.graph_objects as go


def process_invoices(file_path_or_bytes_obj: any):

    def load_and_clean_data(file_obj: any):
        df = pd.read_csv(file_obj)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df.drop_duplicates(inplace=True)
        
        required_cols = ['invoice_date', 'amount', 'product_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the invoice data.")
        
        df = df.dropna(subset=required_cols)

        if 'qty' in df.columns:
            df['qty'] = df['qty'].fillna(1)
        else:
            df['qty'] = 1
            
        if 'job' in df.columns:
            df['job'] = df['job'].fillna("Unknown")
        else:
            df['job'] = "Unknown"
        
        if 'email' in df.columns:
            df['email'] = df['email'].fillna("no-email@unknown.com")
        else:
            df['email'] = "no-email@unknown.com"


        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        if 'qty' in df.columns:
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
        
        df['total_value'] = df['amount'] * df['qty']
        df = df.dropna(subset=['invoice_date']) # Drop NaT dates after conversion
        df['invoice_month'] = df['invoice_date'].dt.to_period('M').astype(str)

        df = df.dropna(subset=['amount', 'total_value'])
        df = df[df['amount'] > 0]

        # --- DEBUG PRINT ---
        print("\n--- After load_and_clean_data ---")
        print("DF shape:", df.shape)
        print("DF columns:", df.columns.tolist())
        print("DF head:\n", df.head())
        print("DF dtypes:\n", df.dtypes)
        print("Missing values after cleaning:\n", df.isnull().sum())
        # --- END DEBUG PRINT ---

        return df


    def customer_segmentation_analysis(df):
        # Ensure 'city' and 'job' are present, or handle their absence
        if 'city' not in df.columns: df['city'] = 'unknown_city' # Use lowercase for consistency
        if 'job' not in df.columns: df['job'] = 'unknown' # Use lowercase for consistency

        # --- DEBUG PRINT ---
        print("\n--- Before Segmentation Groupby ---")
        print("DF head (for segmentation):\n", df[['city', 'job', 'total_value', 'amount', 'product_id']].head())
        print("Unique cities:", df['city'].unique())
        print("Unique jobs:", df['job'].unique())
        # --- END DEBUG PRINT ---

        segmentation = df.groupby(['city', 'job']).agg(
            total_revenue=('total_value', 'sum'),
            avg_invoice_amount=('amount', 'mean'),
            total_invoices=('product_id', 'count')
        ).reset_index()

        top_segments = segmentation.sort_values(by='total_revenue', ascending=False).head(10)

        city_revenue_data = segmentation.groupby('city')['total_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        city_revenue_fig = px.bar(
            city_revenue_data,
            x='city', y='total_revenue',
            title="Top 10 Cities by Revenue",
            labels={'total_revenue': 'Total Revenue', 'city': 'City'}
        )
        city_revenue_fig.update_layout(plot_bgcolor='white')

        monthly_revenue = df.groupby('invoice_month')['total_value'].sum().reset_index()
        monthly_revenue['invoice_month_sort'] = pd.to_datetime(monthly_revenue['invoice_month'])
        monthly_revenue = monthly_revenue.sort_values('invoice_month_sort').drop(columns='invoice_month_sort')

        revenue_trend_fig = px.line(
            monthly_revenue, x='invoice_month', y='total_value',
            title="Monthly Revenue Trend", markers=True,
            labels={'invoice_month': 'Month', 'total_value': 'Revenue'}
        )
        revenue_trend_fig.update_layout(plot_bgcolor='white')

        # --- DEBUG PRINT ---
        print("\n--- After Segmentation Analysis ---")
        print("Top Segments:\n", top_segments)
        print("City Revenue Data (for plot):\n", city_revenue_data)
        # --- END DEBUG PRINT ---

        return top_segments, city_revenue_fig, revenue_trend_fig


    def detect_fraud(df):
        df = df.copy()
        df['fraud_flag_rule'] = 0

        df.loc[df['amount'] <= 0, 'fraud_flag_rule'] = 1
        
        # Rule 2: Duplicate invoices based on date, first_name, product_id, amount
        duplicate_subset_cols = ['invoice_date', 'first_name', 'product_id', 'amount']
        existing_duplicate_cols = [col for col in duplicate_subset_cols if col in df.columns]

        # --- DEBUG PRINT ---
        print("\n--- Before Fraud Detection ---")
        print("DF head (for fraud):\n", df[existing_duplicate_cols + ['amount']].head())
        print("Checking for duplicates with columns:", existing_duplicate_cols)
        # --- END DEBUG PRINT ---

        if len(existing_duplicate_cols) == len(duplicate_subset_cols):
            duplicates = df.duplicated(subset=existing_duplicate_cols, keep=False)
            df.loc[duplicates, 'fraud_flag_rule'] = 1
            # --- DEBUG PRINT ---
            print("Rule-based duplicates found:", df['fraud_flag_rule'].sum())
            # --- END DEBUG PRINT ---
        else:
            print(f"Warning: Skipping rule-based duplicate fraud detection due to missing columns: {list(set(duplicate_subset_cols) - set(existing_duplicate_cols))}")

        # ML-based fraud detection: IsolationForest
        features = df[['amount']].copy().dropna()
        if features.empty:
            df['fraud_flag_ml'] = 0
            print("Warning: 'amount' feature is empty for ML fraud detection.")
        else:
            model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
            df.loc[features.index, 'fraud_flag_ml'] = model.fit_predict(features)
            df['fraud_flag_ml'] = df['fraud_flag_ml'].map({1: 0, -1: 1})
            df['fraud_flag_ml'] = df['fraud_flag_ml'].fillna(0).astype(int)
            print("ML-based fraud flags (total):", df['fraud_flag_ml'].sum())

        df['fraud_suspected'] = df[['fraud_flag_rule', 'fraud_flag_ml']].max(axis=1)

        if 'first_name' not in df.columns:
            df['first_name'] = 'unknown_client'

        suspicious = df[df['fraud_suspected'] == 1][
            ['first_name', 'invoice_date', 'amount', 'fraud_flag_rule', 'fraud_flag_ml']
        ]
        # --- DEBUG PRINT ---
        print("Suspicious Invoices found:\n", suspicious)
        # --- END DEBUG PRINT ---

        return suspicious


    def extract_named_entities(df):
        cols_to_check = ['first_name', 'last_name', 'invoice_date', 'email', 'city', 'amount', 'product_id', 'job']
        for col in cols_to_check:
            if col not in df.columns:
                df[col] = np.nan

        def extract_entities(row):
            return {
                "client_name": f"{row['first_name']} {row['last_name']}".strip() if pd.notna(row['first_name']) or pd.notna(row['last_name']) else "Unknown Client",
                "invoice_date": row['invoice_date'].strftime('%Y-%m-%d') if pd.notna(row['invoice_date']) else 'N/A',
                "email": row['email'] if pd.notna(row['email']) else 'N/A',
                "city": row['city'] if pd.notna(row['city']) else 'N/A',
                "amount": row['amount'] if pd.notna(row['amount']) else 0.0,
                "product_id": row['product_id'] if pd.notna(row['product_id']) else 'N/A',
                "job_role": row['job'] if pd.notna(row['job']) else 'N/A'
            }

        extracted = df.apply(extract_entities, axis=1)
        # --- DEBUG PRINT ---
        print("\n--- After Entity Extraction ---")
        print("Extracted Entities head:\n", extracted.head())
        # --- END DEBUG PRINT ---
        return pd.DataFrame(extracted.tolist())


    def budget_vs_actual_analysis(df):
        df = df.copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        
        if 'job' not in df.columns:
            df['job'] = 'uncategorized'

        df.dropna(subset=["amount", "invoice_date", "job"], inplace=True)
        
        # --- DEBUG PRINT ---
        print("\n--- Before Budget vs Actual Analysis ---")
        print("DF head (for budget):\n", df[['amount', 'invoice_date', 'job']].head())
        print("DF shape (for budget):", df.shape)
        # --- END DEBUG PRINT ---

        if df.empty:
            print("Warning: DataFrame is empty for Budget vs Actual analysis after dropping NaNs.")
            return pd.DataFrame(), pd.DataFrame()

        budget_reference = df.groupby("job")["amount"].sum().reset_index()
        budget_reference.rename(columns={"amount": "actual"}, inplace=True)

        np.random.seed(42)
        if not budget_reference.empty:
            budget_reference["budget"] = budget_reference["actual"] * np.random.uniform(0.8, 1.2, size=len(budget_reference))
        else:
            budget_reference["budget"] = 0

        actual_vs_budget = budget_reference.copy()
        actual_vs_budget["variance"] = actual_vs_budget["actual"] - actual_vs_budget["budget"]
        actual_vs_budget["status"] = actual_vs_budget["variance"].apply(lambda x: "🔴 Over" if x > 0 else "🟢 Under")

        audit_flags = pd.DataFrame()
        
        cols_for_duplicates = ['invoice_date', 'email', 'amount']
        existing_cols = [col for col in cols_for_duplicates if col in df.columns]
        if len(existing_cols) == len(cols_for_duplicates):
            duplicates_for_audit = df[df.duplicated(subset=existing_cols, keep=False)]
            audit_flags = pd.concat([audit_flags, duplicates_for_audit]).drop_duplicates()

        if not df.empty and 'amount' in df.columns and pd.api.types.is_numeric_dtype(df['amount']):
            threshold = df["amount"].quantile(0.95)
            high_value = df[df["amount"] > threshold]
            audit_flags = pd.concat([audit_flags, high_value]).drop_duplicates()
        
        display_cols = [col for col in ['invoice_id', 'invoice_date', 'first_name', 'amount', 'product_id'] if col in audit_flags.columns]
        if not display_cols and not audit_flags.empty:
             audit_flags = audit_flags.head()
        elif display_cols:
             audit_flags = audit_flags[display_cols]

        # --- DEBUG PRINT ---
        print("Budget vs Actual results:\n", actual_vs_budget)
        print("Audit Flags found:\n", audit_flags)
        # --- END DEBUG PRINT ---

        return actual_vs_budget, audit_flags

    # --- Main execution logic for process_invoices ---
    df = load_and_clean_data(file_path_or_bytes_obj)
    
    top_segments, city_revenue_fig, revenue_trend_fig = customer_segmentation_analysis(df.copy())
    suspicious_invoices = detect_fraud(df.copy())
    extracted_entities = extract_named_entities(df.copy())
    actual_vs_budget, audit_flags = budget_vs_actual_analysis(df.copy())

    return df, top_segments, city_revenue_fig, revenue_trend_fig, \
           suspicious_invoices, extracted_entities, actual_vs_budget, audit_flags