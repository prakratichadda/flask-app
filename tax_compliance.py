# tax_compliance.py

import pandas as pd # Not directly used in calculate_tax_liability, but often useful in financial contexts
import numpy as np # Not directly used, but good to have for numerical operations

def calculate_tax_liability(income: float, deductions: float, year: int) -> dict:
    """
    Calculates the tax liability based on income, deductions, and tax year (India, new regime for simplicity).
    Supports tax year 2024.

    Parameters:
        income (float): The gross income.
        deductions (float): The eligible deductions (e.g., 80C, 80D, HRA exemptions, Standard Deduction, etc.).
                            Note: For the new regime, most common deductions are not allowed.
                            This function assumes `deductions` are those allowed under the chosen regime (here, implicitly old regime for simplicity, or specific new regime deductions).
        year (int): The tax year (e.g., 2024 for Assessment Year 2025-26).

    Returns:
        dict: Contains gross income, total deductions, taxable income, total tax, and breakdown.
    """

    # For simplicity, this assumes a basic understanding of deductions against gross income.
    # In actual Indian tax compliance, deductions vary greatly by old vs. new regime.
    # For this current simplified model, we will use the OLD regime slabs as they allow for deductions.
    # The comments in your original code with ₹2.5L etc. align with the old regime.

    taxable_income = max(income - deductions, 0)

    # Define tax slabs based on the year (India's Old Tax Regime for individuals below 60, for example)
    # The provided slabs look like the OLD regime pre-budget 2023 for general category.
    # The new regime (default from FY23-24 / AY24-25) is simpler but with fewer deductions.
    # To use deductions, we stick to what looks like old regime slabs.
    if year == 2024: # Assuming this refers to Financial Year 2023-24 / Assessment Year 2024-25
        # Old Tax Regime slabs (example for individuals below 60 years)
        slabs = [
            (250000, 0.0),      # Up to ₹2.5 Lakhs: No tax
            (500000, 0.05),     # ₹2.5 Lakhs to ₹5 Lakhs: 5%
            (1000000, 0.20),    # ₹5 Lakhs to ₹10 Lakhs: 20%
            (float('inf'), 0.30) # Above ₹10 Lakhs: 30%
        ]
        # Surcharge and Cess are not included in this base calculation for simplicity,
        # but would be added in a full compliance tool.
        # Health & Education Cess: 4% on total tax (if applicable).
    elif year == 2023: # Example: If you need to support a previous year
         slabs = [
            (250000, 0.0),
            (500000, 0.05),
            (1000000, 0.2),
            (float('inf'), 0.3)
        ]
    else:
        raise ValueError("Unsupported tax year. This calculator only supports tax years 2023 and 2024 for India (old regime-like slabs).")

    total_tax = 0.0
    prev_limit = 0.0
    breakdown = []
    income_remaining = taxable_income

    for limit, rate in slabs:
        if income_remaining <= 0:
            break

        slab_upper_bound = limit
        slab_lower_bound = prev_limit

        # Calculate income in the current slab
        taxable_in_this_slab = min(income_remaining, slab_upper_bound - slab_lower_bound)

        if taxable_in_this_slab > 0:
            tax_segment = taxable_in_this_slab * rate
            total_tax += tax_segment
            breakdown.append({
                "slab_range": f"₹{slab_lower_bound:,.0f} - ₹{slab_upper_bound:,.0f}" if slab_upper_bound != float('inf') else f"Above ₹{slab_lower_bound:,.0f}",
                "amount_in_slab": round(taxable_in_this_slab, 2),
                "rate": f"{rate*100:.0f}%",
                "tax_in_segment": round(tax_segment, 2)
            })
        
        income_remaining -= taxable_in_this_slab
        prev_limit = limit

    # Add Health and Education Cess (4% of income tax)
    cess = total_tax * 0.04
    total_tax_with_cess = total_tax + cess

    return {
        "gross_income": income,
        "total_deductions": deductions,
        "taxable_income": round(taxable_income, 2),
        "tax_before_cess": round(total_tax, 2),
        "cess": round(cess, 2),
        "total_tax_liability": round(total_tax_with_cess, 2),
        "tax_breakdown": breakdown
    }

# Example usage (for testing)
if __name__ == "__main__":
    income_example = 1200000  # ₹12 Lakhs
    deductions_example = 150000 # ₹1.5 Lakhs (e.g., 80C)
    tax_year_example = 2024

    result = calculate_tax_liability(income_example, deductions_example, tax_year_example)
    print("Tax Calculation Result:")
    for key, value in result.items():
        if key == "tax_breakdown":
            print(f"  {key.replace('_', ' ').title()}:")
            for segment in value:
                print(f"    - Slab {segment['slab_range']}: Amount in Slab: ₹{segment['amount_in_slab']:,.2f}, Rate: {segment['rate']}, Tax: ₹{segment['tax_in_segment']:,.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: ₹{value:,.2f}" if isinstance(value, (int, float)) else f"  {key.replace('_', ' ').title()}: {value}")

    print("\n--- Testing another scenario ---")
    result2 = calculate_tax_liability(income=400000, deductions=50000, year=2024) # Taxable: 350,000
    print("Tax Calculation Result (Income 4L, Ded 0.5L):")
    for key, value in result2.items():
        if key == "tax_breakdown":
            print(f"  {key.replace('_', ' ').title()}:")
            for segment in value:
                print(f"    - Slab {segment['slab_range']}: Amount in Slab: ₹{segment['amount_in_slab']:,.2f}, Rate: {segment['rate']}, Tax: ₹{segment['tax_in_segment']:,.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: ₹{value:,.2f}" if isinstance(value, (int, float)) else f"  {key.replace('_', ' ').title()}: {value}")