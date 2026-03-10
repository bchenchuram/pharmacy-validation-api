# Pharmacy Validation API

FastAPI-based validation API for weekly pharmacy sales datasets, designed for forecasting readiness and IBM watsonx agent integration.

## Features
- Validates required columns
- Standardizes `datum` to `week_start`
- Checks valid dates
- Checks missing values
- Checks duplicate weekly records
- Checks numeric and non-negative demand fields
- Validates `total_sales` consistency
- Checks chronological order

## Run locally

```bash
pip install -r requirements.txt
python pharmacy_validation_api.py
