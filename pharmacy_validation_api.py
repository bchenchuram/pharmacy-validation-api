from __future__ import annotations

from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

app = FastAPI(
    title="Pharmacy Data Quality API",
    version="1.0.0",
    description="Validates weekly pharmacy sales data before forecasting.",
)

REQUIRED_CATEGORY_COLUMNS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
REQUIRED_COLUMNS = ["week_start", *REQUIRED_CATEGORY_COLUMNS, "total_sales"]


class ValidationRequest(BaseModel):
    file_path: Optional[str] = Field(default=None, description="Optional local CSV path")


class CheckResult(BaseModel):
    check_name: str
    status: str
    details: str


class ValidationResponse(BaseModel):
    dataset_status: str
    rows: int
    columns: int
    checks_performed: List[CheckResult]
    issues_found: List[str]
    recommendation: str
    normalized_columns: List[str]


def _read_csv_from_upload(upload: UploadFile) -> pd.DataFrame:
    try:
        content = upload.file.read()
        if not content:
            raise ValueError("Uploaded file is empty.")
        return pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc


def _read_csv_from_path(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV from path: {exc}") from exc


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "datum" in df.columns and "week_start" not in df.columns:
        df = df.rename(columns={"datum": "week_start"})

    return df


def _ensure_total_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "total_sales" not in df.columns and all(col in df.columns for col in REQUIRED_CATEGORY_COLUMNS):
        df["total_sales"] = df[REQUIRED_CATEGORY_COLUMNS].sum(axis=1)
    return df


def _validate(df: pd.DataFrame) -> ValidationResponse:
    checks: List[CheckResult] = []
    issues: List[str] = []

    df = _normalize_columns(df)
    df = _ensure_total_sales(df)

    # Required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        checks.append(CheckResult(check_name="required_columns", status="fail", details=issues[-1]))
    else:
        checks.append(CheckResult(check_name="required_columns", status="pass", details="All required columns are present."))

    # Date validity
    valid_dates = False
    if "week_start" in df.columns:
        parsed_dates = pd.to_datetime(df["week_start"], errors="coerce")
        invalid_date_count = int(parsed_dates.isna().sum())
        if invalid_date_count > 0:
            issues.append(f"Invalid or missing week_start values: {invalid_date_count}")
            checks.append(CheckResult(check_name="week_start_validity", status="fail", details=issues[-1]))
        else:
            valid_dates = True
            checks.append(CheckResult(check_name="week_start_validity", status="pass", details="week_start contains valid dates."))
            df["week_start"] = parsed_dates
    else:
        checks.append(CheckResult(check_name="week_start_validity", status="fail", details="week_start column not available."))

    # Null check
    available_required = [col for col in REQUIRED_COLUMNS if col in df.columns]
    if available_required:
        null_counts = df[available_required].isna().sum()
        critical_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
        if critical_nulls:
            issues.append(f"Critical nulls found: {critical_nulls}")
            checks.append(CheckResult(check_name="critical_null_check", status="fail", details=issues[-1]))
        else:
            checks.append(CheckResult(check_name="critical_null_check", status="pass", details="No critical nulls in required fields."))

    # Duplicate weekly records
    if "week_start" in df.columns:
        duplicate_weeks = int(df.duplicated(subset=["week_start"]).sum())
        if duplicate_weeks > 0:
            issues.append(f"Duplicate weekly records found: {duplicate_weeks}")
            checks.append(CheckResult(check_name="weekly_uniqueness", status="fail", details=issues[-1]))
        else:
            checks.append(CheckResult(check_name="weekly_uniqueness", status="pass", details="No duplicate weekly records found."))

    # Numeric and non-negative checks
    numeric_failures: List[str] = []
    negative_failures: List[str] = []

    for col in REQUIRED_CATEGORY_COLUMNS + ["total_sales"]:
        if col not in df.columns:
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")
        non_numeric_count = int(numeric_series.isna().sum())
        if non_numeric_count > 0:
            numeric_failures.append(f"{col}: {non_numeric_count} non-numeric values")

        negative_count = int((numeric_series < 0).sum())
        if negative_count > 0:
            negative_failures.append(f"{col}: {negative_count} negative values")

        df[col] = numeric_series

    if numeric_failures:
        issues.append("Non-numeric values found in demand fields: " + "; ".join(numeric_failures))
        checks.append(CheckResult(check_name="numeric_validity", status="fail", details=issues[-1]))
    else:
        checks.append(CheckResult(check_name="numeric_validity", status="pass", details="Demand fields are numeric."))

    if negative_failures:
        issues.append("Negative values found in demand fields: " + "; ".join(negative_failures))
        checks.append(CheckResult(check_name="non_negative_check", status="fail", details=issues[-1]))
    else:
        checks.append(CheckResult(check_name="non_negative_check", status="pass", details="Demand fields are non-negative."))

    # total_sales consistency
    if all(col in df.columns for col in REQUIRED_CATEGORY_COLUMNS + ["total_sales"]):
        calculated_total = df[REQUIRED_CATEGORY_COLUMNS].sum(axis=1).round(2)
        provided_total = pd.to_numeric(df["total_sales"], errors="coerce").round(2)
        mismatches = int((calculated_total != provided_total).sum())
        if mismatches > 0:
            issues.append(f"total_sales consistency failed for {mismatches} row(s).")
            checks.append(CheckResult(check_name="total_sales_consistency", status="fail", details=issues[-1]))
        else:
            checks.append(CheckResult(check_name="total_sales_consistency", status="pass", details="total_sales matches the sum of category fields."))

    # Chronological order
    if valid_dates:
        is_sorted = bool(df["week_start"].is_monotonic_increasing)
        if not is_sorted:
            issues.append("Dataset is not sorted in chronological order by week_start.")
            checks.append(CheckResult(check_name="chronological_order", status="fail", details=issues[-1]))
        else:
            checks.append(CheckResult(check_name="chronological_order", status="pass", details="Dataset is sorted by week_start."))

    dataset_status = "pass" if not issues else "fail"
    recommendation = (
        "Dataset is ready for downstream forecasting."
        if dataset_status == "pass"
        else "Fix the reported issues before using this dataset for forecasting."
    )

    return ValidationResponse(
        dataset_status=dataset_status,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        checks_performed=checks,
        issues_found=issues,
        recommendation=recommendation,
        normalized_columns=list(df.columns),
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Pharmacy Data Quality API is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/data/validate", response_model=ValidationResponse)
def validate_dataset(request: ValidationRequest) -> ValidationResponse:
    if not request.file_path:
        raise HTTPException(status_code=400, detail="Provide file_path in the request body.")
    df = _read_csv_from_path(request.file_path)
    return _validate(df)


@app.post("/api/data/validate-upload", response_model=ValidationResponse)
def validate_uploaded_dataset(file: UploadFile = File(...)) -> ValidationResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    df = _read_csv_from_upload(file)
    return _validate(df)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
