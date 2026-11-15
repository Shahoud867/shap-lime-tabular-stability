"""Download, verify, and emit codebooks for HMEQ, SGCS, and Taiwan Credit datasets."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.datasets import fetch_openml

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CODEBOOK_DIR = BASE_DIR / "data" / "codebook"

DATASET_SOURCES: dict[str, dict[str, Any]] = {
    "hmeq": {
        "name": "Home Equity (HMEQ)",
        "urls": [
            "https://raw.githubusercontent.com/sassoftware/sas-viya-dmml-pipelines/master/data/hmeq.csv",
        ],
        "filename": "hmeq.csv",
        "license": "Public benchmark dataset (Baesens et al., 2016; SAS Viya DMML Pipelines mirror)",
        "source_ref": "Baesens, Roesch & Scheule (2016), Credit Risk Analytics; SAS GitHub mirror",
        "expected_rows": 5960,
        "expected_features": 12,
        "expected_pos_rate_range": (0.19, 0.21),
    },
    "sgcs": {
        "name": "South German Credit (UCI ID 522)",
        "urls": [
            "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00522/SouthGermanCredit.asc",
        ],
        "filename": "south_german_credit.zip",
        "license": "CC BY 4.0 (UCI Machine Learning Repository, DOI: 10.24432/C5X89F)",
        "source_ref": "Groemping (2019), South German Credit, UCI ML Repository",
        "expected_rows": 1000,
        "expected_features": 20,
        "expected_pos_rate_range": (0.29, 0.31),
    },
    "taiwan": {
        "name": "Default of Credit Card Clients (Taiwan, UCI ID 350)",
        "urls": [
            "openml://42477",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        ],
        "filename": "taiwan_default.csv",
        "license": "CC BY 4.0 (UCI Machine Learning Repository, DOI: 10.24432/C55S3H; OpenML ID 42477)",
        "source_ref": "Yeh & Lien (2009), Expert Systems with Applications; UCI ML Repository",
        "expected_rows": 30000,
        "expected_features": 23,
        "expected_pos_rate_range": (0.21, 0.23),
    },
    "bank": {
        "name": "Bank Marketing (UCI / OpenML ID 1461)",
        "urls": ["openml://1461"],
        "filename": "bank_marketing.csv",
        "license": "CC BY 4.0 (UCI ML Repository / OpenML ID 1461)",
        "source_ref": "Moro, Cortez & Rita (2014), Decision Support Systems; OpenML ID 1461",
        "expected_rows": 45211,
        "expected_features": 16,
        "expected_pos_rate_range": (0.11, 0.125),
    },
    "adult": {
        "name": "Adult Census Income (UCI / OpenML ID 1590)",
        "urls": ["openml://1590"],
        "filename": "adult_income.csv",
        "license": "CC BY 4.0 (UCI ML Repository / OpenML ID 1590)",
        "source_ref": "Kohavi (1996), KDD; OpenML ID 1590",
        "expected_rows": 48842,
        "expected_features": 14,
        "expected_pos_rate_range": (0.23, 0.25),
    },
    "heart": {
        "name": "Heart Disease Comprehensive (OpenML ID 43682)",
        "urls": ["openml://43682"],
        "filename": "heart_disease.csv",
        "license": "CC0: Public Domain (OpenML ID 43682)",
        "source_ref": "Siddhartha (2020), Heart Disease Dataset Comprehensive; OpenML ID 43682",
        "expected_rows": 1190,
        "expected_features": 11,
        "expected_pos_rate_range": (0.51, 0.55),
    },
}

SGCS_COL_MAP = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

CODEBOOK_META: dict[str, dict[str, dict[str, Any]]] = {
    "hmeq": {
        "LOAN": {"type": "continuous", "desc": "Requested loan amount", "unit": "USD", "perturbable": True, "cost_role": "exposure_A_i"},
        "MORTDUE": {"type": "continuous", "desc": "Amount due on existing mortgage", "unit": "USD", "perturbable": True, "cost_role": "none"},
        "VALUE": {"type": "continuous", "desc": "Value of current property", "unit": "USD", "perturbable": True, "cost_role": "none"},
        "REASON": {"type": "categorical", "desc": "Loan purpose (DebtCon / HomeImp)", "unit": "category", "perturbable": False, "cost_role": "none"},
        "JOB": {"type": "categorical", "desc": "Occupational category", "unit": "category", "perturbable": False, "cost_role": "none"},
        "YOJ": {"type": "continuous", "desc": "Years at present job", "unit": "years", "perturbable": True, "cost_role": "none"},
        "DEROG": {"type": "count", "desc": "Number of major derogatory reports", "unit": "count", "perturbable": False, "cost_role": "none"},
        "DELINQ": {"type": "count", "desc": "Number of delinquent credit lines", "unit": "count", "perturbable": False, "cost_role": "none"},
        "CLAGE": {"type": "continuous", "desc": "Age of oldest credit line", "unit": "months", "perturbable": True, "cost_role": "none"},
        "NINQ": {"type": "count", "desc": "Number of recent credit inquiries", "unit": "count", "perturbable": False, "cost_role": "none"},
        "CLNO": {"type": "count", "desc": "Number of credit lines", "unit": "count", "perturbable": False, "cost_role": "none"},
        "DEBTINC": {"type": "continuous", "desc": "Debt-to-income ratio", "unit": "percent", "perturbable": True, "cost_role": "none"},
    },
    "sgcs": {
        "status": {"type": "categorical", "desc": "Status of existing checking account", "unit": "code", "perturbable": False, "cost_role": "none"},
        "duration": {"type": "continuous", "desc": "Loan duration", "unit": "months", "perturbable": True, "cost_role": "none"},
        "credit_history": {"type": "categorical", "desc": "Previous credit compliance history", "unit": "code", "perturbable": False, "cost_role": "none"},
        "purpose": {"type": "categorical", "desc": "Loan purpose", "unit": "code", "perturbable": False, "cost_role": "none"},
        "amount": {"type": "continuous", "desc": "Credit amount", "unit": "DEM", "perturbable": True, "cost_role": "exposure_A_i"},
        "savings": {"type": "categorical", "desc": "Savings account / bonds balance", "unit": "code", "perturbable": False, "cost_role": "none"},
        "employment_duration": {"type": "ordinal", "desc": "Duration of present employment", "unit": "code", "perturbable": False, "cost_role": "none"},
        "installment_rate": {"type": "ordinal", "desc": "Installment rate as percentage of disposable income", "unit": "code", "perturbable": False, "cost_role": "none"},
        "personal_status_sex": {"type": "categorical", "desc": "Personal status and sex", "unit": "code", "perturbable": False, "cost_role": "none"},
        "other_debtors": {"type": "categorical", "desc": "Other debtors / guarantors", "unit": "code", "perturbable": False, "cost_role": "none"},
        "present_residence": {"type": "ordinal", "desc": "Years at present residence", "unit": "code", "perturbable": False, "cost_role": "none"},
        "property": {"type": "categorical", "desc": "Most valuable property type", "unit": "code", "perturbable": False, "cost_role": "none"},
        "age": {"type": "continuous", "desc": "Applicant age", "unit": "years", "perturbable": True, "cost_role": "none"},
        "other_installment_plans": {"type": "categorical", "desc": "Other installment plans", "unit": "code", "perturbable": False, "cost_role": "none"},
        "housing": {"type": "categorical", "desc": "Housing type (rent/own/free)", "unit": "code", "perturbable": False, "cost_role": "none"},
        "number_credits": {"type": "ordinal", "desc": "Number of existing credits at this bank", "unit": "code", "perturbable": False, "cost_role": "none"},
        "job": {"type": "categorical", "desc": "Employment skill / job level", "unit": "code", "perturbable": False, "cost_role": "none"},
        "people_liable": {"type": "categorical", "desc": "Number of dependents liable to support", "unit": "code", "perturbable": False, "cost_role": "none"},
        "telephone": {"type": "categorical", "desc": "Registered telephone indicator", "unit": "code", "perturbable": False, "cost_role": "none"},
        "foreign_worker": {"type": "categorical", "desc": "Foreign worker indicator", "unit": "code", "perturbable": False, "cost_role": "none"},
    },
    "taiwan": {
        "LIMIT_BAL": {"type": "continuous", "desc": "Amount of given credit limit", "unit": "NT_dollar", "perturbable": True, "cost_role": "exposure_A_i"},
        "SEX": {"type": "categorical", "desc": "Gender (1=male, 2=female)", "unit": "code", "perturbable": False, "cost_role": "none"},
        "EDUCATION": {"type": "categorical", "desc": "Education level", "unit": "code", "perturbable": False, "cost_role": "none"},
        "MARRIAGE": {"type": "categorical", "desc": "Marital status", "unit": "code", "perturbable": False, "cost_role": "none"},
        "AGE": {"type": "count", "desc": "Age in years", "unit": "years", "perturbable": False, "cost_role": "none"},
        **{f"PAY_{i}": {"type": "ordinal", "desc": f"Repayment status in month {idx}", "unit": "months_delay", "perturbable": False, "cost_role": "none"} for idx, i in enumerate([0, 2, 3, 4, 5, 6], start=1)},
        **{f"BILL_AMT{i}": {"type": "continuous", "desc": f"Bill statement amount month {i}", "unit": "NT_dollar", "perturbable": True, "cost_role": "none"} for i in range(1, 7)},
        **{f"PAY_AMT{i}": {"type": "continuous", "desc": f"Previous payment amount month {i}", "unit": "NT_dollar", "perturbable": True, "cost_role": "none"} for i in range(1, 7)},
    },
    "bank": {
        "age": {"type": "continuous", "desc": "Client age", "unit": "years", "perturbable": True, "cost_role": "none"},
        "job": {"type": "categorical", "desc": "Type of job", "unit": "category", "perturbable": False, "cost_role": "none"},
        "marital": {"type": "categorical", "desc": "Marital status", "unit": "category", "perturbable": False, "cost_role": "none"},
        "education": {"type": "categorical", "desc": "Education level", "unit": "category", "perturbable": False, "cost_role": "none"},
        "default": {"type": "categorical", "desc": "Has credit in default", "unit": "category", "perturbable": False, "cost_role": "none"},
        "balance": {"type": "continuous", "desc": "Average yearly bank balance", "unit": "EUR", "perturbable": True, "cost_role": "none"},
        "housing": {"type": "categorical", "desc": "Has housing loan", "unit": "category", "perturbable": False, "cost_role": "none"},
        "loan": {"type": "categorical", "desc": "Has personal loan", "unit": "category", "perturbable": False, "cost_role": "none"},
        "contact": {"type": "categorical", "desc": "Contact communication type", "unit": "category", "perturbable": False, "cost_role": "none"},
        "day": {"type": "ordinal", "desc": "Last contact day of month", "unit": "day", "perturbable": False, "cost_role": "none"},
        "month": {"type": "categorical", "desc": "Last contact month of year", "unit": "month", "perturbable": False, "cost_role": "none"},
        "duration": {"type": "continuous", "desc": "Last contact duration", "unit": "seconds", "perturbable": True, "cost_role": "none"},
        "campaign": {"type": "continuous", "desc": "Number of contacts during campaign", "unit": "count", "perturbable": True, "cost_role": "none"},
        "pdays": {"type": "continuous", "desc": "Days since client last contacted (-1=uncontacted)", "unit": "days", "perturbable": True, "cost_role": "none"},
        "previous": {"type": "continuous", "desc": "Number of contacts before campaign", "unit": "count", "perturbable": True, "cost_role": "none"},
        "poutcome": {"type": "categorical", "desc": "Outcome of previous marketing campaign", "unit": "category", "perturbable": False, "cost_role": "none"},
    },
    "adult": {
        "age": {"type": "continuous", "desc": "Individual age", "unit": "years", "perturbable": True, "cost_role": "none"},
        "workclass": {"type": "categorical", "desc": "Employment sector", "unit": "category", "perturbable": False, "cost_role": "none"},
        "fnlwgt": {"type": "continuous", "desc": "CPS sampling weight", "unit": "weight", "perturbable": True, "cost_role": "none"},
        "education": {"type": "categorical", "desc": "Highest education level", "unit": "category", "perturbable": False, "cost_role": "none"},
        "education-num": {"type": "ordinal", "desc": "Years of education", "unit": "years", "perturbable": False, "cost_role": "none"},
        "marital-status": {"type": "categorical", "desc": "Marital status", "unit": "category", "perturbable": False, "cost_role": "none"},
        "occupation": {"type": "categorical", "desc": "Occupation category", "unit": "category", "perturbable": False, "cost_role": "none"},
        "relationship": {"type": "categorical", "desc": "Household relationship role", "unit": "category", "perturbable": False, "cost_role": "none"},
        "race": {"type": "categorical", "desc": "Demographic race category", "unit": "category", "perturbable": False, "cost_role": "none"},
        "sex": {"type": "categorical", "desc": "Sex", "unit": "category", "perturbable": False, "cost_role": "none"},
        "capital-gain": {"type": "continuous", "desc": "Capital gains", "unit": "USD", "perturbable": True, "cost_role": "none"},
        "capital-loss": {"type": "continuous", "desc": "Capital losses", "unit": "USD", "perturbable": True, "cost_role": "none"},
        "hours-per-week": {"type": "continuous", "desc": "Hours worked per week", "unit": "hours", "perturbable": True, "cost_role": "none"},
        "native-country": {"type": "categorical", "desc": "Country of origin", "unit": "category", "perturbable": False, "cost_role": "none"},
    },
    "heart": {
        "age": {"type": "continuous", "desc": "Patient age", "unit": "years", "perturbable": True, "cost_role": "none"},
        "sex": {"type": "categorical", "desc": "Patient sex (1=male, 0=female)", "unit": "code", "perturbable": False, "cost_role": "none"},
        "chest_pain_type": {"type": "categorical", "desc": "Chest pain type (1-4)", "unit": "code", "perturbable": False, "cost_role": "none"},
        "resting_bp_s": {"type": "continuous", "desc": "Resting blood pressure", "unit": "mmHg", "perturbable": True, "cost_role": "none"},
        "cholesterol": {"type": "continuous", "desc": "Serum cholesterol", "unit": "mg/dl", "perturbable": True, "cost_role": "none"},
        "fasting_blood_sugar": {"type": "categorical", "desc": "Fasting blood sugar > 120 mg/dl", "unit": "binary", "perturbable": False, "cost_role": "none"},
        "resting_ecg": {"type": "categorical", "desc": "Resting ECG results (0-2)", "unit": "code", "perturbable": False, "cost_role": "none"},
        "max_heart_rate": {"type": "continuous", "desc": "Maximum heart rate achieved", "unit": "bpm", "perturbable": True, "cost_role": "none"},
        "exercise_angina": {"type": "categorical", "desc": "Exercise-induced angina (1=yes, 0=no)", "unit": "binary", "perturbable": False, "cost_role": "none"},
        "oldpeak": {"type": "continuous", "desc": "ST depression induced by exercise relative to rest", "unit": "mm", "perturbable": True, "cost_role": "none"},
        "ST_slope": {"type": "ordinal", "desc": "Slope of peak exercise ST segment (1-3)", "unit": "code", "perturbable": False, "cost_role": "none"},
    },
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


TAIWAN_OPENML_COL_MAP = {
    "x1": "LIMIT_BAL",
    "x2": "SEX",
    "x3": "EDUCATION",
    "x4": "MARRIAGE",
    "x5": "AGE",
    "x6": "PAY_0",
    "x7": "PAY_2",
    "x8": "PAY_3",
    "x9": "PAY_4",
    "x10": "PAY_5",
    "x11": "PAY_6",
    "x12": "BILL_AMT1",
    "x13": "BILL_AMT2",
    "x14": "BILL_AMT3",
    "x15": "BILL_AMT4",
    "x16": "BILL_AMT5",
    "x17": "BILL_AMT6",
    "x18": "PAY_AMT1",
    "x19": "PAY_AMT2",
    "x20": "PAY_AMT3",
    "x21": "PAY_AMT4",
    "x22": "PAY_AMT5",
    "x23": "PAY_AMT6",
    "y": "target",
}


BANK_OPENML_COL_MAP = {
    "V1": "age",
    "V2": "job",
    "V3": "marital",
    "V4": "education",
    "V5": "default",
    "V6": "balance",
    "V7": "housing",
    "V8": "loan",
    "V9": "contact",
    "V10": "day",
    "V11": "month",
    "V12": "duration",
    "V13": "campaign",
    "V14": "pdays",
    "V15": "previous",
    "V16": "poutcome",
    "Class": "target",
}


def download_with_fallback(urls: list[str], dest: Path) -> str:
    """Download from the first working URL (or openml://id) and return the URL used."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        try:
            if url.startswith("openml://"):
                data_id = int(url.split("openml://", 1)[1])
                bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
                df = bunch.frame.copy()
                if data_id == 42477 and "x1" in df.columns:
                    df = df.rename(columns=TAIWAN_OPENML_COL_MAP)
                elif data_id == 1461 and "V1" in df.columns:
                    df = df.rename(columns=BANK_OPENML_COL_MAP)
                df.to_csv(dest, index=False)
                return f"https://www.openml.org/d/{data_id}"
            resp = requests.get(url, timeout=(15, 180))
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return url
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Failed to download {dest.name} from {urls}: {last_err}")


def load_raw_hmeq(raw_path: Path) -> pd.DataFrame:
    """Parse raw HMEQ CSV and normalize target to `target` (1=default/BAD)."""
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"BAD": "target"})
    df["target"] = df["target"].astype(int)
    return df


def load_raw_sgcs(raw_path: Path) -> pd.DataFrame:
    """Parse raw SGCS zip or asc file; map German columns and set 1=bad credit (30%)."""
    if raw_path.suffix == ".zip":
        with zipfile.ZipFile(raw_path, "r") as zf:
            asc_names = [n for n in zf.namelist() if n.endswith(".asc") or n.endswith(".csv")]
            if not asc_names:
                raise ValueError(f"No .asc/.csv found in {raw_path}: {zf.namelist()}")
            content = zf.read(asc_names[0]).decode("utf-8")
    else:
        content = raw_path.read_text(encoding="utf-8")

    df = pd.read_csv(io.StringIO(content), sep=r"\s+")
    if "laufkont" in df.columns:
        df = df.rename(columns=SGCS_COL_MAP)
    # In UCI SGCS: kredit=1 means GOOD credit (70%), kredit=0 means BAD credit (30%).
    # Standardize to target=1 for DEFAULT (bad credit, 30%), matching Ballegeer et al. (2025).
    if "credit_risk" in df.columns:
        good_rate = (df["credit_risk"] == 1).mean()
        if good_rate > 0.5:
            df["target"] = (1 - df["credit_risk"]).astype(int)
        else:
            df["target"] = df["credit_risk"].astype(int)
        df = df.drop(columns=["credit_risk"])
    return df


def load_raw_taiwan(raw_path: Path) -> pd.DataFrame:
    """Parse raw Taiwan Credit CSV, zip, or xls; drop ID and standardize target."""
    if raw_path.suffix == ".csv":
        df = pd.read_csv(raw_path)
    elif raw_path.suffix == ".zip":
        with zipfile.ZipFile(raw_path, "r") as zf:
            xls_names = [n for n in zf.namelist() if n.endswith((".xls", ".xlsx", ".csv"))]
            if not xls_names:
                raise ValueError(f"No spreadsheet found in {raw_path}: {zf.namelist()}")
            target_name = xls_names[0]
            data_bytes = zf.read(target_name)
            if target_name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(data_bytes))
            else:
                df = pd.read_excel(io.BytesIO(data_bytes), header=1)
    else:
        df = pd.read_excel(raw_path, header=1)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "target" not in df.columns:
        target_col = [c for c in df.columns if "default" in str(c).lower()][0]
        df = df.rename(columns={target_col: "target"})
    df["target"] = df["target"].astype(int)
    return df


def load_raw_bank(raw_path: Path) -> pd.DataFrame:
    """Parse raw Bank Marketing CSV and map Class ('2'=subscribed, '1'=no) to target 1/0."""
    df = pd.read_csv(raw_path)
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "target"})
    df["target"] = df["target"].astype(str).str.strip().map({"2": 1, "1": 0, "yes": 1, "no": 0}).astype(int)
    return df


def load_raw_adult(raw_path: Path) -> pd.DataFrame:
    """Parse raw Adult Income CSV and map class ('>50K' -> 1, '<=50K' -> 0)."""
    df = pd.read_csv(raw_path, na_values=["?"])
    if "class" in df.columns:
        df = df.rename(columns={"class": "target"})
    df["target"] = df["target"].astype(str).str.strip().map({">50K": 1, "<=50K": 0, "1": 1, "0": 0}).astype(int)
    return df


def load_raw_heart(raw_path: Path) -> pd.DataFrame:
    """Parse raw Heart Disease Comprehensive CSV with integer target (1/0)."""
    df = pd.read_csv(raw_path)
    df["target"] = df["target"].astype(float).astype(int)
    return df


def emit_codebook(dataset: str, df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Create codebook CSV for a dataset with types, perturbable flag, and missingness."""
    meta = CODEBOOK_META[dataset]
    rows = []
    for col in df.columns:
        if col == "target":
            rows.append({
                "variable": "target",
                "type": "binary_target",
                "description": "Default / bad credit indicator (1=default, 0=non-default)",
                "unit": "binary",
                "transformation": "standardized to 1=default",
                "perturbable": False,
                "cost_exposure_role": "target",
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100.0), 4),
            })
            continue
        info = meta[col]
        trans = (
            "median_impute + standard_scale"
            if info["type"] in ("continuous", "count")
            else "mode_impute + one_hot_encode"
            if info["type"] == "categorical"
            else "median_impute + standard_scale"
        )
        rows.append({
            "variable": col,
            "type": info["type"],
            "description": info["desc"],
            "unit": info["unit"],
            "transformation": trans,
            "perturbable": bool(info["perturbable"]),
            "cost_exposure_role": info["cost_role"],
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100.0), 4),
        })
    cb = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cb.to_csv(out_path, index=False)
    return cb


def main(force_download: bool = False) -> dict[str, Any]:
    """Download all three datasets, verify integrity, emit processed CSVs and codebooks."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    loaders = {
        "hmeq": load_raw_hmeq,
        "sgcs": load_raw_sgcs,
        "taiwan": load_raw_taiwan,
        "bank": load_raw_bank,
        "adult": load_raw_adult,
        "heart": load_raw_heart,
    }

    manifest: dict[str, Any] = {}
    for key, spec in DATASET_SOURCES.items():
        raw_path = RAW_DIR / spec["filename"]
        if force_download or not raw_path.exists():
            used_url = download_with_fallback(spec["urls"], raw_path)
        else:
            used_url = spec["urls"][0]

        df = loaders[key](raw_path)
        n_rows = len(df)
        n_features = df.shape[1] - 1
        pos_rate = float(df["target"].mean())
        lo, hi = spec["expected_pos_rate_range"]

        assert n_rows == spec["expected_rows"], f"{key}: expected {spec['expected_rows']} rows, got {n_rows}"
        assert n_features == spec["expected_features"], f"{key}: expected {spec['expected_features']} features, got {n_features}"
        assert lo <= pos_rate <= hi, f"{key}: positive rate {pos_rate:.4f} outside [{lo}, {hi}]"

        clean_path = PROCESSED_DIR / f"{key}_clean.csv"
        df.to_csv(clean_path, index=False)

        cb_path = CODEBOOK_DIR / f"{key}_codebook.csv"
        emit_codebook(key, df, cb_path)

        manifest[key] = {
            "name": spec["name"],
            "source_url": used_url,
            "raw_file": str(raw_path.relative_to(BASE_DIR)),
            "sha256": sha256_file(raw_path),
            "license": spec["license"],
            "source_ref": spec["source_ref"],
            "rows": n_rows,
            "features": n_features,
            "positive_rate": round(pos_rate, 6),
            "total_missing_cells": int(df.isna().sum().sum()),
        }

    manifest_path = RAW_DIR / "checksums_and_licenses.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary, indent=2))
