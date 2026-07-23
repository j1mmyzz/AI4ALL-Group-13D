"""
This will do an audit before an after EDA of the dataset

The original CSV is never modified. This script audits the raw data, removes
duplicate post IDs and the exported index column, and then audits the cleaned
DataFrame again.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).with_name("depression_dataset_reddit_cleaned.csv")


def audit_data(df, title):
    print(f"\n{title}")

    print("\n1. Dataset structure")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)

    print("\n2. Missing values")
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean().mul(100).round(2),
        }
    )
    print(missing)

    print("\n3. Target-label checks")
    label_summary = pd.DataFrame(
        {
            "count": df["is_depression"].value_counts().sort_index(),
            "percent": (
                df["is_depression"]
                .value_counts(normalize=True)
                .sort_index()
                .mul(100)
                .round(2)
            ),
        }
    )
    print(label_summary)

    print("\n4. Duplicate checks")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")
    print(f"Repeated post text: {df.duplicated('clean_text').sum():,}")
    conflicting_text = df.groupby("clean_text")["is_depression"].nunique().gt(1).sum()
    print(f"Repeated texts with conflicting labels: {conflicting_text:,}")


def main():
    df = pd.read_csv(DATA_PATH)

    audit_data(df, "RAW DATA AUDIT")

    # remove duplicate posts with identical text
    df = df.drop_duplicates(subset="clean_text", keep="first")

    audit_data(df, "CLEANED DATA AUDIT")


if __name__ == "__main__":
    main()
