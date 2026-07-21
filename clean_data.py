"""
This will do an audit before an after EDA of the dataset

The original CSV is never modified. This script audits the raw data, removes
duplicate post IDs and the exported index column, and then audits the cleaned
DataFrame again.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).with_name("Mental-Health-Twitter.csv")


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
            "count": df["label"].value_counts().sort_index(),
            "percent": (
                df["label"].value_counts(normalize=True).sort_index().mul(100).round(2)
            ),
        }
    )
    print(label_summary)

    print("\n4. Duplicate checks")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")
    print(f"Repeated post IDs: {df.duplicated('post_id').sum():,}")
    print(f"Repeated post text: {df.duplicated('post_text').sum():,}")
    print(
        "Repeated user-and-text pairs: "
        f"{df.duplicated(['user_id', 'post_text']).sum():,}"
    )
    conflicting_text = df.groupby("post_text")["label"].nunique().gt(1).sum()
    print(f"Repeated texts with conflicting labels: {conflicting_text:,}")

    print("\n5. User-level checks")
    print(f"Unique users: {df['user_id'].nunique():,}")


def main():
    df = pd.read_csv(DATA_PATH)

    audit_data(df, "RAW DATA AUDIT")

    # remove duplicate post ids and posts with identical text
    df = df.drop_duplicates(subset="post_id", keep="first")
    df = df.drop_duplicates(subset="post_text", keep="first")
    df = df.drop(columns=["Unnamed: 0"])

    audit_data(df, "CLEANED DATA AUDIT")


if __name__ == "__main__":
    main()
