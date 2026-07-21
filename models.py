from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).with_name("Mental-Health-Twitter.csv")


def main():
    # Load and clean the data using the same rules as clean_data.py
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset="post_id", keep="first")
    df = df.drop_duplicates(subset="post_text", keep="first")

    # The post text is the input and the label is what we predict
    X = df["post_text"]
    y = df["label"]

    # Split the data into 20% of the posts for testing and 80% for training the model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Convert text into TF-IDF features, then train Logistic Regression
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Logistic Regression results:")
    print(classification_report(y_test, predictions))

    # Test a new post written by a user
    new_post = input("\nEnter a post to classify: ")
    prediction = model.predict([new_post])[0]

    if prediction == 1:
        print("Prediction: depression-associated language")
    else:
        print("Prediction: non-depression-associated language")


if __name__ == "__main__":
    main()
