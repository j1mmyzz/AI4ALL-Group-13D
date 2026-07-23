from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path(__file__).with_name("depression_dataset_reddit_cleaned.csv")
BERT_MODEL_PATH = Path(__file__).with_name("saved_bert_model")


# Store tokenized posts and labels for standalone BERT training
class PostDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def main():
    # Load and clean the data using the same rules as clean_data.py
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset="clean_text", keep="first")

    # The post text is the input and the label is what we predict
    X = df["clean_text"]
    y = df["is_depression"]

    # Split the data into 20% of the posts for testing and 80% for training the model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Load a saved pre-trained BERT model when one is available
    saved_bert_exists = (BERT_MODEL_PATH / "config.json").exists()

    if saved_bert_exists:
        print("Loading saved BERT model...")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    else:
        print("No saved BERT model found. Training BERT for the first time...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            id2label={0: "0", 1: "1"},
            label2id={"0": 0, "1": 1},
        )

    # prepare tokenized posts for Hugging Face Trainer
    bert_train_data = PostDataset(X_train, y_train, tokenizer)
    bert_test_data = PostDataset(X_test, y_test, tokenizer)

    # Letet Trainer handle batching, optimization, and device selection

    training_arguments = TrainingArguments(
        output_dir=str(Path(__file__).with_name("bert_output")),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=False,
        dataloader_pin_memory=False,
        seed=42,
    )

    bert_trainer = Trainer(
        model=bert_model,
        args=training_arguments,
        train_dataset=bert_train_data,
        eval_dataset=bert_test_data,
    )

    # Train once, then save BERT so future runs can skip training
    if not saved_bert_exists:
        bert_trainer.train()
        bert_trainer.save_model(BERT_MODEL_PATH)
        tokenizer.save_pretrained(BERT_MODEL_PATH)
        print(f"Saved fine-tuned BERT model to: {BERT_MODEL_PATH}")

    bert_output = bert_trainer.predict(bert_test_data)
    bert_predictions = np.argmax(bert_output.predictions, axis=1)

    # Convert text into TF-IDF features, then train Logistic Regression
    logistic_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    logistic_model.fit(X_train, y_train)
    predictions = logistic_model.predict(X_test)

    # Train and evaluate Linear SVC using TF-IDF features
    linear_svc_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", LinearSVC()),
        ]
    )

    linear_svc_model.fit(X_train, y_train)
    svc_predictions = linear_svc_model.predict(X_test)

    # model results
    print("\nLogistic Regression results:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nLinear SVC results:")
    print(f"Accuracy: {accuracy_score(y_test, svc_predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, svc_predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, svc_predictions))

    print("\nBERT results:")
    print(f"Accuracy: {accuracy_score(y_test, bert_predictions):.2%}")

    print("\nClassification report:")
    print(classification_report(y_test, bert_predictions))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, bert_predictions))

    # user input on cli only for now
    new_post = input("\nEnter a post to classify: ")
    logistic_prediction = logistic_model.predict([new_post])[0]
    svc_prediction = linear_svc_model.predict([new_post])[0]

    # text-classification pipeline for the new post/text
    bert_classifier = pipeline(
        "text-classification",
        model=bert_trainer.model,
        tokenizer=tokenizer,
    )
    bert_result = bert_classifier(new_post)[0]
    bert_prediction = int(bert_result["label"])

    prediction_names = {
        0: "non-depression-associated language",
        1: "depression-associated language",
    }

    print(f"Logistic Regression prediction: {prediction_names[logistic_prediction]}")
    print(f"Linear SVC prediction: {prediction_names[svc_prediction]}")
    print(f"Standalone BERT prediction: {prediction_names[bert_prediction]}")


if __name__ == "__main__":
    main()
