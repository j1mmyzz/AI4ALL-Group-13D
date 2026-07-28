from pathlib import Path

import joblib
import streamlit as st
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

LOGISTIC_MODEL_PATH = Path(__file__).with_name("saved_logistic_model.joblib")

SVC_MODEL_PATH = Path(__file__).with_name("saved_svc_model.joblib")

BERT_MODEL_PATH = Path(__file__).with_name("saved_bert_model")


st.set_page_config(
    page_title="Depressive Text Detection",
    layout="wide",
)


@st.cache_resource
def load_models():
    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
    svc_model = joblib.load(SVC_MODEL_PATH)

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_PATH
    )
    bert_classifier = pipeline(
        "text-classification",
        model=bert_model,
        tokenizer=bert_tokenizer,
    )

    return logistic_model, svc_model, bert_classifier


logistic_model, svc_model, bert_classifier = load_models()


st.title("Depressive Text Detection")

st.warning("This project identifies language patterns and is not a medical diagnosis.")

user_text = st.text_area(
    "Enter text to classify",
    placeholder="Type or paste a post here...",
)

if st.button("Classify"):
    if not user_text.strip():
        st.error("Enter some text before classifying.")

    else:
        logistic_prediction = logistic_model.predict([user_text])[0]
        svc_prediction = svc_model.predict([user_text])[0]
        bert_result = bert_classifier(user_text)[0]
        bert_label = str(bert_result["label"])
        bert_prediction = int(bert_label.split("_")[-1])

        prediction_names = {
            0: "Non-depressive language",
            1: "Depression-associated language",
        }

        st.subheader("Predictions")

        column_1, column_2, column_3 = st.columns(3)

        with column_1:
            st.metric(
                "Logistic Regression",
                prediction_names[logistic_prediction],
            )

        with column_2:
            st.metric(
                "Linear SVC",
                prediction_names[svc_prediction],
            )

        with column_3:
            st.metric(
                "BERT",
                prediction_names[bert_prediction],
            )
