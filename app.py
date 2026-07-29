import os
from pathlib import Path

import pandas as pd
import streamlit as st

LOGISTIC_MODEL_PATH = Path(__file__).with_name("saved_logistic_model.joblib")
SVC_MODEL_PATH = Path(__file__).with_name("saved_svc_model.joblib")
BERT_MODEL_PATH = Path(__file__).with_name("saved_bert_model")

MODEL_RESULTS = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Linear SVC", "BERT"],
        "Accuracy": [94.51, 95.16, 98.43],
        "Precision": [94.50, 95.50, 98.50],
        "Recall": [94.50, 95.00, 98.50],
        "F1-score": [94.50, 95.00, 98.50],
    }
).set_index("Model")

CONFUSION_MATRICES = {
    "Logistic Regression": [[747, 31], [53, 699]],
    "Linear SVC": [[747, 31], [43, 709]],
    "BERT": [[771, 7], [17, 735]],
}


st.set_page_config(
    page_title="Depressive Text Detection",
    layout="wide",
)


@st.cache_resource
def load_models():
    # Import Transformers only when the classifier page needs BERT.
    import joblib
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    # because the saved BERT model is too big for github and would cause loading issues anyway on streamlit
    if BERT_MODEL_PATH.is_dir():
        bert_model_source = str(BERT_MODEL_PATH)
    else:
        bert_model_source = os.getenv("BERT_MODEL_ID")

    if not bert_model_source:
        raise FileNotFoundError(
            "The saved BERT model is not included in the deployed app. "
            "Upload it to Hugging Face and add its repository name as the "
            "Streamlit secret BERT_MODEL_ID."
        )

    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
    svc_model = joblib.load(SVC_MODEL_PATH)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_source)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_source)
    bert_classifier = pipeline(
        "text-classification",
        model=bert_model,
        tokenizer=bert_tokenizer,
    )

    return logistic_model, svc_model, bert_classifier


def show_classifier_page():
    st.title("Depressive Text Detection")
    st.write("Compare predictions from Logistic Regression, Linear SVC, and BERT.")
    st.warning(
        "This project identifies language patterns and is not a medical diagnosis."
    )

    try:
        logistic_model, svc_model, bert_classifier = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "The Results and Documentation pages are still available from "
            "the sidebar."
        )
        st.stop()
    except ModuleNotFoundError as error:
        st.error(
            f"The classifier could not load because {error.name!r} is not "
            "installed in the Python environment running Streamlit."
        )
        st.code("python3 -m pip install transformers torch joblib")
        st.stop()
    except OSError as error:
        st.error(f"A saved model could not be loaded: {error}")
        st.stop()

    user_text = st.text_area(
        "Enter text to classify",
        placeholder="Type or paste a post here...",
    )

    if st.button("Classify", type="primary"):
        if not user_text.strip():
            st.error("Enter some text before classifying.")
        else:
            logistic_prediction = int(logistic_model.predict([user_text])[0])
            svc_prediction = int(svc_model.predict([user_text])[0])
            bert_result = bert_classifier(user_text)[0]
            bert_label = str(bert_result["label"])
            bert_prediction = int(bert_label.split("_")[-1])

            prediction_names = {
                0: "Non-depressive language",
                1: "Depression-associated language",
            }

            st.subheader(
                "Predictions (Non-depressive language or Depression-associated language)"
            )
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


def show_results_page():
    st.title("Model Results")
    st.write(
        "All three models were evaluated on the same held-out test set of "
        "1,530 posts."
    )

    st.subheader("Metric Comparisons")
    st.write("Each chart compares one metric across the three models.")

    accuracy_column, precision_column = st.columns(2)

    with accuracy_column:
        st.markdown("#### Accuracy")
        st.bar_chart(
            MODEL_RESULTS[["Accuracy"]],
            y_label="Accuracy (%)",
            x_label="Model",
        )

    with precision_column:
        st.markdown("#### Precision")
        st.bar_chart(
            MODEL_RESULTS[["Precision"]],
            y_label="Precision (%)",
            x_label="Model",
        )

    recall_column, f1_column = st.columns(2)

    with recall_column:
        st.markdown("#### Recall")
        st.bar_chart(
            MODEL_RESULTS[["Recall"]],
            y_label="Recall (%)",
            x_label="Model",
        )

    with f1_column:
        st.markdown("#### F1-score")
        st.bar_chart(
            MODEL_RESULTS[["F1-score"]],
            y_label="F1-score (%)",
            x_label="Model",
        )

    st.subheader("Complete Results")
    formatted_results = MODEL_RESULTS.map(lambda score: f"{score:.2f}%")
    st.dataframe(formatted_results, use_container_width=True)

    st.subheader("Confusion Matrices")
    st.write(
        "Rows show the actual label and columns show the model's predicted "
        "label. Correct predictions are on the main diagonal."
    )

    matrix_columns = st.columns(3)
    for column, (model_name, values) in zip(
        matrix_columns,
        CONFUSION_MATRICES.items(),
    ):
        matrix = pd.DataFrame(
            values,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )

        with column:
            st.markdown(f"#### {model_name}")
            st.dataframe(
                matrix,
                use_container_width=True,
            )

    st.caption(
        "Label 0 means non-depressive text. Label 1 means "
        "depression-associated text."
    )

    st.subheader("Interpretation")
    st.write(
        "BERT had the highest accuracy, precision, recall, and F1-score. "
        "Linear SVC performed slightly better than Logistic Regression. "
        "These scores describe performance on this dataset and do not mean "
        "that any model can diagnose depression."
    )

    st.caption(
        "Accuracy is the percentage of all correct predictions. Precision "
        "measures how often a positive prediction was correct. Recall "
        "measures how many positive examples were found. F1-score balances "
        "precision and recall."
    )


def show_documentation_page():
    st.title("Documentation")

    st.header("Project Purpose")
    st.write(
        "This AI4ALL Ignite project explores whether natural language "
        "processing models can distinguish depression-associated Reddit "
        "posts from non-depressive posts based on their text."
    )

    st.header("Dataset")
    st.markdown("""
- Source: [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)
- Original dataset: 7,731 English-language Reddit posts
- Removed duplicates: 81 posts
- Cleaned dataset: 7,650 unique posts
- Label 0: non-depressive text
- Label 1: depression-associated text
- Training set: 6,120 posts
- Test set: 1,530 posts
""")

    st.header("Project Workflow")
    st.markdown("""
1. Load and audit the dataset.
2. Remove duplicate text entries.
3. Preserve the dataset's binary labels.
4. Create a stratified 80/20 training and test split.
5. Train Logistic Regression and Linear SVC with TF-IDF features.
6. Train BERT model as a binary text classifier.
7. Evaluate all models on the same test set.
8. Use the saved models to classify new text in this app.
""")

    st.header("Models")
    st.markdown("""
**Logistic Regression:** A linear classifier trained on TF-IDF text features.

**Linear SVC:** A support vector classifier trained on TF-IDF text features.

**BERT:** A transformer model trained directly on the labeled Reddit posts.
""")

    st.header("Responsible Use")
    st.write(
        "The predictions identify language patterns found in the training "
        "data. They are not medical advice and must not be used to diagnose "
        "depression or make decisions about a person's mental health."
    )
    st.write(
        "Results may be affected by mislabeled examples, social-media writing "
        "styles, demographic and cultural differences, and limitations in "
        "the source dataset. A prediction can be wrong even when the models "
        "agree."
    )

    st.header("Team")
    st.write(
        "Jimmy Zheng, Fajar Alim, Kaila Grant, Addishiwot Dagnew, "
        "Sahasra Bobbala, Shreesh LillyPrabhu, and Varija Manglik"
    )


page = st.sidebar.radio(
    "Navigation",
    ["Classifier", "Results", "Documentation"],
)

if page == "Classifier":
    show_classifier_page()
elif page == "Results":
    show_results_page()
else:
    show_documentation_page()
