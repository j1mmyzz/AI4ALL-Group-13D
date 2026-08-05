import os
import math
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

AVERAGE_CONFIDENCE = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Linear SVC", "BERT"],
        "Average confidence": [84.70, 74.66, 99.70],
    }
).set_index("Model")


st.set_page_config(
    page_title="Depressive Text Detection",
    layout="wide",
)


def apply_reddit_style():
    st.markdown(
        """
        <style>
        :root {
            --reddit-orange: #ff4500;
            --reddit-blue: #0079d3;
            --reddit-navy: #1a1a1b;
            --reddit-background: #f2f4f5;
            --reddit-border: #d7d9dc;
            --reddit-muted: #787c7e;
        }

        .stApp {
            background-color: var(--reddit-background);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            margin-top: 1.5rem;
            margin-bottom: 2rem;
            padding: 2rem 2.25rem 2.5rem;
            background-color: #ffffff;
            border: 1px solid var(--reddit-border);
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }

        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4 {
            color: var(--reddit-navy);
        }

        [data-testid="stMain"] h2 {
            margin-top: 2.25rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #edeff1;
        }

        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid var(--reddit-border);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 999px;
            padding: 0.35rem 0.6rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background-color: #f6f7f8;
            color: var(--reddit-blue);
        }

        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"] {
            background-color: #fdfdfd;
            border: 1px solid var(--reddit-border);
            border-radius: 8px;
            padding: 0.85rem;
        }

        div[data-testid="stVegaLiteChart"] {
            overflow: visible;
            padding: 0.25rem 0.25rem 1.5rem;
        }

        div[data-testid="stMetric"] {
            border-top: 4px solid var(--reddit-orange);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--reddit-blue);
            font-weight: 700;
        }

        .stButton > button {
            background-color: var(--reddit-orange);
            border: 1px solid var(--reddit-orange);
            border-radius: 999px;
            color: #ffffff;
            font-weight: 700;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        .stButton > button:hover {
            background-color: #e03d00;
            border-color: #e03d00;
            color: #ffffff;
        }

        .stTextArea textarea {
            background-color: #ffffff;
            border: 1px solid #878a8c;
            border-radius: 8px;
        }

        .stTextArea textarea:focus {
            border-color: var(--reddit-blue);
            box-shadow: 0 0 0 1px var(--reddit-blue);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        .reddit-page-header {
            margin: -0.25rem 0 1.5rem;
            padding-bottom: 1.15rem;
            border-bottom: 1px solid #edeff1;
        }

        .reddit-page-header h1 {
            margin: 0;
            color: var(--reddit-navy);
            font-size: 1.85rem;
            line-height: 1.2;
        }

        .reddit-page-header p {
            margin: 0.2rem 0 0;
            color: var(--reddit-muted);
            font-size: 0.88rem;
        }

        hr {
            border-color: var(--reddit-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_reddit_style()


def show_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="reddit-page-header">
            <div>
                <h1>{title}</h1>
                <p>r/depressive_text_detection &nbsp;·&nbsp; {subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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


def get_linear_influences(model, text, prediction, limit=5):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    text_features = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    contributions = text_features.multiply(classifier.coef_[0]).toarray()[0]

    if prediction == 1:
        matching_features = [
            (feature_names[index], contribution)
            for index, contribution in enumerate(contributions)
            if contribution > 0
        ]
    else:
        matching_features = [
            (feature_names[index], abs(contribution))
            for index, contribution in enumerate(contributions)
            if contribution < 0
        ]

    matching_features.sort(key=lambda item: item[1], reverse=True)
    return [feature for feature, _ in matching_features[:limit]]


def get_bert_influences(
    bert_classifier,
    text,
    predicted_label,
    original_confidence,
    limit=5,
    max_words=30,
):
    words = text.split()
    words_to_test = min(len(words), max_words)
    mask_token = bert_classifier.tokenizer.mask_token
    masked_texts = []

    for index in range(words_to_test):
        changed_words = words.copy()
        if mask_token:
            changed_words[index] = mask_token
        else:
            changed_words.pop(index)
        masked_texts.append(" ".join(changed_words))

    if not masked_texts:
        return []

    masked_results = bert_classifier(
        masked_texts,
        top_k=None,
        batch_size=16,
        truncation=True,
        max_length=128,
    )

    word_impacts = {}
    for index, result in enumerate(masked_results):
        result_list = result if isinstance(result, list) else [result]
        masked_score = next(
            (
                float(label_result["score"])
                for label_result in result_list
                if str(label_result["label"]) == predicted_label
            ),
            original_confidence,
        )
        impact = original_confidence - masked_score
        word = words[index].strip(".,!?;:\"'()[]{}").lower()

        if word and impact > word_impacts.get(word, 0):
            word_impacts[word] = impact

    ranked_words = sorted(
        word_impacts.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [word for word, impact in ranked_words if impact > 0][:limit]


def show_classifier_page():
    show_page_header(
        "Depressive Text Detection",
        "Classify one post and inspect the prediction",
    )
    st.write("Compare predictions from Logistic Regression, Linear SVC, and BERT.")
    st.warning(
        "This project identifies language patterns and is not a medical diagnosis."
    )

    try:
        logistic_model, svc_model, bert_classifier = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "The Results and Documentation pages are still available from the sidebar."
        )
        st.stop()
    except ModuleNotFoundError as error:
        st.error(
            f"The classifier could not load because {error.name!r} is not installed in the Python environment running Streamlit."
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
            logistic_probabilities = logistic_model.predict_proba([user_text])[0]
            logistic_confidence = float(max(logistic_probabilities))

            svc_prediction = int(svc_model.predict([user_text])[0])
            if hasattr(svc_model, "predict_proba"):
                svc_probabilities = svc_model.predict_proba([user_text])[0]
                svc_confidence = float(max(svc_probabilities))
                svc_confidence_is_calibrated = True
            else:
                svc_decision_score = float(svc_model.decision_function([user_text])[0])
                svc_confidence = 1 / (1 + math.exp(-min(abs(svc_decision_score), 709)))
                svc_confidence_is_calibrated = False

            bert_result = bert_classifier(
                user_text,
                truncation=True,
                max_length=128,
            )[0]
            bert_label = str(bert_result["label"])
            bert_prediction = int(bert_label.split("_")[-1])
            bert_confidence = float(bert_result["score"])

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
                st.caption(f"Model confidence: {logistic_confidence:.2%}")

            with column_2:
                st.metric(
                    "Linear SVC",
                    prediction_names[svc_prediction],
                )
                st.caption(f"Model confidence: {svc_confidence:.2%}")

            with column_3:
                st.metric(
                    "BERT",
                    prediction_names[bert_prediction],
                )
                st.caption(f"Model confidence: {bert_confidence:.2%}")

            st.info(
                "Confidence shows how strongly a model favors its prediction. "
                "It is not the probability that a person has depression."
            )
            if not svc_confidence_is_calibrated:
                st.caption(
                    "Linear SVC confidence is an uncalibrated score derived "
                    "from its distance to the decision boundary."
                )

            logistic_influences = get_linear_influences(
                logistic_model,
                user_text,
                logistic_prediction,
            )
            svc_influences = get_linear_influences(
                svc_model,
                user_text,
                svc_prediction,
            )
            with st.spinner("Estimating BERT word influences..."):
                bert_influences = get_bert_influences(
                    bert_classifier,
                    user_text,
                    bert_label,
                    bert_confidence,
                )

            st.subheader("Influential Words or Phrases")
            st.write(
                "These features pushed each model toward the prediction " "shown above."
            )
            explanation_columns = st.columns(3)
            explanations = [
                ("Logistic Regression", logistic_influences),
                ("Linear SVC", svc_influences),
                ("BERT estimate", bert_influences),
            ]

            for column, (model_name, influences) in zip(
                explanation_columns,
                explanations,
            ):
                with column:
                    st.markdown(f"#### {model_name}")
                    if influences:
                        st.write(", ".join(influences))
                    else:
                        st.write(
                            "No strong influential features were found in "
                            "this input."
                        )

            st.caption(
                "The linear-model explanations use TF-IDF feature weights. "
                "The BERT explanation is an estimate based on masking one "
                "word at a time. These are learned associations, not "
                "clinical explanations or diagnoses."
            )


def show_results_page():
    show_page_header(
        "Model Results",
        "Overall evaluation on held-out test data",
    )
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
            height=380,
        )

    with precision_column:
        st.markdown("#### Precision")
        st.bar_chart(
            MODEL_RESULTS[["Precision"]],
            y_label="Precision (%)",
            x_label="Model",
            height=380,
        )

    recall_column, f1_column = st.columns(2)

    with recall_column:
        st.markdown("#### Recall")
        st.bar_chart(
            MODEL_RESULTS[["Recall"]],
            y_label="Recall (%)",
            x_label="Model",
            height=380,
        )

    with f1_column:
        st.markdown("#### F1-score")
        st.bar_chart(
            MODEL_RESULTS[["F1-score"]],
            y_label="F1-score (%)",
            x_label="Model",
            height=380,
        )

    st.subheader("Average Model Confidence")
    st.write(
        "This graph shows each model's average confidence across the 1,530 "
        "test posts."
    )
    st.bar_chart(
        AVERAGE_CONFIDENCE,
        y=["Average confidence"],
        y_label="Average confidence (%)",
        x_label="Model",
        height=380,
    )
    st.caption(
        "Confidence is not the same as accuracy. BERT's high average "
        "confidence does not mean it is correct 99.70% of the time. Linear "
        "SVC uses an uncalibrated score based on distance from its decision "
        "boundary, so these values should not be directly compared as true "
        "probabilities."
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
    show_page_header(
        "Documentation",
        "Purpose, methods, limitations, and responsible use",
    )

    st.header("Project Purpose")
    st.write(
        "This AI4ALL Ignite project explores whether natural language "
        "processing models can distinguish depression-associated Reddit "
        "posts from non-depressive posts based on their text."
    )

    st.header("Real-World Use Case")
    st.write(
        "Imagine you're a social scientist, researcher, or social worker "
        "trying to understand current discussions around mental health. "
        "There are thousands of Reddit posts every day, making it "
        "unrealistic to manually read everything. Instead of reading every "
        "post, the classifier automatically flags posts containing "
        "depression-associated language so users can prioritize which "
        "discussions to examine first."
    )

    st.header("What the Labels Mean")
    st.markdown("""
- **Non-depressive language:** The post is more similar to examples labeled
  `0` in the Reddit training dataset.
- **Depression-associated language:** The post contains patterns that are more
  similar to examples labeled `1` in the training dataset.

These labels describe similarities to the dataset. They do not determine the
author's emotional state or whether the author has depression. Positive,
sarcastic, quoted, or supportive posts can still be misclassified when context
is missing.
""")

    st.header("Prediction and Evaluation Are Different")
    st.markdown("""
- **Classifier page:** Shows what each model predicts for one post, its
  confidence score, and influential words or phrases.
- **Results page:** Shows how the models performed across all 1,530 held-out
  test posts using accuracy, precision, recall, F1-score, and confusion
  matrices.

A confidence score belongs to one prediction. The evaluation metrics describe
overall test-set performance and should not be interpreted as confidence in a
single post.
""")

    st.header("Why Compare Three Models")
    st.write(
        "Logistic Regression provides a simple, interpretable baseline, while "
        "Linear SVC is effective for high-dimensional TF-IDF text features. "
        "BERT captures more contextual information and achieved the highest "
        "test accuracy. The classifier shows all three so users can inspect "
        "whether the models agree, while the Results page explains why BERT "
        "was the strongest overall model."
    )

    st.header("What Happens After Classification")
    st.write(
        "The app reports the predicted language category, a model-confidence "
        "score, and influential words or phrases. These explanations show "
        "patterns the models learned from the Reddit dataset. They are not "
        "clinical explanations, and a flagged post should only be treated as "
        "a possible item for further human review."
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

    st.header("Confidence Scores")
    st.markdown("""
Confidence describes how strongly a model favors its prediction for one
specific input. It is separate from accuracy, precision, recall, and F1-score,
which evaluate performance across the full test set.

- **Logistic Regression:** Uses the highest probability returned by
  predict_proba().
- **Linear SVC:** Uses a sigmoid-scaled distance from the decision boundary.
  This is an uncalibrated confidence score, not a true probability.
- **BERT:** Uses softmax to convert the model's output scores into values
  between 0 and 1.

Confidence values are calculated differently for each model and should not be
directly compared. A high-confidence prediction can still be incorrect, and
confidence does not represent the probability that a person has depression.
""")

    st.header("Prediction Explanations")
    st.markdown("""
After classifying an input, the app displays words or phrases that influenced
each model's selected label.

- **Logistic Regression and Linear SVC:** The explanation combines each
  TF-IDF feature value with the coefficient learned by the classifier.
- **BERT:** The app masks one word at a time and measures how much the score
  for BERT's selected label decreases. For responsiveness, it tests up to the
  first 30 words.

These explanations show associations learned from the Reddit training data.
They do not prove why a person wrote something and are not clinical
explanations or diagnoses.
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
    st.write("Jimmy Zheng, Fajar Alim, and Shreeshkumar Lillyprabhu")


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
