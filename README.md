# AI4ALL Group 13D – Depressive Text Detection Trained On Reddit Posts

A Natural Language Processing (NLP) project that detects depressive language patterns in Reddit posts using TF-IDF and machine learning classifiers.

## Project Overview

This project was completed as part of the **AI4ALL Ignite Summer 2026** program.

Our goal is to develop/train a machine learning model capable of identifying depressive language patterns from Reddit posts. The project applies Natural Language Processing (NLP) techniques to classify posts while emphasizing Responsible AI principles such as fairness, transparency, and reproducibility.

---

## Team Members

- Jimmy Zheng
- Fajar Alim
- Kaila Grant
- Addishiwot Dagnew
- Sahasra Bobbala
- Shreesh LillyPrabhu
- Varija Manglik

---

## Problem Statement

Depression is a growing global mental health concern, and many individuals express their emotions through social media before seeking professional support.

This project explores whether machine learning models can identify depressive language patterns in Reddit posts to assist researchers and mental health professionals. The model is intended for research and educational purposes only and **is not designed to diagnose depression**.

---

## Dataset

Dataset: ["Depression: Reddit Dataset (Cleaned)"](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)

- Approximately ~7,000 Reddit posts
- Binary labels:
  - 0 = Non-depressive
  - 1 = Depressive
- Used for supervised machine learning classification.

---

## Project Workflow

1. Load the dataset
2. Perform a basic data audit
3. Clean and preprocess text
4. Split the dataset into training and testing sets (80/20)
5. Convert text into numerical features using TF-IDF
6. Train multiple machine learning models
7. Evaluate model performance
8. Compare model accuracy
9. Select the best-performing model

---

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Linear Support Vector Classifier (Linear SVC)
- Bidirectional Encoder Representations from Transformers (BERT)

---

## Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   94.51% |    94.50% | 94.50% |   94.50% |
| Linear SVC          |   95.16% |    95.50% | 95.00% |   95.00% |
| BERT                |   98.43% |    98.50% | 98.50% |   98.50% |

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**BERT** achieved the highest score on all metrics.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Linear SVC
- Random Forest
- Kaggle Notebooks
- GitHub

---

## Acknowledgements

This project was completed as part of the AI4ALL Ignite Summer 2026 program. We thank our instructors, mentors, and teammates for their support throughout the project.
