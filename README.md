# AI4ALL Group 13D – Depression Detection from Twitter Posts
A Natural Language Processing (NLP) project that detects depressive language patterns in Twitter (X) posts using TF-IDF and machine learning classifiers.

## Project Overview

This project was completed as part of the **AI4ALL Ignite Summer 2026** program.

Our goal is to develop a machine learning model capable of identifying depressive language patterns from Twitter (X) posts. The project applies Natural Language Processing (NLP) techniques to classify posts while emphasizing Responsible AI principles such as fairness, transparency, and reproducibility.

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

This project explores whether machine learning models can identify depressive language patterns in Twitter posts to assist researchers and mental health professionals. The model is intended for research and educational purposes only and **is not designed to diagnose depression**.

---

## Dataset

Dataset: Mental Health Twitter Dataset

- Approximately 20,000 Twitter posts
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
5. Convert text into numerical features using **TF-IDF**
6. Train multiple machine learning models
7. Evaluate model performance
8. Compare model accuracy
9. Select the best-performing model

---

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Linear Support Vector Classifier (Linear SVC)
- Random Forest Classifier

---

## Results

| Model | Accuracy |
|-------|---------:|
| Logistic Regression | **75.60%** |
| Linear SVC | 75.55% |
| Random Forest | 73.81% |

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Selected Model

**Logistic Regression**

It achieved the highest overall accuracy while remaining simple, interpretable, and computationally efficient.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Kaggle Notebooks
- GitHub

---

## Repository Structure

```
AI4ALL-Group-13D/
│
├── clean_data.py              # Data cleaning and preprocessing
├── models.py                  # Machine learning models
├── Mental-Health-Twitter.csv  # Dataset
├── README.md
```

---

## Responsible AI Considerations

Our project incorporates Responsible AI principles by:

- Cleaning and auditing the dataset before training
- Comparing multiple machine learning models
- Using transparent evaluation metrics
- Selecting an interpretable model
- Recognizing dataset limitations and potential bias
- Emphasizing that the model supports research rather than clinical diagnosis

---
## How to Run

1. Clone the repository.
2. Open the notebook in Kaggle.
3. Run all notebook cells from top to bottom.
4. Review the model evaluation and comparison results.


## Future Improvements

Potential improvements include:

- Fine-tuning hyperparameters
- Testing transformer-based models (e.g., BERT)
- Expanding the dataset
- Building a Streamlit web application
- Improving fairness evaluation across different demographic groups
- Deploying the model using cloud services

## Acknowledgements

This project was completed as part of the AI4ALL Ignite Summer 2026 program. We thank our instructors, mentors, and teammates for their support throughout the project.

## License

This repository was created for educational purposes as part of the **AI4ALL Ignite Summer 2026** program.
