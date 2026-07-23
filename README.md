# Depressive Text Detection Trained on Reddit Posts

Built and compared three natural language processing models that classify depression-associated language in Reddit posts. This project applied Python, text preprocessing, TF-IDF, supervised machine learning, and transformer fine-tuning as part of the AI4ALL Ignite Summer 2026 program.

## Problem Statement <!--- do not change this line -->

Depression is a widespread mental health concern, and people often discuss difficult emotions through social media. This project examines whether machine learning models can distinguish depression-associated Reddit posts from non-depressive posts based on their text.

The project is intended for research and education. Its predictions describe language patterns found in the training data and should not be treated as a depression diagnosis. Errors, cultural differences, and bias in the source data could affect how the models classify new writing.

## Key Results <!--- do not change this line -->

1. Audited and cleaned a binary dataset containing 7,731 English-language Reddit posts.
   - Removed 81 duplicate text entries.
   - Retained 7,650 unique posts.
   - Used 3,889 non-depressive posts and 3,761 depression-associated posts.
2. Split the cleaned data into 80% training data and 20% testing data.
   - Training set: 6,120 posts
   - Test set: 1,530 posts
3. Trained and evaluated three text-classification approaches.

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 94.51% | 94.50% | 94.50% | 94.50% |
| Linear SVC | 95.16% | 95.50% | 95.00% | 95.00% |
| BERT | 98.43% | 98.50% | 98.50% | 98.50% |

4. BERT achieved the highest overall performance on the held-out test set.
5. Evaluated each model with accuracy, precision, recall, F1-score, and a confusion matrix.

## Methodologies <!--- do not change this line -->

We loaded and audited the Reddit dataset with pandas, checked its structure and class distribution, and removed duplicate text entries. We preserved the original binary labels, where `0` represents non-depressive text and `1` represents depression-associated text.

We used a stratified 80/20 split so the training and test sets maintained similar class proportions. Logistic Regression and Linear SVC received TF-IDF features generated from the cleaned post text. For the third approach, we fine-tuned `bert-base-uncased` as a binary sequence classifier with Hugging Face Transformers and PyTorch.

After training, we compared each model on the same held-out test set. The program also accepts a new text input and returns each model's predicted language category. Because social-media labels are noisy and do not represent clinical assessments, the output is described as depression-associated or non-depressive language rather than a diagnosis.

## Data Sources <!--- do not change this line -->

- Kaggle: [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)
- The downloaded CSV contains cleaned English-language text and a binary `is_depression` label.
- Kaggle states that the raw text was collected by scraping subreddits and then cleaned with NLP techniques.

## Technologies Used <!--- do not change this line -->

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Linear SVC
- PyTorch
- Hugging Face Transformers
- BERT (`bert-base-uncased`)
- Git and GitHub
- Kaggle

## Authors <!--- do not change this line -->

This project was completed as part of AI4ALL Ignite Summer 2026 by:

- Jimmy Zheng
- Fajar Alim
- Kaila Grant
- Addishiwot Dagnew
- Sahasra Bobbala
- Shreesh LillyPrabhu
- Varija Manglik
