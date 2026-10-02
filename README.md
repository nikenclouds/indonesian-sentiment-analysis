# Indonesian Sentiment Analysis

## Project Overview

This project focuses on Indonesian text sentiment analysis using Natural Language Processing (NLP) and machine learning techniques.

The project compares three traditional machine learning algorithms with an Indonesian Transformer-based model:

- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- IndoBERT-Lite

## Dataset

The dataset contains 11,000 Indonesian text samples with three sentiment classes:

- Positive
- Negative
- Neutral

## NLP Pipeline

The research pipeline consists of:

1. Data Loading
2. Exploratory Data Analysis
3. Case Folding
4. Text Cleaning
5. Stopword Removal
6. Stemming using Sastrawi
7. Train-Test Split
8. TF-IDF Feature Extraction
9. Machine Learning Classification
10. IndoBERT-Lite Fine-Tuning
11. Model Evaluation

## Machine Learning Models

### 1. Naive Bayes

Multinomial Naive Bayes is used as one of the baseline classification models.

### 2. Logistic Regression

Logistic Regression is used for multiclass sentiment classification based on TF-IDF features.

### 3. Support Vector Machine

Linear SVM is used to classify Indonesian text based on TF-IDF representations.

### 4. IndoBERT-Lite

IndoBERT-Lite is used as a Transformer-based model for Indonesian sentiment classification.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 81.95% | 85.00% | 69.74% | 73.79% |
| Logistic Regression | 86.14% | 85.97% | 79.49% | 82.07% |
| SVM | 85.73% | 84.28% | 81.10% | 82.55% |
| IndoBERT-Lite | 67.64% | 42.05% | 46.12% | 43.78% |

> Note: IndoBERT-Lite was fine-tuned for one epoch with a maximum sequence length of 64 tokens. The results should therefore be interpreted as an experimental benchmark under the specified training configuration.

## Project Structure

```text
indonesian-sentiment-analysis/
│
├── Data/
│
├── Src/
│   ├── 01_load_dataset.py
│   ├── 02_load_training_dataset.py
│   ├── 03_text_preprocessing.py
│   ├── 04_clean_text.py
│   ├── 05_split_data.py
│   ├── 06_tfidf.py
│   ├── 07_naive_bayes.py
│   ├── 08_confusion_matrix.py
│   ├── 09_logistic_regression.py
│   ├── 10_logistic_confusion_matrix.py
│   ├── 11_svm.py
│   ├── 12_model_comparison.py
│   └── 13_class_f1_comparison.py
│
├── Results/
│
├── NLP_Pipeline.ipynb
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
├── README.md
└── .gitignore
