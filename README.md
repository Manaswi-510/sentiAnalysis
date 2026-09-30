# Sentiment Analysis

A machine learning project for classifying text into three sentiment categories: positive, negative, and neutral.

## Overview

This project uses Natural Language Processing (NLP) techniques to analyze the sentiment of a given text.

The model classifies text into:

- Positive
- Neutral
- Negative

The project uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features and an SVM (Support Vector Machine) classifier for sentiment prediction.

## Dataset

The dataset contains two main columns:

- `phrase` - the input text
- `sentiment` - the sentiment label

The sentiment labels are:

- `negative`
- `neutral`
- `positive`

The dataset is not included in this repository.

## Methodology

The project follows these steps:

1. Load the sentiment dataset.
2. Convert sentiment labels into numerical values.
3. Split the data into training and testing sets using an 80:20 ratio.
4. Convert text into numerical features using TF-IDF.
5. Train an SVM classifier on the training data.
6. Evaluate the trained model on the test data.
7. Use the trained model to predict the sentiment of new text.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- TF-IDF
- Support Vector Machine (SVM)

## Project Structure

```text
sentimentAnalysis/
│
├── data/
│   └── dataset files
│
├── sentimentAnalysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore