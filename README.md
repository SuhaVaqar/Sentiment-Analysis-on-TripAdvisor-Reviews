# Sentiment Analysis on TripAdvisor Reviews

This repository contains a sentiment analysis project based on reviews from TripAdvisor's website. The project aims to classify reviews as either positive or negative, using various preprocessing steps and machine learning models. The key focus is on managing class imbalance and optimizing the model’s performance through preprocessing, feature extraction, and hyperparameter tuning.


# Project Overview

This project uses a Tf-idf vectorizer for text feature extraction and explores various classification models, including:

- **Random Forest**
- **Naive Bayes** (Multinomial and Complement)
- **Linear Support Vector Classifier** (LinearSVC)
- **K-Nearest Neighbors** (KNN)

Through thorough hyperparameter tuning and cross-validation, the best-performing model, **LinearSVC**, achieves an accuracy of **97.03%** on the development set, and **97.55%** on unseen data.

## Dataset

- **Total Reviews**: 41,077
- **Development Set**: 28,754 reviews

### Class Distribution:

- **Positive**: 19,532 (68%)
- **Negative**: 9,222 (32%)

## Data Exploration

Key insights from the exploratory data analysis:

- **Review Length**: Positive reviews tend to be longer (mean: 701 characters) compared to negative reviews (mean: 624 characters).
- **Frequent Word Analysis**: Using the fpgrowth algorithm, frequent words and bigrams were extracted, and sentiment-related emojis were examined.
- **Language Detection**: The vast majority of reviews are in English.

## Preprocessing

The preprocessing pipeline includes:

- **Tf-idf Vectorization**: Converts the text data into numerical features, scaling word counts based on their importance.
- **Stop Word Removal**: Common Italian words like "of," "the," "that" were removed.
- **Stemming**: Porter Stemmer was applied to reduce words to their root forms.
- **Emoji Processing**: Emojis were handled based on sentiment, with positive/negative emojis replaced by placeholder strings.
- **Normalization**: Each document vector is normalized to L2 norm=1.

## Model Selection

Several classifiers were tested:

- **Random Forest**: Medium accuracy and feature interpretability.
- **Multinomial and Complement Naive Bayes**: Fast but less accurate.
- **LinearSVC**: Highest accuracy, robust to high-dimensional data.
- **K-Nearest Neighbors**: Struggles due to the curse of dimensionality.

After initial testing, **LinearSVC** was selected as the best model due to its high performance on the development set.

## Model Tuning and Validation

Hyperparameters were tuned via grid search on the following:

- Minimum word length
- Stop words removal
- Stemming
- Emojis preprocessing
- N-gram range (unigrams & bigrams)
- Tf-idf parameters (max_df, min_df)
- SVC parameters (loss, tolerance, C regularization)

### The Best Configuration:

- **Stop words removal**: Yes (with whitelisted common words like "not," "but")
- **Stemming**: Yes
- **Emojis Preprocessing**: Yes
- **N-gram Range**: Unigrams & Bigrams
- **SVC Loss**: Hinge
- **C (Regularization)**: 3.0

## Results

The final model achieved the following scores:

- **Development Set Accuracy**: 97.03%
- **Unseen Data Accuracy**: 97.55%

The results demonstrated that the model was able to generalize well on new data.

