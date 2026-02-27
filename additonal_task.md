# Additional ML & LLM Tasks

---

# 1️⃣ How Would You Build a Spam Classification Model?

## Objective

Build a binary classifier to classify messages as Spam or Not Spam (Ham).

## Approach

### 1. Data Collection

- Use labeled spam/ham dataset.
- Ensure proper train-test split with stratification.

### 2. Text Preprocessing

- Lowercasing
- Removing punctuation/special characters
- Tokenization
- Stopword removal
- Optional lemmatization

### 3. Feature Engineering

#### Text-Based Features

- TF-IDF vectors
- N-grams (bi-grams / tri-grams)
- Message length
- Count of URLs
- Capital word ratio

#### Metadata Features (if available)

- Sender frequency
- Domain reputation
- Time patterns

### 4. Model Selection

#### Baseline

- Logistic Regression
- Naive Bayes

#### Advanced

- Fine-tuned Transformer (e.g., BERT)

### 5. Evaluation Metrics

- Precision
- Recall
- F1-score
- Confusion Matrix

Accuracy alone is not sufficient due to class imbalance.

### Production Considerations

- Tune probability threshold
- Monitor false positives
- Deploy as API endpoint
- Add model drift monitoring

---

# 2️⃣ What Features Would You Use in a CTR System?

CTR prediction is a probability estimation problem.

CTR = Clicks / Impressions

## User Features

- User ID (encoded)
- Historical user CTR
- Device type
- Location
- Time since last interaction
- Behavioral embeddings

## Ad Features

- Ad ID
- Ad category
- Historical ad CTR
- Campaign budget
- Advertiser quality score

## Context Features

- Time of day
- Day of week
- Platform (mobile/web)
- Page position

## Interaction Features (Critical)

- User × Ad historical interaction
- User-category affinity score
- Exposure frequency

## Feature Engineering Techniques

- Target encoding
- Crossed features
- Embeddings for high-cardinality IDs
- Normalized historical CTR

---

# 3️⃣ How Do You Handle Class Imbalance?

CTR datasets are typically highly imbalanced.

## Strategy 1: Use Proper Metrics

- AUC-ROC
- Log Loss
- Precision-Recall Curve

Avoid relying only on accuracy.

## Strategy 2: Class Weighting

Use class_weight="balanced" in models like Logistic Regression.

## Strategy 3: Resampling

- Oversampling (SMOTE)
- Undersampling
- Stratified train-test split

## Strategy 4: Threshold Tuning

Adjust decision threshold instead of default 0.5.

## Strategy 5: Boosting Methods

Gradient boosting models handle imbalance better.

---

# 4️⃣ LLM Prompt Tasks

## Extract Structured Data

You are an information extraction assistant.

Extract the following fields from the provided text:

- Name
- Date
- Organization
- Amount
- Email

Return output strictly in valid JSON format.

Text:
<Insert Document Here>

---

## Summarize a Legal Document

You are a legal assistant.

Summarize the following legal document in under 200 words.
Highlight:

- Key obligations
- Risks
- Deadlines
- Financial penalties

Maintain formal tone.

Document:
<Insert Legal Text>

---

## Detect Sentiment

You are a sentiment analysis model.

Classify the sentiment of the following text as:
Positive, Negative, or Neutral.

Also provide a confidence score between 0 and 1.

Text:
<Insert Text>

---

## Convert SQL → English

You are a data analyst assistant.

Explain the following SQL query in plain English.
Describe:

- What tables are used
- What filters are applied
- What aggregation is performed
- What the final result represents

SQL Query:
<Insert SQL Query>

---
