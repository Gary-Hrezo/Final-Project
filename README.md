This code contains the entire pipeline for authorship. It will show stats for supervised and unsupervised learning.
# Victorian Author Authorship Attribution Using Machine Learning

## Overview

This project explores authorship attribution using machine learning techniques. The goal is to determine the author of an unknown text passage by analyzing writing style instead of simply recognizing names or keywords.

The project uses novels from three nineteenth-century British authors:

* Jane Austen
* Charles Dickens
* George Eliot

Each novel is downloaded from Project Gutenberg, cleaned, divided into text chunks, and converted into numerical features using TF-IDF. Several supervised and unsupervised machine learning algorithms are then used to compare their performance.

---

## Objectives

The objectives of this project are to:

* Build a machine learning model that predicts the author of a text passage.
* Compare several supervised classification algorithms.
* Compare supervised learning with unsupervised clustering.
* Evaluate model performance using common machine learning metrics.

---

## Dataset

The books were obtained from Project Gutenberg.

### Jane Austen

* Emma
* Pride and Prejudice
* Sense and Sensibility

### Charles Dickens

* Oliver Twist
* A Tale of Two Cities
* Great Expectations

### George Eliot

* Middlemarch
* The Mill on the Floss
* Silas Marner

---

## Methods

### Data Preparation

The text is preprocessed by:

* Removing Project Gutenberg headers and footers
* Converting all text to lowercase
* Removing punctuation and numbers
* Removing proper nouns to reduce data leakage
* Splitting each novel into equal-sized text chunks

### Feature Extraction

* TF-IDF Vectorization
* Stylometric features including:

  * Function word density
  * Lexical diversity
  * Average word length
  * Comma usage
  * Sentence punctuation

### Supervised Learning

The following classifiers were evaluated:

* Logistic Regression
* Complement Naive Bayes
* Support Vector Machine (Linear SVM)

### Unsupervised Learning

The following clustering algorithms were evaluated:

* K-Means
* Agglomerative Clustering
* DBSCAN

---

## Evaluation

Classification performance was measured using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Clustering performance was measured using:

* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)
* Silhouette Score

To prevent data leakage, entire books were held out for testing rather than randomly splitting text chunks.

---

## Project Structure

```
Final-Project/
│
├── author_ml_pipeline_student_style.py
├── README.md
├── requirements.txt
│
├── gutenberg_books/
│
├── output/
│   ├── author_v2_main.png
│   ├── author_v2_cv.png
│   ├── author_v2_features.png
│   └── author_v2_dendrogram.png
│
└── report/
```

---

## Required Python Libraries

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
nltk
```

Install them using:

```
pip install -r requirements.txt
```

---

## Running the Project

1. Download the Project Gutenberg books.
2. Place the books in the `gutenberg_books` folder.
3. Run:

```
python author_ml_pipeline_student_style.py
```

The program trains the models, evaluates performance, and generates several plots.

---

## Results

The supervised classifiers consistently outperformed the clustering algorithms. Logistic Regression and Linear SVM produced the highest classification accuracy, while K-Means and Agglomerative Clustering discovered meaningful stylistic groupings without using author labels.

---

## Future Improvements

Possible improvements include:

* Testing additional authors
* Using transformer-based language models
* Comparing TF-IDF with word embeddings
* Increasing the size of the training corpus

---

## Author

Gary Hrezo

Indiana University
Graduate Certificate in Artificial Intelligence
