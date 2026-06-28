# Victorian Author Machine Learning Project
# This version uses a book-level train/test split so the model is tested on books
# it did not see during training.

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------
# Basic settings
# ------------------------------------------------------------

books_dir = r"C:\Users\ghrez\gutenberg_books"
chunk_size = 300
random_seed = 42

book_author = {
    "Emma.txt": "Jane Austen",
    "Pride_and_Prejudice.txt": "Jane Austen",
    "Sense_and_Sensibility.txt": "Jane Austen",

    "Oliver_Twist.txt": "Charles Dickens",
    "A_Tale_of_Two_Cities.txt": "Charles Dickens",
    "Great_Expectations.txt": "Charles Dickens",

    "The_Mill_on_the_Floss.txt": "George Eliot",
    "Middlemarch.txt": "George Eliot",
    "Silas_Marner.txt": "George Eliot",
}

# These books are saved for the final test set.
# This is better than randomly splitting chunks because it avoids data leakage.
test_books = [
    "Sense_and_Sensibility.txt",
    "Oliver_Twist.txt",
    "Silas_Marner.txt"
]


# ------------------------------------------------------------
# Text cleaning functions
# ------------------------------------------------------------

def remove_gutenberg_text(text):
    """Remove most of the Project Gutenberg header and footer."""

    start_match = re.search(r"\*{3}\s*START OF.*?\*{3}", text, re.IGNORECASE)
    end_match = re.search(r"\*{3}\s*END OF.*?\*{3}", text, re.IGNORECASE)

    if start_match:
        text = text[start_match.end():]

    if end_match:
        text = text[:end_match.start()]

    return text.strip()


def remove_proper_nouns(text):
    """Remove likely character names and place names.

    This is a simple approach. It removes capitalized words that are not the
    first word of a sentence. The goal is to make the model learn writing style
    instead of just memorizing names from a book.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)
    new_sentences = []

    for sentence in sentences:
        words = sentence.split()
        new_words = []

        for i in range(len(words)):
            word = words[i]

            if i == 0:
                new_words.append(word)
            elif re.match(r"^[A-Z][a-z]{1,}$", word):
                new_words.append("")
            else:
                new_words.append(word)

        new_sentence = " ".join(new_words)
        new_sentences.append(new_sentence)

    return " ".join(new_sentences)


def clean_text(text):
    """Lowercase the text and keep only letters and spaces."""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def make_chunks(text, size):
    """Break a long text into chunks with a set number of words."""

    words = text.split()
    chunks = []

    for i in range(0, len(words) - size + 1, size):
        chunk = words[i:i + size]
        chunks.append(" ".join(chunk))

    return chunks


# ------------------------------------------------------------
# Extra style features
# ------------------------------------------------------------

function_words = set("""
the a an and or but if in on at to of for with by from as is was
are were be been being have has had do does did will would shall
should may might must can could that this these those it its
he she they we you i me him her us them his their our your my
not no nor so yet both either neither each every all any few more
most other such than too very just because though although while
when where who which whom what how much many
""".split())


def get_style_features(chunks):
    """Create a few simple writing-style features for each chunk."""

    all_features = []

    for chunk in chunks:
        words = chunk.split()
        total_words = len(words)

        if total_words == 0:
            total_words = 1

        function_count = 0
        total_length = 0

        for word in words:
            if word in function_words:
                function_count += 1
            total_length += len(word)

        function_density = function_count / total_words
        lexical_diversity = len(set(words)) / total_words
        average_word_length = total_length / total_words
        comma_rate = chunk.count(",") / total_words
        question_exclaim_rate = (chunk.count("?") + chunk.count("!")) / total_words

        features = [
            function_density,
            lexical_diversity,
            average_word_length,
            comma_rate,
            question_exclaim_rate
        ]

        all_features.append(features)

    return np.array(all_features)


# ------------------------------------------------------------
# Load books and build dataframe
# ------------------------------------------------------------

print("Loading books...")

records = []
missing_files = []

for file_name in book_author:
    author = book_author[file_name]
    file_path = os.path.join(books_dir, file_name)

    if not os.path.exists(file_path):
        missing_files.append(file_name)
        continue

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    text = remove_gutenberg_text(text)
    text = remove_proper_nouns(text)
    text = clean_text(text)
    chunks = make_chunks(text, chunk_size)

    if file_name in test_books:
        split_type = "test"
    else:
        split_type = "train"

    for chunk in chunks:
        record = {
            "text": chunk,
            "author": author,
            "book": file_name,
            "split": split_type
        }
        records.append(record)

    print(file_name, "--", author, "--", split_type, "--", len(chunks), "chunks")

if len(missing_files) > 0:
    print("Missing files:", missing_files)

if len(records) == 0:
    raise FileNotFoundError("No book files were found. Check the books_dir path.")

df = pd.DataFrame(records)

print("\nTotal chunks:", len(df))
print("Train/test counts:")
print(df["split"].value_counts())
print("\nAuthor and split counts:")
print(pd.crosstab(df["author"], df["split"]))


# ------------------------------------------------------------
# Build TF-IDF and style feature matrices
# ------------------------------------------------------------

train_rows = df["split"] == "train"
test_rows = df["split"] == "test"

x_train_text = df.loc[train_rows, "text"]
x_test_text = df.loc[test_rows, "text"]

y_train_names = df.loc[train_rows, "author"]
y_test_names = df.loc[test_rows, "author"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    min_df=5
)

# Fit the vectorizer on the training data only.
x_train_tfidf = vectorizer.fit_transform(x_train_text)
x_test_tfidf = vectorizer.transform(x_test_text)

train_style = get_style_features(x_train_text.tolist())
test_style = get_style_features(x_test_text.tolist())

train_style_sparse = csr_matrix(train_style)
test_style_sparse = csr_matrix(test_style)

x_train = hstack([x_train_tfidf, train_style_sparse])
x_test = hstack([x_test_tfidf, test_style_sparse])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_names)
y_test = label_encoder.transform(y_test_names)

book_encoder = LabelEncoder()
train_groups = book_encoder.fit_transform(df.loc[train_rows, "book"])

print("\nFeature matrix sizes:")
print("Training:", x_train.shape)
print("Testing:", x_test.shape)
print("Classes:", list(label_encoder.classes_))


# ------------------------------------------------------------
# Supervised machine learning models
# ------------------------------------------------------------

print("\nSupervised Classification")
print("-------------------------")

models = {}

models["Logistic Regression"] = LogisticRegression(
    max_iter=2000,
    random_state=random_seed
)

models["Complement Naive Bayes"] = ComplementNB(alpha=0.5)

models["SVM"] = LinearSVC(
    C=0.1,
    max_iter=3000,
    random_state=random_seed
)

cross_validation = GroupKFold(n_splits=6)

cv_scores = {}
reports = {}
confusion_matrices = {}

for model_name in models:
    print("\nModel:", model_name)

    model = models[model_name]

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cross_validation,
        groups=train_groups,
        scoring="accuracy"
    )

    cv_scores[model_name] = scores

    print("Cross-validation average:", round(scores.mean(), 4))
    print("Cross-validation std:", round(scores.std(), 4))
    print("Each fold:", np.round(scores, 4))

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    reports[model_name] = report
    confusion_matrices[model_name] = confusion_matrix(y_test, predictions)

    print("Test accuracy:", round(report["accuracy"], 4))
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))


summary_rows = []

for model_name in models:
    row = {
        "Model": model_name,
        "CV Accuracy": round(cv_scores[model_name].mean(), 4),
        "CV Std": round(cv_scores[model_name].std(), 4),
        "Test Accuracy": round(reports[model_name]["accuracy"], 4),
        "Macro F1": round(reports[model_name]["macro avg"]["f1-score"], 4)
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print("\nClassification Summary")
print(summary_df)


# ------------------------------------------------------------
# Unsupervised clustering
# ------------------------------------------------------------

print("\nUnsupervised Clustering")
print("-----------------------")

x_all_tfidf = vectorizer.transform(df["text"])
all_style = get_style_features(df["text"].tolist())
all_style_sparse = csr_matrix(all_style)
x_all = hstack([x_all_tfidf, all_style_sparse])
y_all = label_encoder.transform(df["author"])

svd = TruncatedSVD(n_components=100, random_state=random_seed)
x_reduced = svd.fit_transform(x_all)

print("SVD explained variance:", round(svd.explained_variance_ratio_.sum(), 4))

cluster_models = {}
cluster_models["KMeans"] = KMeans(n_clusters=3, random_state=random_seed, n_init=20)
cluster_models["Agglomerative"] = AgglomerativeClustering(n_clusters=3)
cluster_models["DBSCAN"] = DBSCAN(eps=2.0, min_samples=15)

cluster_results = {}

for cluster_name in cluster_models:
    print("\nClustering model:", cluster_name)

    cluster_model = cluster_models[cluster_name]
    cluster_labels = cluster_model.fit_predict(x_reduced)

    # DBSCAN can label some points as -1, which means noise.
    good_rows = cluster_labels != -1
    number_of_clusters = len(set(cluster_labels))

    if -1 in cluster_labels:
        number_of_clusters -= 1

    ari = adjusted_rand_score(y_all[good_rows], cluster_labels[good_rows])
    nmi = normalized_mutual_info_score(y_all[good_rows], cluster_labels[good_rows])

    unique_cluster_count = len(set(cluster_labels[good_rows]))

    if good_rows.sum() > 1 and unique_cluster_count > 1:
        sil = silhouette_score(x_reduced[good_rows], cluster_labels[good_rows])
    else:
        sil = np.nan

    cluster_results[cluster_name] = {
        "labels": cluster_labels,
        "clusters": number_of_clusters,
        "ARI": ari,
        "NMI": nmi,
        "Silhouette": sil
    }

    print("Clusters found:", number_of_clusters)
    print("ARI:", round(ari, 4))
    print("NMI:", round(nmi, 4))
    print("Silhouette:", round(sil, 4))

    cluster_table = pd.crosstab(cluster_labels, df["author"])
    print(cluster_table)


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

print("\nMaking plots...")

# Plot 1: confusion matrices
for model_name in confusion_matrices:
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        confusion_matrices[model_name],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted Author")
    plt.ylabel("Actual Author")
    plt.tight_layout()

    file_name = model_name.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(file_name + "_confusion_matrix.png", dpi=150)
    plt.close()


# Plot 2: compare model accuracy
model_names = list(models.keys())
cv_means = []
test_scores = []

for model_name in model_names:
    cv_means.append(cv_scores[model_name].mean())
    test_scores.append(reports[model_name]["accuracy"])

plt.figure(figsize=(8, 5))
x_positions = np.arange(len(model_names))
bar_width = 0.35

plt.bar(x_positions - bar_width / 2, cv_means, width=bar_width, label="CV Accuracy")
plt.bar(x_positions + bar_width / 2, test_scores, width=bar_width, label="Test Accuracy")

plt.xticks(x_positions, model_names, rotation=20)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("classifier_accuracy_comparison.png", dpi=150)
plt.close()


# Plot 3: 2-D clustering view
svd_2d = TruncatedSVD(n_components=2, random_state=random_seed)
x_2d = svd_2d.fit_transform(x_all)

for cluster_name in cluster_results:
    labels = cluster_results[cluster_name]["labels"]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, alpha=0.5, s=10)
    plt.title(cluster_name + " Clustering View")
    plt.xlabel("SVD 1")
    plt.ylabel("SVD 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    file_name = cluster_name.lower().replace(" ", "_")
    plt.savefig(file_name + "_clusters.png", dpi=150)
    plt.close()


# Plot 4: dendrogram sample
np.random.seed(random_seed)
sample_size = min(250, len(x_reduced))
sample_index = np.random.choice(len(x_reduced), sample_size, replace=False)

sample_labels = []
for i in sample_index:
    author_name = df["author"].iloc[i]
    last_name = author_name.split()[-1]
    sample_labels.append(last_name)

z = linkage(x_reduced[sample_index], method="ward")

plt.figure(figsize=(15, 5))
dendrogram(z, labels=sample_labels, leaf_rotation=90, leaf_font_size=5)
plt.title("Dendrogram Sample")
plt.xlabel("Text Chunk")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogram_sample.png", dpi=150)
plt.close()


# Plot 5: feature importance for Logistic Regression
log_reg_model = models["Logistic Regression"]
feature_names = list(vectorizer.get_feature_names_out())
feature_names = feature_names + [
    "function_density",
    "lexical_diversity",
    "average_word_length",
    "comma_rate",
    "question_exclaim_rate"
]

coefficients = log_reg_model.coef_
top_n = 15

for class_number in range(len(label_encoder.classes_)):
    author = label_encoder.classes_[class_number]
    author_coefficients = coefficients[class_number]

    top_indexes = np.argsort(author_coefficients)[-top_n:]
    top_features = []
    top_values = []

    for index in top_indexes:
        top_features.append(feature_names[index])
        top_values.append(author_coefficients[index])

    plt.figure(figsize=(8, 5))
    plt.barh(top_features, top_values)
    plt.title("Top Features for " + author)
    plt.xlabel("Logistic Regression Coefficient")
    plt.tight_layout()

    file_name = author.lower().replace(" ", "_")
    plt.savefig(file_name + "_top_features.png", dpi=150)
    plt.close()


# ------------------------------------------------------------
# Final notes printed at the end
# ------------------------------------------------------------

print("\nFinal Notes")
print("-----------")
print("This version uses a book-level split, so testing is done on full books that were not used for training.")
print("That makes the results more realistic than a random chunk split.")
print("Supervised models should usually perform better than clustering because they are given the author labels during training.")
print("Clustering is harder because it tries to find author groups without being told the answers.")
print("\nDone.")
