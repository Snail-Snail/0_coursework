import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, word_tokenize

#  question a
df = pd.read_csv("/Users/sisigao/Desktop/Birkbeck_master/Natural_language_processing/0_coursework/cw-pack-2026/texts/hansard10000.csv")
# display(df.head(2))

# question a(i)
# rename the labours(Co-op) in party column to labour
df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
# display(df.head(2))

# question a(ii)
# remove any rows where the value of the ‘party’ column is not one of the four
# most common party names, and remove the ‘Speaker’ value.
parties_in_df = df["party"].unique()
df_cleaned = df[df["party"].isin(["Labour", "Conservative", "Scottish National Party", "Liberal Democrat"])]

# remove the speakername column
df_cleaned = df_cleaned.drop(columns=["speakername"])
# display(df_cleaned.head(2))
# print(len(df_cleaned))

# question a(iii)
# remove any rows where the value in the ‘speech class’ column is not ‘Speech’.
df_cleaned = df_cleaned[df_cleaned["speech_class"] == "Speech"]
# print(len(df_cleaned))

# question a(iv)
# remove any rows where the text in the ‘speech’ column is less than 1000 characters long.
df_cleaned = df_cleaned[df_cleaned["speech"].str.len() >= 1000]
# print(len(df_cleaned))
print(f"Shape of cleaned DataFrame: {df_cleaned.shape}")

##############################################################################################
# question 2b
# vectorise using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(df_cleaned["speech"])
y = df_cleaned["party"]

# train test split with stratifed sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=26, stratify=y
)

# train a Random Forest classifier
rf = RandomForestClassifier(random_state=26, n_estimators=300)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("macro F1-score:", f1_score(y_test, rf_pred, average="macro"))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# train a SVM with linear kernel
svm = LinearSVC(random_state=26)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("macro F1-score:", f1_score(y_test, svm_pred, average="macro"))
print("SVM (Linear) Classification Report:")
print(classification_report(y_test, svm_pred))

##############################################################################################
# question 2c with n-grams
vectorizer_ngram = TfidfVectorizer(
    stop_words="english", max_features=3000, ngram_range= (1, 3)
)
X_ngram = vectorizer_ngram.fit_transform(df_cleaned["speech"])

X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(
    X_ngram, df_cleaned["party"], test_size=0.2, random_state=26, stratify=df_cleaned["party"]
)

# train a Random Forest classifier
rf_ng = RandomForestClassifier(random_state=26, n_estimators=300)
rf_ng.fit(X_train_ng, y_train_ng)
rf_ng_pred = rf_ng.predict(X_test_ng)
print("macro F1-score:", f1_score(y_test_ng, rf_ng_pred, average="macro"))
print("Random Forest Classification Report:")
print(classification_report(y_test_ng, rf_ng_pred))

# train a SVM with linear kernel
svm_ng = LinearSVC(random_state=26)
svm_ng.fit(X_train_ng, y_train_ng)
svm_ng_pred = svm_ng.predict(X_test_ng)
print("macro F1-score:", f1_score(y_test_ng, svm_ng_pred, average="macro"))
print("SVM (Linear) Classification Report:")
print(classification_report(y_test_ng, svm_ng_pred))

##############################################################################################
# question 2d
# lemmatise to reduce vocabulary noise
# POS filtering to keep only nouns, verbs, adj. most likely 
# keep bigram for party-specific phrases (normally two words)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def get_wordnet_pos(treebank_tag):
    """Convert Penn Treebank POS tags to WordNet POS tags for lemmatizer."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

def custom_tokenizer(text):
    # 1. lowercase and remove non-alpha characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # 2. tokenize
    tokens = word_tokenize(text)

    # 3. POS tag
    tagged = pos_tag(tokens)

    # 4. keep only nouns, verbs, adjectives, adverbs; lemmatize; remove stopwords
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in stop_words
        and len(word) > 2
        and tag.startswith(("N", "V", "J", "R"))  # nouns, verbs, adj, adv
    ]

    return lemmatized

# vectorise with custom tokenizer + bigrams, max 3000 features
vectorizer_custom = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    max_features=3000,
    # unigrams + bigrams (trigrams add noise with lemmatized tokens)
    ngram_range=(1, 2),   
    # ignore very rare terms (reduces noise)
    min_df=2,             
    # log-scale TF dampens very frequent terms
    sublinear_tf=True,    
)

X_custom = vectorizer_custom.fit_transform(df_cleaned["speech"])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_custom, df_cleaned["party"], test_size=0.2, random_state=26, stratify=df_cleaned["party"]
)

# evaluate classifiers, report the best
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=26),
    "SVM (Linear)":  LinearSVC(random_state=26),
}

best_name, best_score, best_pred = None, 0, None

for name, clf in classifiers.items():
    clf.fit(X_train_c, y_train_c)
    pred = clf.predict(X_test_c)
    score = f1_score(y_test_c, pred, average="macro")
    print(f"{name} — macro F1: {score:.4f}")
    if score > best_score:
        best_score = score
        best_name  = name
        best_pred  = pred

print(f"\nBest classifier: {best_name}")
print(f"macro F1-score: {best_score:.4f}")
print("Classification Report:")
print(classification_report(y_test_c, best_pred))

