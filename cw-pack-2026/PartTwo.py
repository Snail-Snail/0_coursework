import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

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