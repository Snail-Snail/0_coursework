#NLP assessment template 2026

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
import pandas as pd
from pathlib import Path
import re   
import string ## to do fancy things with punctuation
import nltk
# nltk.download('punkt_tab')
# nltk.download("cmudict")
from nltk.corpus import cmudict
import pickle
from collections import Counter
import math

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



# count the number of syllables in one word
cmu = cmudict.dict()
def count_syllables(word):
    """Counts the number of syllables in a word using the cmudict."""
    phones = cmu.get(word.lower())
    if phones:
        return sum(1 for ph in phones[0] if ph[-1].isdigit())
    return None

# count syllables in a text
def count_syllables_in_text(text):
    """Counts the total number of syllables in a text."""
    tokens = clean_text(text)
    sum_syllables = 0
    for t in tokens:
        syllables = count_syllables(t)
        if syllables is not None:
            sum_syllables += syllables
    return sum_syllables

# count sentences in a text 
def count_sentences(text):
    """Counts the number of sentences in a text using nltk.sent_tokenize."""
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

# calculate fk level for each novel and return a dictionary mapping title to fk level
def fk_level():
    """returns a diction mapping title to Flesch–Kincaid Grade Level."""
    path = Path.cwd()/ "cw-pack-2026" / "texts" / "novels"
    d_title2fk = dict()
    for f in path.glob("*.txt"):
        title, author, year = f.stem.split("-")
        text = f.read_text()
        tokens = clean_text(text)
        n_sentences = count_sentences(text)
        n_syllables = count_syllables_in_text(text)
        fk_grade = 0.39 * (len(tokens) / n_sentences) + 11.8 * (n_syllables / len(tokens)) - 15.59  
        d_title2fk[title] = fk_grade
    return d_title2fk


def read_novels(path=Path.cwd()/ "cw-pack-2026" / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    rows = []
    for f in path.glob("*.txt"):
        title, author, year = f.stem.split("-")
        text = f.read_text()
        rows.append({ "text": text, "title": title, "author": author, "year": year})
    novels_df = pd.DataFrame(rows)
    # sort the dataframe by year and reset the index
    novels_df = novels_df.sort_values("year")
    novels_df = novels_df.reset_index(drop=True)
    return novels_df


def parse(df, store_path=Path.cwd() /"cw-pack-2026" / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path.mkdir(parents=True, exist_ok=True)  # creates the folder if it doesn't exist
    df["Doc"] = df["text"].apply(nlp)
    with open (store_path / out_name, "wb") as f:
        pickle.dump(df, f)
    return df

def clean_text(text):
    """Cleans a text by removing punctuation and converting to lowercase."""
    tokens = nltk.word_tokenize(text) # use nltk to tokenize the text into words
    tokens = [t.lower() for t in tokens]
    re_punc = re.compile("[%s]" % re.escape(string.punctuation))
    tokens = [re_punc.sub("", t) for t in tokens]
    tokens = [t for t in tokens if t.strip() != ""]
    return tokens

def nltk_ttr():
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # df = read_novels()
    path = Path.cwd()/ "cw-pack-2026" / "texts" / "novels"
    d_title2ttr = dict()
    for f in path.glob("*.txt"):
        title, author, year = f.stem.split("-")
        text = f.read_text()
        tokens = clean_text(text)
        ttr = len(set(tokens)) / len(tokens)
        d_title2ttr[title] = ttr

    return d_title2ttr  


#.. add functions for part (e) here
def get_subjects(doc):
    """Returns a list of the 10 most common syntactic subjects in a spaCy Doc."""
    subjects = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subjects.append(token.text.lower())
    
    counter = Counter(subjects)
    return counter.most_common(10)


# function for question e (ii)
def get_pmi_verbs(doc, subject="he"):
    """return verbs most associated with the given subject, ordered by PMI"""
    he_verb_counts = Counter()
    all_subject_verb_counts = Counter()
    he_as_subject = 0 
    total_subject_tokens = 0

    for token in doc:
        if token.dep_ == "nsubj":
            verb = token.head.lemma_.lower()
            subj = token.text.lower()
            
            total_subject_tokens += 1
            all_subject_verb_counts[verb] += 1  # count for ALL subjects
            
            if subj == subject:
                he_as_subject += 1
                he_verb_counts[verb] += 1  # count only for "he"

    # calculate PMI for each verb that occurs with "he"
    pmi_scores = {}
    for verb, count in he_verb_counts.items():
        p_he_verb = count / total_subject_tokens
        p_he = he_as_subject / total_subject_tokens
        p_verb = all_subject_verb_counts[verb] / total_subject_tokens
        
        pmi = math.log(p_he_verb / (p_he * p_verb))
        pmi_scores[verb] = round(pmi, 4)

    sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pmi


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() /"cw-pack-2026" / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    # nltk.download("cmudict")
    
    """parse(df)"""  # uncomment this later
    print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    
    # load Load the dataframe from the pickle file and use it for the remainder of this coursework part. 
    with open (Path.cwd() / "cw-pack-2026" / "pickles" / "parsed.pickle", "rb") as f:
        df = pickle.load(f)
    # call functions for part (e) here.
    # for question e, i: title of each novel and the 10 most common syntactic subjects.
    for idx, row in df.iterrows():
        title = row["title"]
        doc = row["Doc"]
        subjects = get_subjects(doc)
        print(f"Title: {title}, 10 most common syntactic subjects: {subjects}")
    
    # for quesiton e, ii: title of each novel and a list of verbs most likely to occur with the subject he
    # write the loop
    for idx, row in df.iterrows():
        title = row["title"]
        doc = row["Doc"]
        pmi_verbs = get_pmi_verbs(doc, subject="he")
        print(f"Title: {title}, Verbs most associated with 'he' by PMI: {pmi_verbs[:10]}")
