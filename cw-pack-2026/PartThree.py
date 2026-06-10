# question 3a
# model was suggested by ChatGPT
"""I used google/flan-t5-base accessed locally via the HuggingFace transformers library. 
This model was chosen because it is lightweight (~300MB), runs on CPU without requiring a GPU, 
and was designed to follow natural language instructions, 
making it suitable for prompt-based classification. 
max_new_tokens was set to 10 since the expected output is a single party name requiring no more than a few tokens. 
Input was truncated to 512 tokens to fit the model's context window. 
The model was run locally rather than through Ollama or OpenRouter due to network restrictions preventing access to external inference APIs."""

##############################################################################################
# question 3b zero shot classifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report, f1_score

df = pd.read_csv("/Users/sisigao/Desktop/Birkbeck_master/Natural_language_processing/0_coursework/cw-pack-2026/texts/hansard500.csv")
# display(df.head(2))

# to use the same label set as part Two

# unify the party name
df["party"] = df["party"].replace("Labour (Co-op)", "Labour")

# filter for the four most popular parties
# The error is because hansard500.The error is because hansard500.csv has so few Liberal Democrat speeches
# after filtering, only 1 remains — not enough for stratified splitting.
# therefore, Liberal Democrat is removed from the dataset
df_cleaned = df[df["party"].isin(["Labour", "Conservative", "Scottish National Party"])]

# remove any rows where the value in the ‘speech class’ column is not ‘Speech’.
df_cleaned = df_cleaned[df_cleaned["speech_class"] == "Speech"]

# remove any rows where the text in the ‘speech’ column is less than 1000 characters long.
df_cleaned = df_cleaned[df_cleaned["speech"].str.len() >= 1000]

# print(df_cleaned.shape)
# print(df["party"].value_counts())

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# function format was learned from Claude
def zero_shot_classify(speech_text):
    prompt = f"""this is a political speech classifier. 
    Classify the following UK parliamentary speech into exactly one of these parties:
    Conservative, Labour, Scottish National Party.
    
    Output only the party name, nothing else. 
    Speech: {speech_text[:500]}

    Party: """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# split the train test dataset without data leakage
speeches = df_cleaned["speech"].values
y = df_cleaned["party"].values

# train, test split on raw text
speech_train, speech_test, y_train, y_test = train_test_split(
    speeches, y, test_size=0.2, random_state=26, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train = vectorizer.fit_transform(speech_train)
X_test = vectorizer.transform(speech_test)

# run zero-shot on test set
y_pred = [zero_shot_classify(speech) for speech in speech_test]  # raw text, not vectorised
print("macro F1-score:", f1_score(y_test, y_pred, average="macro"))
print("Zero shot Classification Report:")
print(classification_report(y_test, y_pred))


##############################################################################################
# question 3c few shot classifier
import numpy as np

# function to pick examples from df
def get_few_shot_example(speech_train, y_train, n_per_class = 1):
    """select 1 example per class from training data"""
    examples = []
    for label in ["Conservative", "Labour", "Scottish National Party"]:
        # get indices for this class. suggested by Gemini.
        indices = np.where(y_train == label)[0]
        # pick the first one
        idx = indices[0]
        examples.append((speech_train[idx][:300], label))
    return examples

few_shot_examples = get_few_shot_example(speech_train, y_train, n_per_class=1)

# function learned from Claude
def build_few_shot_prompt(speech_text, examples):
    # build the example section
    examples_text = ""
    for speech, label in examples:
        examples_text += f"Speech: {speech}\nParty: {label}\n\n"
    
    prompt = f"""Classify the following UK parliamentary speech into exactly one of these parties: 
    Conservative, Labour, Scottish National Party.
    Output only the party name, nothing else.

    Here are some examples:
    {examples_text}
    
    Now classify this speech:
    Speech: {speech_text[:500]}
    Party:"""

    return prompt

# function suggested by Claude
def few_shot_classify(speech_text):
    prompt = build_few_shot_prompt(speech_text, few_shot_examples)
    inputs = tokenizer(prompt, return_tensors="pt", truncation= True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens = 10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# run on test set
y_pred_few = [few_shot_classify(speech) for speech in speech_test]

print("Few-shot macro F1-score:", f1_score(y_test, y_pred_few, average="macro"))
print("Few-shot Classification Report:")
print(classification_report(y_test, y_pred_few))

##############################################################################################
# question 3d
# the reason why that small models perform worse on few-shot was suggested by Claude.
"""Zero-shot vs Few-shot Comparison

The zero-shot approach produced a F1 score of 0.39, slightly higher than the few-shot score of 0.34. 
Although few-shot prompting is often expected to improve performance by providing labelled examples, this was not the case in this experiment.

One likely reason is the limitation of the model used, flan-t5-base, which has around 250 million parameters and a maximum input length of 512 tokens. 

This highlights a common issue with smaller language models: 
adding examples can reduce the amount of context available for the actual task. 
Larger models with longer context windows are generally better able to benefit from few-shot prompting without losing important input information.

Despite the difference between the two approaches, both achieved relatively low scores compared with the traditional machine learning models evaluated in Part Two. 
This suggests that the model struggled to identify subtle linguistic differences between political parties.
"""
