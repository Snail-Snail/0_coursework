import pandas as pd

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