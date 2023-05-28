import pandas as pd
import spacy

# Read the dataset from CSV
df1 = pd.read_csv('/content/drive/MyDrive/NLP/data/data3.csv')

# Read the second CSV file
df2 = pd.read_csv('/content/drive/MyDrive/NLP/data/data7.csv')

# Concatenate the dataframes vertically
df = pd.concat([df1, df2], axis=0)

# Reset the index of the combined dataframe
df.reset_index(drop=True, inplace=True)

# Create a spacy model
nlp = spacy.load("en_core_web_sm")

# Extract the summaries
summaries = []
for i in range(len(df)):
  # Get the abstract
  abstract = df["abstract"][i]

  # Create a document object
  doc = nlp(abstract)

  # Extract the discourse topics
  topics = []
  for sent in doc.sents:
    for rel in sent._.discourse_relations:
      if rel.label == "Topic-Comment":
        topics.append(rel.from_)

  # Extract the most important sentences
  sentences = []
  for topic in topics:
    for sent in doc.sents:
      if sent.text == topic:
        sentences.append(sent)

  # Create a summary
  summary = " ".join([sent.text for sent in sentences])

  # Calculate the ROUGE score
  scores = rouge_score.rouge_n(summary, df["abstract"][i])

  # Add the summary and the ROUGE score to the list
  summaries.append((summary, scores))

# Save the summaries to a file
df["summary"] = summaries
df.to_csv("output.csv", index=False)