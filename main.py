import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from belu import compute_belief

nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load the Dataset
df = pd.read_csv('your_dataset.csv')

# Step 2: Preprocess the Data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.lower()  # Convert text to lowercase
    return text

df['preprocessed_abstract'] = df['abstract'].apply(preprocess_text)

# Step 3: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['preprocessed_abstract'])

# Step 4: Sentence Scoring
sentence_scores = []
for i in range(len(df)):
    sentence_scores.append({})
    sentence_tokens = nltk.sent_tokenize(df['abstract'][i])
    for sentence in sentence_tokens:
        sentence = preprocess_text(sentence)
        sentence_words = nltk.word_tokenize(sentence)
        for word in sentence_words:
            if word in vectorizer.vocabulary_:
                if word in sentence_scores[i]:
                    sentence_scores[i][word] += tfidf_matrix[i, vectorizer.vocabulary_[word]]
                else:
                    sentence_scores[i][word] = tfidf_matrix[i, vectorizer.vocabulary_[word]]

# Step 5: Select Top Sentences
summary_sentences = []
for i in range(len(df)):
    top_sentences = sorted(sentence_scores[i], key=sentence_scores[i].get, reverse=True)[:3]  # Select top 3 sentences
    summary = ' '.join(top_sentences)
    summary_sentences.append(summary)

df['summary'] = summary_sentences

# Step 6: Evaluate Performance
rouge = Rouge()
scores = rouge.get_scores(df['summary'], df['title'])
belu_score = compute_belief(df['summary'], df['title'])

print("ROUGE Score:")
print(scores)
print("BELU Score:", belu_score)
