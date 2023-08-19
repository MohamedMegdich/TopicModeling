import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

quora = pd.read_csv('quora_questions.csv')

print(quora.head())

# Preprocessing
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(quora['Question'])

# Non-negative Matrix Factorization
nmf_model = NMF(n_components=20, random_state=42)
nmf_model.fit(dtm)

# Print our the top 15 most common words for each of the 20 topics.
for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# Attaching Discovered Topic Labels to Original Articles
topic_results = nmf_model.transform(dtm)

quora['topic'] = topic_results.argmax(axis=1)

print(quora.head(15))



