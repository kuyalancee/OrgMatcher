import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3
import string
import os
from synonyms import SYNONYM_MAP

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class OrgMatcher:
    def __init__(self, db_path="db_and_csvs/unt_clubs.db"):
        self.db_path = db_path
        self.lemmatizer = WordNetLemmatizer()
        self.data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._initialize()

    def _initialize(self):
        print("Initializing Recommendation Engine...")
        self.data = self._access_and_load_db()
        
        stop_words = list(text.ENGLISH_STOP_WORDS)
        lemmatized_stop_words = [self._lemmatize_text(word, expand_synonyms=False) for word in stop_words]

        self.tfidf_vectorizer = TfidfVectorizer(
            preprocessor=lambda x: self._lemmatize_text(x, expand_synonyms=False),
            stop_words=lemmatized_stop_words,
            token_pattern=r"\b\w\w+\b"
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['document'])
        print("Initialization complete.")

    def _access_and_load_db(self):
        if not os.path.exists(self.db_path):
            if os.path.exists(os.path.join("..", self.db_path)):
                self.db_path = os.path.join("..", self.db_path)
            elif os.path.exists(os.path.basename(self.db_path)):
                 self.db_path = os.path.basename(self.db_path)
            
        conn = sqlite3.connect(self.db_path)
        try:
            data = pd.read_sql_query("SELECT * FROM organizations", conn)
        finally:
            conn.close()
        
        # define 'document' as described on the paper
        # increase weight on short_name
        data = data.fillna('')
        data['document'] = (
            data['name'] + " " + 
            data['short_name'].astype(str) + " " + data['short_name'].astype(str) + " " +
            data['summary'] + " " + 
            data['description']
        )
        #print(type(data))
        return data

    # define lemmatizer
    def _lemmatize_text(self, text, expand_synonyms=True):
        text = text.lower().translate(str.maketrans('','', string.punctuation))
        words = text.split()
        # dont expand synonym_map for stopwords
        if expand_synonyms:
            new_words = [SYNONYM_MAP.get(word, word) for word in words]
        else:
            new_words = words
        return " ".join([self.lemmatizer.lemmatize(word) for word in new_words])

    def search(self, query, top_n=5):
        if not query:
            return []
            
        expanded_query = self._lemmatize_text(query, expand_synonyms=True)
        query_vectorized = self.tfidf_vectorizer.transform([expanded_query])
        cosine_sim = cosine_similarity(query_vectorized, self.tfidf_matrix)
        
        # get top 5 most relevant
        top_indices = np.argsort(cosine_sim[0])[-top_n:][::-1]
        
        # return as list of dicts for display
        results = self.data.iloc[top_indices]
        return results[['name', 'short_name', 'summary', 'description']].to_dict(orient='records')

if __name__ == "__main__":
    try:
        engine = OrgMatcher()
        while True:
            usr_input = input("\nEnter what you're looking for in an organization(or 'q' to quit): ")
            if usr_input.lower() == 'q':
                break
            
            recommendations = engine.search(usr_input)
            print(f'\nTop matches for "{usr_input}":')
            for i, res in enumerate(recommendations, 1):
                print(f"{i}. {res['name']} ({res['short_name']})")
    except Exception as e:
        print(f"Error: {e}")
