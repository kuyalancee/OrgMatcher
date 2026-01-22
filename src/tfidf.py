import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3
import string

def access_and_load_db():
    conn = sqlite3.connect("db_and_csvs/unt_clubs.db")
    data = pd.read_sql_query("SELECT * FROM organizations", conn)
    try:
        conn.close()
        print('db closed')
    except Exception as e:
        print(f'db err: {e}')
    # define 'document' as described on the paper
    # 2* weight on short_name
    data = data.fillna('')
    data['document'] = (
        data['name'] + " " + 
        data['short_name'].astype(str) + " " + data['short_name'].astype(str) + " " +
        data['summary'] + " " + 
        data['description']
    )
    return data


# define lemmatizer
lemmatize = WordNetLemmatizer()
def lemmatize_text(text):
    text = text.lower().translate(str.maketrans('','', string.punctuation))
    terms = text.split()
    return " ".join([lemmatize.lemmatize(word) for word in terms])


def get_recommendations(query, data, tfidf_vectorizer, tfidf_matrix):
    query_vectorized = tfidf_vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vectorized, tfidf_matrix)
    top_five = np.argsort(cosine_sim[0])[-5:][::-1]
    return data.iloc[top_five]

if __name__ == "__main__":
    data = access_and_load_db()

    stop_words = list(text.ENGLISH_STOP_WORDS)
    lemmatized_stop_words = [lemmatize.lemmatize(word) for word in stop_words]

    tfidf = TfidfVectorizer(
        preprocessor=lemmatize_text,
        stop_words=lemmatized_stop_words,
        token_pattern=r"\b\w\w+\b"
    )
    tfidf_matrix = tfidf.fit_transform(data['document'])

    usr_input = input("Enter what you're looking for in an organization: ")
    res = get_recommendations(usr_input, data, tfidf, tfidf_matrix)

    print(f'top 5 matches for {usr_input}:')
    res_display = res[['name', 'short_name']].reset_index(drop=True)
    res_display.index = res_display.index + 1
    print(res_display)