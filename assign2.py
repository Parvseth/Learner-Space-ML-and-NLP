# ================================
#  Assignment 2 – FULL SOLUTION
#  Author: <your-name-here>
#  Tested with: Python 3.11, scikit-learn 1.5, gensim 4.3, NLTK 3.8
#  =================================

# ---------- Common Imports ----------
import re
import string
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, download

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# one-time NLTK downloads (no-ops if already present)
download('punkt')
download('stopwords')
download('wordnet')
download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------- Utility: load Word2Vec (shared) ----------
def load_w2v(path: str) -> KeyedVectors:
    """
    Load GoogleNews 300-dimensional Word2Vec from the given .bin or .bin.gz path.
    """
    print("Loading Word2Vec … this can take a few minutes ⏳")
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
    print("✓ Word2Vec loaded")
    return w2v_model

# ---------- Utility: sentence → average vector ----------
def sentence_vector(sentence: list[str], w2v: KeyedVectors) -> np.ndarray:
    """
    Average the Word2Vec vectors of the given list of tokens.
    Returns a 300-D numpy array; if no token is in the vocab, returns zeros.
    """
    vectors = [w2v[w] for w in sentence if w in w2v]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

# ======================================================
#  PROBLEM 1 – SMS Spam Collection
# ======================================================
def preprocess_sms(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and t.isalpha()]
    return tokens

def solve_problem1(sms_csv_path: str, w2v_path: str):
    # 1. Load data
    df = pd.read_csv(sms_csv_path, sep='\t', names=['Label', 'Message'])
    print(df.head())

    # 2. Pre-processing & vectorisation
    w2v = load_w2v(w2v_path)
    df['tokens'] = df['Message'].apply(preprocess_sms)
    X = np.vstack(df['tokens'].apply(lambda toks: sentence_vector(toks, w2v)))
    y = df['Label'].map({'ham': 0, 'spam': 1}).values

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = clf.predict(X_test)
    print("Problem 1 – Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    return clf, w2v

def predict_message_class(model: LogisticRegression, w2v: KeyedVectors, message: str) -> str:
    tokens = preprocess_sms(message)
    vec = sentence_vector(tokens, w2v).reshape(1, -1)
    pred = model.predict(vec)[0]
    return 'spam' if pred == 1 else 'ham'

# ======================================================
#  PROBLEM 2 – Twitter US Airline Sentiment
# ======================================================
CONTRACTIONS = {
    "don't": "do not", "can't": "can not", "i'm": "i am",
    "it's": "it is", "won't": "will not", "didn't": "did not",
    # … extend list as needed
}

def expand_contractions(text: str) -> str:
    pattern = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))
    return pattern.sub(lambda x: CONTRACTIONS[x.group(0)], text)

URL_RE   = re.compile(r'https?://\S+|www\.\S+')
MENTION  = re.compile(r'@\w+')
HASHTAG  = re.compile(r'#\w+')

def clean_tweet(text: str) -> list[str]:
    text = expand_contractions(text.lower())
    text = URL_RE.sub('', text)
    text = MENTION.sub('', text)
    text = HASHTAG.sub('', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in STOPWORDS]
    return tokens

def solve_problem2(twitter_csv_path: str, w2v_path: str):
    # 1. Load data
    df = pd.read_csv(twitter_csv_path)
    df = df[['airline_sentiment', 'text']].dropna()
    print(df.head())

    # 2. Pre-processing & vectorisation
    w2v = load_w2v(w2v_path)
    df['tokens'] = df['text'].apply(clean_tweet)
    X = np.vstack(df['tokens'].apply(lambda toks: sentence_vector(toks, w2v)))
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['airline_sentiment'].map(sentiment_map).values

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Multiclass Logistic Regression (one-vs-rest)
    clf = LogisticRegression(max_iter=1000, multi_class='ovr')
    clf.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = clf.predict(X_test)
    print("Problem 2 – Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(
        y_test, y_pred,
        target_names=['negative', 'neutral', 'positive']
    ))

    return clf, w2v

def predict_tweet_sentiment(model: LogisticRegression, w2v: KeyedVectors, tweet: str) -> str:
    tokens = clean_tweet(tweet)
    vec = sentence_vector(tokens, w2v).reshape(1, -1)
    idx = model.predict(vec)[0]
    return {0: 'negative', 1: 'neutral', 2: 'positive'}[idx]

# ======================================================
#  EXAMPLE USAGE (uncomment to run)
# ======================================================
# if __name__ == "__main__":
#     SMS_CSV   = "spam.csv"          # put your SMS Spam TSV here
#     TW_CSV    = "Tweets.csv"        # put your Twitter dataset here
#     W2V_PATH  = "GoogleNews-vectors-negative300.bin.gz"
#
#     # ---------- Problem 1 ----------
#     sms_model, sms_w2v = solve_problem1(SMS_CSV, W2V_PATH)
#     print(predict_message_class(sms_model, sms_w2v,
#           "Congratulations! You've won a free cruise to the Bahamas. Call now."))
#
#     # ---------- Problem 2 ----------
#     tw_model, tw_w2v = solve_problem2(TW_CSV, W2V_PATH)
#     print(predict_tweet_sentiment(tw_model, tw_w2v,
#           "I love how friendly the flight attendants were on my @SouthwestAir flight!"))
