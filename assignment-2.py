# ML Assignment 2 - SMS Spam Detection & Twitter Sentiment Analysis
# Required libraries installation:
# pip install pandas numpy scikit-learn nltk gensim matplotlib seaborn

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =====================================================================
# PROBLEM 1: SMS SPAM DETECTION
# =====================================================================

class SMSSpamDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.w2v_model = None
        self.classifier = None
        
    def load_word2vec_model(self, model_path):
        """Load pre-trained Word2Vec model"""
        print("Loading Word2Vec model...")
        # For Google News Word2Vec model (binary format)
        self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("Word2Vec model loaded successfully!")
        
    def preprocess_message(self, message):
        """Preprocess SMS message: tokenize, remove stop words, lowercase"""
        # Convert to lowercase
        message = message.lower()
        
        # Remove punctuation and special characters
        message = re.sub(r'[^a-zA-Z\s]', '', message)
        
        # Tokenize
        tokens = word_tokenize(message)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        return tokens
    
    def message_to_vector(self, message):
        """Convert message to vector by averaging Word2Vec vectors"""
        tokens = self.preprocess_message(message)
        
        # Get vectors for tokens that exist in vocabulary
        vectors = []
        for token in tokens:
            if token in self.w2v_model.key_to_index:
                vectors.append(self.w2v_model[token])
        
        if vectors:
            # Average the vectors
            return np.mean(vectors, axis=0)
        else:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.w2v_model.vector_size)
    
    def vectorize_messages(self, messages):
        """Convert all messages to vectors"""
        print("Vectorizing messages...")
        vectors = []
        for message in messages:
            vector = self.message_to_vector(message)
            vectors.append(vector)
        return np.array(vectors)
    
    def train(self, df):
        """Train the SMS spam detector"""
        print("Training SMS Spam Detector...")
        
        # Vectorize messages
        X = self.vectorize_messages(df['Message'])
        y = df['Label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"SMS Spam Detection Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_message_class(self, message):
        """Predict class for a single message"""
        if self.classifier is None or self.w2v_model is None:
            raise ValueError("Model not trained or Word2Vec model not loaded")
        
        vector = self.message_to_vector(message)
        prediction = self.classifier.predict([vector])[0]
        probability = self.classifier.predict_proba([vector])[0].max()
        
        return prediction, probability

# =====================================================================
# PROBLEM 2: TWITTER SENTIMENT ANALYSIS
# =====================================================================

class TwitterSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.w2v_model = None
        self.classifier = None
        
        # Common contractions dictionary
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is", "that's": "that is",
            "what's": "what is", "where's": "where is", "how's": "how is",
            "when's": "when is", "why's": "why is", "who's": "who is"
        }
    
    def load_word2vec_model(self, model_path):
        """Load pre-trained Word2Vec model"""
        print("Loading Word2Vec model...")
        self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("Word2Vec model loaded successfully!")
    
    def expand_contractions(self, text):
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def preprocess_tweet(self, tweet):
        """Preprocess tweet with comprehensive cleaning"""
        # Convert to lowercase
        tweet = tweet.lower()
        
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        tweet = re.sub(r'@\w+', '', tweet)
        
        # Remove hashtags (keep the text after #)
        tweet = re.sub(r'#', '', tweet)
        
        # Expand contractions
        tweet = self.expand_contractions(tweet)
        
        # Remove punctuation and special characters
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        
        # Tokenize
        tokens = word_tokenize(tweet)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 1
        ]
        
        return tokens
    
    def tweet_to_vector(self, tweet):
        """Convert tweet to vector by averaging Word2Vec vectors"""
        tokens = self.preprocess_tweet(tweet)
        
        # Get vectors for tokens that exist in vocabulary
        vectors = []
        for token in tokens:
            if token in self.w2v_model.key_to_index:
                vectors.append(self.w2v_model[token])
        
        if vectors:
            # Average the vectors
            return np.mean(vectors, axis=0)
        else:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.w2v_model.vector_size)
    
    def vectorize_tweets(self, tweets):
        """Convert all tweets to vectors"""
        print("Vectorizing tweets...")
        vectors = []
        for tweet in tweets:
            vector = self.tweet_to_vector(tweet)
            vectors.append(vector)
        return np.array(vectors)
    
    def train(self, df):
        """Train the Twitter sentiment analyzer"""
        print("Training Twitter Sentiment Analyzer...")
        
        # Vectorize tweets
        X = self.vectorize_tweets(df['text'])
        y = df['airline_sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiclass classifier
        self.classifier = LogisticRegression(random_state=42, multi_class='ovr', max_iter=1000)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Twitter Sentiment Analysis Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_tweet_sentiment(self, tweet):
        """Predict sentiment for a single tweet"""
        if self.classifier is None or self.w2v_model is None:
            raise ValueError("Model not trained or Word2Vec model not loaded")
        
        vector = self.tweet_to_vector(tweet)
        prediction = self.classifier.predict([vector])[0]
        probability = self.classifier.predict_proba([vector])[0].max()
        
        return prediction, probability

# =====================================================================
# EXAMPLE USAGE AND TESTING
# =====================================================================

def main():
    # Note: You need to download the Google News Word2Vec model
    # Download from: https://code.google.com/archive/p/word2vec/
    # File: GoogleNews-vectors-negative300.bin
    
    # Example paths - update these with your actual file paths
    W2V_MODEL_PATH = r"D:\Learner Space ML and NLP\GoogleNews-vectors-negative300.bin"
    SMS_DATA_PATH = r"D:\Learner Space ML and NLP\spam.csv" # SMS Spam Collection dataset
    TWITTER_DATA_PATH = r"D:\Learner Space ML and NLP\Tweets.csv"  # Twitter US Airline Sentiment dataset
    
    print("=" * 60)
    print("ML ASSIGNMENT 2 - SOLUTION")
    print("=" * 60)
    
    try:
        # Problem 1: SMS Spam Detection
        print("\n" + "="*30)
        print("PROBLEM 1: SMS SPAM DETECTION")
        print("="*30)
        
        # Load SMS data
        # Note: The SMS dataset typically has different column names
        # You may need to adjust based on your actual dataset structure
        try:
            sms_df = pd.read_csv(SMS_DATA_PATH, encoding='latin-1')
            # Assuming the dataset has columns like 'v1' and 'v2'
            if 'v1' in sms_df.columns and 'v2' in sms_df.columns:
                sms_df = sms_df.rename(columns={'v1': 'Label', 'v2': 'Message'})
            print(f"SMS Dataset loaded: {len(sms_df)} messages")
            print(f"Label distribution:\n{sms_df['Label'].value_counts()}")
        except FileNotFoundError:
            print("SMS dataset not found. Please download the SMS Spam Collection dataset.")
            print("Creating sample data for demonstration...")
            sms_df = pd.DataFrame({
                'Label': ['ham', 'spam', 'ham', 'spam', 'ham'] * 100,
                'Message': [
                    'Hello how are you doing today?',
                    'URGENT! You have won $1000! Click here now!',
                    'Can you pick up milk on your way home?',
                    'FREE iPhone! Limited time offer! Call now!',
                    'Thanks for the dinner invitation'
                ] * 100
            })
        
        # Initialize and train SMS detector
        sms_detector = SMSSpamDetector()
        
        # For demonstration, we'll use a smaller subset if Word2Vec model is not available
        try:
            sms_detector.load_word2vec_model(W2V_MODEL_PATH)
            sms_accuracy = sms_detector.train(sms_df)
            
            # Test prediction function
            test_messages = [
                "Hello, how are you doing today?",
                "URGENT! You have won $1000! Click here now!",
                "Can you call me when you get this message?"
            ]
            
            print("\nTesting SMS Prediction Function:")
            for msg in test_messages:
                pred, prob = sms_detector.predict_message_class(msg)
                print(f"Message: '{msg}'")
                print(f"Prediction: {pred} (Confidence: {prob:.4f})\n")
                
        except FileNotFoundError:
            print("Word2Vec model not found. Please download GoogleNews-vectors-negative300.bin")
        
        # Problem 2: Twitter Sentiment Analysis
        print("\n" + "="*35)
        print("PROBLEM 2: TWITTER SENTIMENT ANALYSIS")
        print("="*35)
        
        try:
            twitter_df = pd.read_csv(TWITTER_DATA_PATH)
            print(f"Twitter Dataset loaded: {len(twitter_df)} tweets")
            print(f"Sentiment distribution:\n{twitter_df['airline_sentiment'].value_counts()}")
        except FileNotFoundError:
            print("Twitter dataset not found. Please download the Twitter US Airline Sentiment dataset.")
            print("Creating sample data for demonstration...")
            twitter_df = pd.DataFrame({
                'airline_sentiment': ['positive', 'negative', 'neutral'] * 200,
                'text': [
                    'Great flight experience! Thank you @airline',
                    'Terrible service. Flight delayed for 3 hours!',
                    'Flight was okay. Nothing special.'
                ] * 200
            })
        
        # Initialize and train Twitter analyzer
        twitter_analyzer = TwitterSentimentAnalyzer()
        
        try:
            twitter_analyzer.load_word2vec_model(W2V_MODEL_PATH)
            twitter_accuracy = twitter_analyzer.train(twitter_df)
            
            # Test prediction function
            test_tweets = [
                "Amazing flight experience! The crew was fantastic!",
                "Worst airline ever. Lost my luggage and rude staff.",
                "Flight was on time. Standard service.",
                "@airline thanks for the upgrade! Great service!"
            ]
            
            print("\nTesting Twitter Sentiment Prediction Function:")
            for tweet in test_tweets:
                pred, prob = twitter_analyzer.predict_tweet_sentiment(tweet)
                print(f"Tweet: '{tweet}'")
                print(f"Sentiment: {pred} (Confidence: {prob:.4f})\n")
                
        except FileNotFoundError:
            print("Word2Vec model not found. Please download GoogleNews-vectors-negative300.bin")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo run this code successfully, you need:")
        print("1. Download the SMS Spam Collection dataset from Kaggle")
        print("2. Download the Twitter US Airline Sentiment dataset from Kaggle")
        print("3. Download the Google News Word2Vec model")
        print("4. Update the file paths in the code")

# Standalone prediction functions as requested
def predict_message_class(model, w2v_model, message):
    """
    Predict class for a single SMS message
    
    Args:
        model: Trained LogisticRegression classifier
        w2v_model: Pre-trained Word2Vec model
        message: Single message string
    
    Returns:
        Predicted class (spam or ham)
    """
    detector = SMSSpamDetector()
    detector.classifier = model
    detector.w2v_model = w2v_model
    
    prediction, _ = detector.predict_message_class(message)
    return prediction

def predict_tweet_sentiment(model, w2v_model, tweet):
    """
    Predict sentiment for a single tweet
    
    Args:
        model: Trained LogisticRegression classifier
        w2v_model: Pre-trained Word2Vec model
        tweet: Single tweet string
    
    Returns:
        Predicted sentiment (positive, negative, or neutral)
    """
    analyzer = TwitterSentimentAnalyzer()
    analyzer.classifier = model
    analyzer.w2v_model = w2v_model
    
    prediction, _ = analyzer.predict_tweet_sentiment(tweet)
    return prediction

if __name__ == "__main__":
    main()