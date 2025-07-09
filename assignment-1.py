import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# ================================
# PROBLEM 1: NumPy Array Operations
# ================================

print("="*50)
print("PROBLEM 1: NumPy Array Operations")
print("="*50)

# Generate 2D array
np.random.seed(42)  # For reproducible results
array_2d = np.random.randint(1, 51, size=(5, 4))
print("Generated 2D Array:")
print(array_2d)

# Extract anti-diagonal elements (top-right to bottom-left)
anti_diagonal = []
rows, cols = array_2d.shape
for i in range(min(rows, cols)):
    anti_diagonal.append(array_2d[i, cols-1-i])
print(f"\nAnti-diagonal elements: {anti_diagonal}")

# Maximum value in each row
max_per_row = np.max(array_2d, axis=1)
print(f"Maximum value in each row: {max_per_row}")

# Elements less than or equal to overall mean
overall_mean = np.mean(array_2d)
print(f"Overall mean: {overall_mean:.2f}")
elements_le_mean = array_2d[array_2d <= overall_mean]
print(f"Elements <= mean: {elements_le_mean}")

def numpy_boundary_traversal(matrix):
    """
    Returns elements along the boundary of the matrix in clockwise order
    starting from top-left corner.
    """
    if matrix.size == 0:
        return []
    
    rows, cols = matrix.shape
    boundary = []
    
    if rows == 1:
        return matrix[0].tolist()
    if cols == 1:
        return matrix[:, 0].tolist()
    
    # Top row (left to right)
    boundary.extend(matrix[0, :].tolist())
    
    # Right column (top to bottom, excluding corners)
    boundary.extend(matrix[1:rows-1, cols-1].tolist())
    
    # Bottom row (right to left, excluding right corner)
    if rows > 1:
        boundary.extend(matrix[rows-1, ::-1][1:].tolist())
    
    # Left column (bottom to top, excluding corners)
    if cols > 1:
        boundary.extend(matrix[rows-2:0:-1, 0].tolist())
    
    return boundary

# Test boundary traversal
boundary_result = numpy_boundary_traversal(array_2d)
print(f"Boundary traversal: {boundary_result}")

# ================================
# PROBLEM 2: 1D NumPy Array Operations
# ================================

print("\n" + "="*50)
print("PROBLEM 2: 1D NumPy Array Operations")
print("="*50)

# Generate 1D array of random floats
np.random.seed(42)
array_1d = np.random.uniform(0, 10, 20)
print("Generated 1D Array:")
print(array_1d)

# Round to 2 decimal places
array_rounded = np.round(array_1d, 2)
print(f"\nRounded to 2 decimal places: {array_rounded}")

# Calculate statistics
min_val = np.min(array_1d)
max_val = np.max(array_1d)
median_val = np.median(array_1d)
print(f"Minimum: {min_val:.2f}")
print(f"Maximum: {max_val:.2f}")
print(f"Median: {median_val:.2f}")

# Replace elements < 5 with their squares
array_modified = array_1d.copy()
mask = array_modified < 5
array_modified[mask] = array_modified[mask] ** 2
print(f"\nArray after replacing elements < 5 with squares:")
print(np.round(array_modified, 2))

def numpy_alternate_sort(array):
    """
    Returns array sorted in alternating pattern: smallest, largest, 
    second smallest, second largest, etc.
    """
    sorted_array = np.sort(array)
    result = []
    left, right = 0, len(sorted_array) - 1
    
    while left <= right:
        if len(result) % 2 == 0:  # Even index: add smallest
            result.append(sorted_array[left])
            left += 1
        else:  # Odd index: add largest
            result.append(sorted_array[right])
            right -= 1
    
    return np.array(result)

# Test alternate sort
alt_sorted = numpy_alternate_sort(array_1d)
print(f"Alternate sorted array: {np.round(alt_sorted, 2)}")

# ================================
# PROBLEM 3: Pandas DataFrame Operations
# ================================

print("\n" + "="*50)
print("PROBLEM 3: Pandas DataFrame Operations")
print("="*50)

# Create student records DataFrame
np.random.seed(42)
subjects = ['Math', 'Science', 'English', 'History', 'Art']
students_data = {
    'Name': [f'Student_{i+1}' for i in range(10)],
    'Subject': [random.choice(subjects) for _ in range(10)],
    'Score': np.random.randint(50, 101, 10),
    'Grade': ['' for _ in range(10)]
}
df_students = pd.DataFrame(students_data)

# Assign grades based on scores
def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df_students['Grade'] = df_students['Score'].apply(assign_grade)
print("Student Records DataFrame:")
print(df_students)

# Sort by score in descending order
df_sorted = df_students.sort_values('Score', ascending=False)
print(f"\nDataFrame sorted by Score (descending):")
print(df_sorted)

# Average score for each subject
avg_by_subject = df_students.groupby('Subject')['Score'].mean()
print(f"\nAverage score by subject:")
print(avg_by_subject)

def pandas_filter_pass(dataframe):
    """
    Returns DataFrame containing only records with grades A or B.
    """
    return dataframe[dataframe['Grade'].isin(['A', 'B'])].copy()

# Test filter function
passed_students = pandas_filter_pass(df_students)
print(f"\nStudents with grades A or B:")
print(passed_students)

# ================================
# PROBLEM 4: Movie Review Classification
# ================================

print("\n" + "="*50)
print("PROBLEM 4: Movie Review Classification")
print("="*50)

# Create synthetic movie reviews with variety
np.random.seed(42)

# Positive adjectives, nouns, and phrases for variation
positive_words = ["fantastic", "amazing", "excellent", "brilliant", "outstanding", "superb", "incredible", "wonderful", "great", "perfect"]
positive_nouns = ["movie", "film", "story", "acting", "direction", "cinematography", "plot", "performance", "screenplay", "experience"]
positive_phrases = ["highly recommend", "must watch", "loved every minute", "exceeded expectations", "masterpiece", "top-notch", "well done", "impressive work", "captivating", "engaging"]

negative_words = ["terrible", "awful", "horrible", "disappointing", "boring", "poor", "bad", "weak", "mediocre", "pathetic"]
negative_nouns = ["movie", "film", "story", "acting", "direction", "cinematography", "plot", "performance", "screenplay", "experience"]
negative_phrases = ["waste of time", "not recommended", "poorly executed", "completely boring", "major disappointment", "poorly written", "lacks depth", "uninspiring", "forgettable", "overrated"]

# Generate 50 unique positive reviews
positive_reviews = []
for i in range(50):
    templates = [
        f"This {random.choice(positive_nouns)} was {random.choice(positive_words)} and {random.choice(positive_phrases)}",
        f"{random.choice(positive_words).title()} {random.choice(positive_nouns)} with {random.choice(positive_words)} {random.choice(positive_nouns)}",
        f"Really {random.choice(positive_words)} {random.choice(positive_nouns)}, {random.choice(positive_phrases)}",
        f"The {random.choice(positive_nouns)} was {random.choice(positive_words)}, {random.choice(positive_phrases)}",
        f"{random.choice(positive_words).title()} and {random.choice(positive_words)} {random.choice(positive_nouns)}"
    ]
    positive_reviews.append(random.choice(templates))

# Generate 50 unique negative reviews
negative_reviews = []
for i in range(50):
    templates = [
        f"This {random.choice(negative_nouns)} was {random.choice(negative_words)} and {random.choice(negative_phrases)}",
        f"{random.choice(negative_words).title()} {random.choice(negative_nouns)} with {random.choice(negative_words)} {random.choice(negative_nouns)}",
        f"Really {random.choice(negative_words)} {random.choice(negative_nouns)}, {random.choice(negative_phrases)}",
        f"The {random.choice(negative_nouns)} was {random.choice(negative_words)}, {random.choice(negative_phrases)}",
        f"{random.choice(negative_words).title()} and {random.choice(negative_words)} {random.choice(negative_nouns)}"
    ]
    negative_reviews.append(random.choice(templates))

# Create DataFrame
reviews_data = {
    'Review': positive_reviews + negative_reviews,
    'Sentiment': ['positive'] * 50 + ['negative'] * 50
}
df_reviews = pd.DataFrame(reviews_data)

# Shuffle the data
df_reviews = df_reviews.sample(frac=1, random_state=42).reset_index(drop=True)

print("Movie Reviews Dataset:")
print(df_reviews.head())

# Tokenize using CountVectorizer
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df_reviews['Review'])
y = df_reviews['Sentiment']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Test accuracy
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nNaive Bayes Accuracy: {accuracy:.2f}")

def predict_review_sentiment(model, vectorizer, review):
    """
    Predicts sentiment for a single review.
    """
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)[0]
    return prediction

# Test prediction function
test_review = "This movie was absolutely amazing and wonderful"
predicted_sentiment = predict_review_sentiment(nb_model, vectorizer, test_review)
print(f"Test review: '{test_review}'")
print(f"Predicted sentiment: {predicted_sentiment}")

# ================================
# PROBLEM 5: Text Classification with TF-IDF
# ================================

print("\n" + "="*50)
print("PROBLEM 5: Text Classification with TF-IDF")
print("="*50)

# Create synthetic text feedback dataset with variety
np.random.seed(42)

# Product feedback vocabulary for variation
good_adjectives = ["excellent", "amazing", "fantastic", "great", "outstanding", "perfect", "wonderful", "superb", "brilliant", "impressive"]
good_nouns = ["product", "item", "service", "quality", "features", "design", "performance", "experience", "value", "support"]
good_phrases = ["works perfectly", "highly recommend", "exceeded expectations", "great value", "fast delivery", "easy to use", "reliable", "top quality", "satisfied", "impressed"]

bad_adjectives = ["terrible", "awful", "poor", "disappointing", "horrible", "bad", "defective", "useless", "overpriced", "unreliable"]
bad_nouns = ["product", "item", "service", "quality", "features", "design", "performance", "experience", "value", "support"]
bad_phrases = ["doesn't work", "not recommended", "waste of money", "poor quality", "slow delivery", "difficult to use", "unreliable", "many problems", "disappointed", "regret buying"]

# Generate 50 unique good feedback samples
good_feedback = []
for i in range(50):
    templates = [
        f"This {random.choice(good_nouns)} is {random.choice(good_adjectives)} and {random.choice(good_phrases)}",
        f"{random.choice(good_adjectives).title()} {random.choice(good_nouns)} with {random.choice(good_adjectives)} {random.choice(good_nouns)}",
        f"Really {random.choice(good_adjectives)} {random.choice(good_nouns)}, {random.choice(good_phrases)}",
        f"The {random.choice(good_nouns)} {random.choice(good_phrases)} and has {random.choice(good_adjectives)} {random.choice(good_nouns)}",
        f"{random.choice(good_adjectives).title()} {random.choice(good_nouns)}, {random.choice(good_phrases)}"
    ]
    good_feedback.append(random.choice(templates))

# Generate 50 unique bad feedback samples
bad_feedback = []
for i in range(50):
    templates = [
        f"This {random.choice(bad_nouns)} is {random.choice(bad_adjectives)} and {random.choice(bad_phrases)}",
        f"{random.choice(bad_adjectives).title()} {random.choice(bad_nouns)} with {random.choice(bad_adjectives)} {random.choice(bad_nouns)}",
        f"Really {random.choice(bad_adjectives)} {random.choice(bad_nouns)}, {random.choice(bad_phrases)}",
        f"The {random.choice(bad_nouns)} {random.choice(bad_phrases)} and has {random.choice(bad_adjectives)} {random.choice(bad_nouns)}",
        f"{random.choice(bad_adjectives).title()} {random.choice(bad_nouns)}, {random.choice(bad_phrases)}"
    ]
    bad_feedback.append(random.choice(templates))

# Create DataFrame
feedback_data = {
    'Text': good_feedback + bad_feedback,
    'Label': ['good'] * 50 + ['bad'] * 50
}
df_feedback = pd.DataFrame(feedback_data)

# Shuffle the data
df_feedback = df_feedback.sample(frac=1, random_state=42).reset_index(drop=True)

print("Text Feedback Dataset:")
print(df_feedback.head())

# Preprocess using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df_feedback['Text'])
y_tfidf = df_feedback['Label']

# Split into train/test sets (75/25)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y_tfidf, test_size=0.25, random_state=42
)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_tfidf, y_train_tfidf)

# Predict and calculate metrics
y_pred_tfidf = lr_model.predict(X_test_tfidf)

precision = precision_score(y_test_tfidf, y_pred_tfidf, pos_label='good')
recall = recall_score(y_test_tfidf, y_pred_tfidf, pos_label='good')
f1 = f1_score(y_test_tfidf, y_pred_tfidf, pos_label='good')

print(f"\nLogistic Regression Results:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

def text_preprocess_vectorize(texts, vectorizer):
    """
    Takes a list of text samples and a fitted TfidfVectorizer,
    returns the vectorized feature matrix.
    """
    return vectorizer.transform(texts)

# Test the function
test_texts = ["This is a great product", "Poor quality item"]
vectorized_texts = text_preprocess_vectorize(test_texts, tfidf_vectorizer)
print(f"\nVectorized shape for test texts: {vectorized_texts.shape}")

print("\n" + "="*50)
print("ALL PROBLEMS COMPLETED SUCCESSFULLY!")
print("="*50)