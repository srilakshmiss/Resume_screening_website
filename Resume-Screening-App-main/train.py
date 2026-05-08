import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("UpdatedResumeDataSet.csv")

# =====================================================
# DATA
# =====================================================

X = df["Resume"]
y = df["Category"]

# =====================================================
# LABEL ENCODER
# =====================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# =====================================================
# TFIDF
# =====================================================

tfidf = TfidfVectorizer(stop_words="english")

X = tfidf.fit_transform(X)

# =====================================================
# MODEL
# =====================================================

model = LinearSVC()

model.fit(X, y)

# =====================================================
# SAVE NEW FILES
# =====================================================

pickle.dump(model, open("clf.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("NEW MODEL CREATED SUCCESSFULLY")