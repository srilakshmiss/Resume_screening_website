import pickle

model = pickle.load(open("clf.pkl", "rb"))

print(type(model))