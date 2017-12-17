## Deep or Wide? Why Not Both?

# Our project is concerned with predicting whether or not a KKBox user will play a song again within a month of playing it for the first time. This problem is a binary classification task where we make predictions based on each user’s demographic profile as well as each song’s metadata. Our goal is to implement a wide and deep recommender system in TensorFlow that pairs a linear classifier with a fully-connected neural network. We then attempt to improve on this model architecture to achieve competitive results in the KKBox Music Recommendation Challenge on Kaggle.

# Files

wide_and_deep.ipynb: trains a wide and deep model using predictions made by a pretrained gradient boosted decision tree (GBDT) model as a feature

light GBM.ipynb: trains a GBDT model, which is then used to perform feature transformations in the wide and deep model
