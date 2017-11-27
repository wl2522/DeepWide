# Files
light GBM.ipynb: trains a gradient boosted decision tree (GBDT) model, which is then used to perform feature transformations in the wide (logistic regression) model

wide.ipynb: builds a baseline logistic regression model in Tensorflow

baseline.ipynb: builds a baseline multilayer perceptron model in Tensorflow with the following attributes:

- FEATURES: indicator_gender, indicator_city, indicator_language, indicator_tab, indicator_screen, indicator_source, indicator_registered

- PARAMETERS: hidden_units=[1024, 512, 256], optimizer=tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')), train_steps=600000, batch_size=100

Deep_model.ipynb: builds a deep and wide neural network in Tensorflow
