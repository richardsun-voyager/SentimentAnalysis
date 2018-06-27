# SentimentAnalysis

In this project, we explored different sentiment analysis models and aimed to design one model that could reach start-of-art performance. We realized such models:

- TfIdf + Logistic, lightGBM model
- Word Embeddings + 3-kernels two-layer CNN model
- Word Embeddings + Doc2vec Model
- Word Embeddings + bidirectional GRU Model
- Word Embeddings + CNN + GRU Model
- Word Embeddings + self attention +bidirectional GRU Model

**The accuracy on testing data set can be achieved 96% for CNN+GRU model, and 90% for TfIdf+Logistic Model.**



We preprocessed the original texts using such methods:

- remove punctuation like ',', '.'
- normalize words, particularly lemmatize for example, map 'does, did, done' to 'do'
- Cut longer sentences if their lengths exceed a specific number
- Select most frequent words and replace the less common ones as 'unknown'



Other techiniques:

- Use mask for CNN layers, that means for the sentences whose lengths are shorter than a specific number, we pad their embeddings as 0s.
- Use dynamic GRU model for RNN



Packages:

- NLTK
- Spacy, English module
- Tensorflow
- Scikit-learn
- Numpy
- Pandas