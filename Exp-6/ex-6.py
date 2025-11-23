#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# In[6]:


get_ipython().system('pip install nltk')



# In[8]:


data = {
"review": [
"I love this movie! It is amazing and wonderful",
"This film was terrible. I hated every moment",
"What a fantastic experience, absolutely loved it!",
"Worst movie ever, waste of time",
"The story was good but acting was bad"
],
"sentiment": [1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)
print(df)


# In[9]:


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# In[19]:


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# In[20]:


df['clean_review'] = df['review'].apply(preprocess_text)
print("\nPreprocessed Reviews:\n", df['clean_review'])


# In[21]:


max_words = 1000
max_len = 20

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['clean_review'])
sequences = tokenizer.texts_to_sequences(df['clean_review'])


# In[22]:


X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df['sentiment'])


# In[23]:


print("\nTokenized & Padded:\n", X)


# In[24]:


model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=16, input_length=max_len))
model.add(SimpleRNN(16))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=2, verbose=1)


# In[25]:


test_sentence = "The movie was awesome and fantastic"
test_clean = preprocess_text(test_sentence)
test_seq = tokenizer.texts_to_sequences([test_clean])

test_pad = pad_sequences(test_seq, maxlen=max_len)


# In[26]:


prediction = model.predict(test_pad)[0][0]
print("\nTest Review:", test_sentence)
print("Cleaned Review:", test_clean)
print("Predicted Sentiment:", "Positive 😊" if prediction > 0.5 else "Negative 😞")


# In[ ]:





# In[ ]:





# In[ ]:




