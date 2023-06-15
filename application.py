from flask import Flask, render_template, request, url_for
import json
import pickle
from flask import Flask





import re
from wordcloud import WordCloud, STOPWORDS
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
from keras.utils import pad_sequences
from keras.models import load_model
model = load_model('mijo.h5')

with open('tokemiiizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
application  =  Flask(__name__)

@application.route('/')
def home():
    return render_template('index.html')
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return result
def predictar(news):
    ligma = preprocess(news)
    train_sequences = tokenizer.texts_to_sequences([ligma])
    padded_train = pad_sequences(train_sequences, maxlen=128, padding='post', truncating='post')
    input_data = [padded_train]
    prediction = model.predict(input_data)
    return prediction
@application.route('/predict',methods=['POST'])
def predict():
 if request.method == 'POST':
     message = request.json['message']
     pred    = predictar(message)[0][0]
     print(pred)
     result = "True" if pred > 0.5 else "False"
     return {'pred': result, 'confidence' :round(float(pred),3)}
    #  return render_template('index.html', pred=result, might=might)




if __name__ == '__main__':
    application.run(host="0.0.0.0",port=5000,debug=True)
    # application.run(port=5000,debug=False)/