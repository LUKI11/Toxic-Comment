# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:42:15 2018

@author: luki
"""

import re, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import  GlobalMaxPool1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import dataset
train = pd.read_csv('../database/train.csv')
test = pd.read_csv('../database/test.csv')
submission = pd.read_csv('../database/sample_submission.csv')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
#Data cleaning
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # Conver words to lowercase
        text = text.lower()
        # Delete non-alphanumeric characters
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text) 
        # restore abbreviation
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        # replace some special symbols
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')   
        comment_list.append(text)
    return comment_list

list_sentences_train = clean_text(train['comment_text'])
list_sentences_test = clean_text(test['comment_text'])
max_features = 20000
#tokenizer data
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
maxlen = 400
#pad data
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

#build LSTM model
def get_model():
    inp = Input(shape=(maxlen, ))
    embed_size = 128
    x = Embedding(len(tokenizer.word_index)+1, embed_size)(inp)
    #use LSTM
    x = LSTM(100, return_sequences=True,name='lstm_layer')(x)
    ## reshape the 3D tensor into a 2D one
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
# separate training dataset (80%) and testing dataset (20%)
x_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=2018)
model = get_model()
batch_size = 1024
epochs = 50
file_path="weights_base.best.hdf5"
#build callback function with patience = 10
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list = [checkpoint, early]
#Apply callback into model
history = model.fit(x_train,y_train, validation_data=(x_test,y_test),batch_size=batch_size, epochs=epochs,callbacks=callbacks_list)
model.summary()

# Show the trend of loss value and accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# apply model into testing dataset
scores = model.evaluate(x_test, y_test, verbose=0)
#print accuracy
print("LSTM Accuracy: %.2f%%"% (scores[1]*100))
#Apply model into test file and do prediction
y_test = model.predict([X_te], batch_size=1024, verbose=1)
submission[list_classes] = y_test
submission.to_csv('../result/submission_LSTM.csv', index=False)