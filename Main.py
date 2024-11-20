from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from scipy.io import wavfile
import python_speech_features
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sounddevice as sd

import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, sosfilt
import pickle

main = tkinter.Tk()
main.title("Speech To Text")
main.geometry("1300x1200")

word2index = {
    # core words
    "down": 0,
    "eight": 1,
    "five": 2,
    "four": 3,
    "go": 4,
    "happy": 5,
    "house": 6,
    "left": 7,
    "marvin": 8,
    "nine": 9,
    "no": 10,
    "off": 11,
    "on": 12,
    "one": 13,
    "right": 14,
    "seven": 15,
    "sheila": 16,
    
}

index2word = [word for word in word2index]

global model, X, Y, recorded_feature

def loadModel():
    global model, X, Y
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
    text.delete('1.0', END)
    train_data, validation_data, train_classes, validation_classes = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(99, 20)))
    model.add(keras.layers.Conv1D(64, kernel_size=8, activation="relu"))
    model.add(keras.layers.MaxPooling1D(pool_size=3))
    model.add(keras.layers.Conv1D(128, kernel_size=8, activation="relu"))
    model.add(keras.layers.MaxPooling1D(pool_size=3))
    model.add(keras.layers.Conv1D(256, kernel_size=5, activation="relu"))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(17, activation='softmax'))
    model.summary()
    sgd = keras.optimizers.SGD()
    loss_fn = keras.losses.SparseCategoricalCrossentropy() # use Sparse because classes are represented as integers not as one-hot encoding
    model.compile(optimizer=sgd, loss=loss_fn, metrics=["accuracy"])
    model.load_weights("model/model.h5")
    print(model.summary())
    text.insert(END,"CNN model loaded\n\n")
    text.insert(END,"Total voices found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Different words used in dataset : "+str(word2index.keys())+"\n\n")
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[74] * 100
    text.insert(END,"Neural Network Training Accuracy = "+str(accuracy))

def audio2feature(audio):
    audio = audio.astype(np.float)
    # normalize data
    audio -= audio.mean()
    audio /= np.max((audio.max(), -audio.min()))
    # compute MFCC coefficients
    features = python_speech_features.mfcc(audio, samplerate=16000, winlen=0.025, winstep=0.01, numcep=20, nfilt=40, nfft=512, lowfreq=100, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
    return features

# load .wav-file, add some noise and compute MFCC features
def wav2feature(filepath):
    samplerate, data = wavfile.read(filepath)
    data = data.astype(np.float)
    # normalize data
    data -= data.mean()
    data /= np.max((data.max(), -data.min()))
    # add gaussian noise
    data += np.random.normal(loc=0.0, scale=0.025, size=data.shape)
    # compute MFCC coefficients
    features = python_speech_features.mfcc(data, samplerate=16000, winlen=0.025, winstep=0.01, numcep=20, nfilt=40, nfft=512, lowfreq=100, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
    return features

def extract_loudest_section(audio, length):
    audio = audio[:, 0].astype(np.float) # to avoid integer overflow when squaring
    audio_pw = audio**2 # power
    window = np.ones((length, ))
    conv = np.convolve(audio_pw, window, mode="valid")
    begin_index = conv.argmax()
    return audio[begin_index:begin_index+length]

def startRecording():
    global recorded_feature
    text.delete('1.0', END)
    samplerate = 16000  
    text.insert(END,"Please start recording\n\n")
    text.update_idletasks()
    recording = sd.rec(int(3 * samplerate), samplerate=samplerate, channels=1, dtype=np.float, blocking=True)
    print(recording.shape)
    recording = extract_loudest_section(recording, int(1*samplerate))
    sd.play(recording, blocking=True)
    recorded_feature = audio2feature(recording)
    recorded_feature = np.expand_dims(recorded_feature, 0) # add "fake" batch dimension 1
    
    print(recorded_feature.shape)
    text.insert(END,"Recording completed")
    text.update_idletasks()
        

def recognize():
    global recorded_feature, model
    text.delete('1.0', END)
    prediction = model.predict(recorded_feature).reshape((17, ))
    prediction /= prediction.sum()
    prediction_sorted_indices = prediction.argsort()
    print("candidates:\n-----------------------------")
    output = ""
    for k in range(3):
        i = int(prediction_sorted_indices[-1-k])
        output += index2word[i]+" "
    text.insert(END,"Recorded Speech Recognized As : "+output) 
    
font = ('times', 16, 'bold')
title = Label(main, text='Speech To Text using CNN Classifier')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')


loadButton = Button(main, text="Generate & Load Neural Network Model", command=loadModel)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

startButton = Button(main, text="Start Recording", command=startRecording)
startButton.place(x=50,y=150)
startButton.config(font=font1) 

recognizeButton = Button(main, text="Recognize Speech", command=recognize)
recognizeButton.place(x=250,y=150)
recognizeButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
