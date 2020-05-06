import random
import tensorflow as tf
import tflearn as tfl
import json as js
import numpy as np
import nltk
import os
import shutil
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from nltk.stem.rslp import RSLPStemmer
from googletrans import Translator


nltk.download('rslp')
with open("intents.json") as file:
    data = js.load(file)

palavras = []
intencoes = []
sentencas = []
saidas = []

for intent in data["intents"]:

    tag = intent['tag']

    if tag not in intencoes:
        intencoes.append(tag)

    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern, language='portuguese')
        palavras.extend(wrds)
        sentencas.append(wrds)
        saidas.append(tag)

stemer = RSLPStemmer()

stemmed_words = [stemer.stem(w.lower()) for w in palavras]
stemmed_words = sorted(list(set(stemmed_words)))

training = []
output = []
# criando um array preenchido com 0
outputEmpty = [0 for _ in range(len(intencoes))]

for x, frase in enumerate(sentencas):
    bag = []
    wds = [stemer.stem(k.lower()) for k in frase]
    for w in stemmed_words:
        if w in wds:
            bag.append(1)
        else:
            bag.append(0)

    outputRow = outputEmpty[:]
    outputRow[intencoes.index(saidas[x])] = 1

    training.append(bag)
    output.append(outputRow)

training = np.array(training)
output = np.array(output)

# reiniciando os dados
tf.reset_default_graph()

# camada de entrada
net = tfl.input_data(shape=[None, len(training[0])])
# neuronios por camada oculta
net = tfl.fully_connected(net, 32)
# camada de saida
net = tfl.fully_connected(net, len(output[0]), activation="softmax")
#
net = tfl.regression(net)

# criando o modelo
model = tfl.DNN(net)

model.fit(training, output, n_epoch=80, batch_size=8, show_metric=True)
model.save("model.chatbot30G")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Funcao responsavel por falar


def cria_audio_bot(text):
    tts = gTTS(text, lang='pt-br')
    # Salva o arquivo de audio
    stringNomeAudio = str('audios/audiobot'+str(random.random())+'.mp3')
    tts.save(stringNomeAudio)
    # Da play ao audio
    print(text)
    playsound(stringNomeAudio)

# Funcao responsavel por ouvir e reconhecer a fala


def ouvir_microfone():
    # Habilita o microfone para ouvir o usuario
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        # Chama a funcao de reducao de ruido disponivel na speech_recognition
        microfone.adjust_for_ambient_noise(source)
        # Avisa ao usuario que esta pronto para ouvir
        print("Diga alguma coisa: ")
        # Armazena a informacao de audio na variavel
        audio = microfone.listen(source)

    try:
        # Passa o audio para o reconhecedor de padroes do speech_recognition
        frase = microfone.recognize_google(audio, language='pt-BR')
        # Após alguns segundos, retorna a frase falada
        print("Você disse: " + frase)

        # Caso nao tenha reconhecido o padrao de fala, exibe esta mensagem
    except sr.UnknownValueError:
        frase = "Não entendi"
        cria_audio_bot(frase)

    return frase


def chat():
    if os.path.exists('./audios'):
        shutil.rmtree('./audios/')
        os.mkdir('./audios')
    else:
        os.mkdir('./audios')
    print("Esse é o bot de teste! Converse com ele")
    Online = True
    while Online:
        inp = ouvir_microfone()
        bag_usuario = bag_of_words(inp, stemmed_words)
        results = model.predict([bag_usuario])
        results_index = np.argmax(results)
        tag = intencoes[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                cria_audio_bot(random.choice(responses))
        if tag == "agradecimento":
            Online = False
            shutil.rmtree('./audios/')


chat()
