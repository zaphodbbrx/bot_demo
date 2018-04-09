#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:56:26 2018

@author: lsm
"""

# things we need for NLP
import nltk
from nltk.stem.snowball import RussianStemmer
stemmer = RussianStemmer()
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle


# import our chat-bot intents file
import json

    
class BotDemo():
    
    def __init__(self):
        data = pickle.load( open( "./notebooks/training_data", "rb" ) )
        self.__words = data['words']
        self.__classes = data['classes']
        self.__train_x = data['train_x']
        self.__train_y = data['train_y']
        with open('./notebooks/intents_ru.txt') as json_data:
            self.__intents = json.load(json_data)
        self.__model = self.__build_network()
        self.context = {}
        self.ERROR_THRESHOLD = 0.25
        self.context = {}
        
    def __build_network(self):
        # Build neural network
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(self.__train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.__train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        
        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir='./notebooks/tflearn_logs')
        model.load('./notebooks/model_ru.tflearn')
        return model
    def __clean_up_sentence(self,sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words
    
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def __bow(self,sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.__clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
    
        return(np.array(bag))
    def classify(self,sentence):
        # generate probabilities from the model
        results = self.__model.predict([self.__bow(sentence, self.__words)])[0]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.__classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list
    
    def response(self,sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        print(results)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.__intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            if not userID in self.context:
                                self.context[userID] = []
                            self.context[userID].append(i['context_set'])
    
                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in self.context and 'context_filter' in i and i['context_filter'] in self.context[userID][-2:]):
                            if show_details: print ('tag:', i['tag'])
                            # a random response from the intent
                            return print(random.choice(i['responses']))
                            #print(random.choice(i['responses']))
    
                results.pop(0)
if __name__ == '__main__':
    bot = BotDemo()
    while True:
        msg = input()
        print('\n')
        bot.response(msg,show_details = True)