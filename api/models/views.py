from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from models.apps import ModelsConfig
import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import underthesea
from underthesea import word_tokenize


# Create your views here.
# Class based view to predict based on RNN model
class Intent_Model(APIView):
    def post(self, request, format=None):
        data = request.data
        for key in data:
            doc = data[key]
        words = np.array([self.word2token(w) for w in word_tokenize(doc, format="text").split(' ')[:50] if (w in ModelsConfig.cbow_model.wv.key_to_index and w != '' and ((w +'\n') not in ModelsConfig.stopwords))])
        words = np.pad(words, (50 - len(words)%50 , 0), 'constant')
        a = ModelsConfig.modelRNN.predict(words.reshape((1, -1)) )
        f = np.where(a>=0.5,1,0)
        for (a,b) in zip(ModelsConfig.header, f[0]):
            if b == 1:
                response_dict = {"Intent": a}
        return Response(response_dict, status=200)


    def word2token(self, word):
        try:
            return ModelsConfig.cbow_model.wv.key_to_index[word]
        except KeyError:
            return 0
    def token2word(self, token):
        return ModelsConfig.cbow_model.wv.index_to_key[token]
        