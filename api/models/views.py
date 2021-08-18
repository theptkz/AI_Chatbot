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

from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated



# Create your views here.
# Class based view to predict based on RNN model
class Models(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]


    def post(self, request, format=None):
        data = request.data
        for key in data:
            doc = data[key]
        doc = doc.lower()
        words = np.array([self.word2token(w) for w in word_tokenize(doc, format="text").split(' ')[:30] if (w in ModelsConfig.fasttext.wv.key_to_index and w != '')])
        words = pad_sequences([words], maxlen=30)
        response_dict = {}
        self.intentClassification(words, response_dict)
        self.entityClassification(words, response_dict)
        return Response(response_dict, status=200)

    def intentClassification(self, words, response_dict):
        RNN_predict = ModelsConfig.modelRNN.predict(words)
        f1 = np.where(RNN_predict>=0.5,1,0)
        for (a,b) in zip(ModelsConfig.intent_header, f1[0]):
            if b == 1:
                response_dict["Intent"] = a
                
    def entityClassification(self, words, response_dict):
        BiLSTM_predict = ModelsConfig.modelBiLSTM.predict(words)
        f1 = np.where(BiLSTM_predict>=0.5,1,0)
        for (a,b) in zip(ModelsConfig.entity_header, f1[0]):
            if b == 1:
                response_dict["Entity"] = a

    def word2token(self, word):
        try:
            return ModelsConfig.cbow_model.wv.key_to_index[word]
        except KeyError:
            return 0
    def token2word(self, token):
        return ModelsConfig.cbow_model.wv.index_to_key[token]
        