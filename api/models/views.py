from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from .apps import ModelsConfig
import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import underthesea
from underthesea import word_tokenize
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated



# Create your views here.
# Class based view to predict based on RNN model
class Models(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]


    def post(self, request, format=None):
        if request.method == 'POST' and request.data['message']:
            doc = request.data['message']
            doc = doc.lower()
            words = np.array([self.word2token(w) for w in word_tokenize(doc, format="text").split(' ')[:30] if (w in ModelsConfig.fasttext.wv.key_to_index)])
            words = pad_sequences([words], maxlen=30)
            response_dict = {}
            intent = self.intentClassification(words, response_dict)
            entity = self.entityClassification(words, response_dict)
            res = self.botResponse(intent, entity, None)
            response_dict['res'] = res
            
            return Response(response_dict, status=200)
        else:
            return Response(status=400)

    def intentClassification(self, words, response_dict):
        RNN_predict = ModelsConfig.modelRNN.predict(words)
        f1 = np.where(RNN_predict>=0.5,1,0)
        for (a,b) in zip(ModelsConfig.intent_header, f1[0]):
            if b == 1:
                return a
        return None
                
    def entityClassification(self, words, response_dict):
        BiLSTM_predict = ModelsConfig.modelBiLSTM.predict(words)
        f1 = np.where(BiLSTM_predict>=0.5,1,0)
        for (a,b) in zip(ModelsConfig.entity_header, f1[0]):
            if b == 1:
                return a
        return None
    def botResponse(self, intent, entity, product):
        dic = {
            "Hello": {
                'Xin chào quý khách, Quý khách muốn giúp gì ạ?'
            },
            'Request': {
                'material_product':{
                    'material_product'
                },
                'cost_product':{
                    'cost_product'
                },
                'product_image':{
                    'product_image'
                },
                'amount_product':{
                    'amount_product'
                },
            },
            'Inform':{
                'size':{
                    'size'
                }
            },
            'Done':{
                'Tạm biệt'
            }

        }
        print(intent, " ", entity)
        response = dic[intent][entity] if entity else dic[intent]
        return response


    def word2token(self, word):
        try:
            return ModelsConfig.fasttext.wv.key_to_index[word]
        except KeyError:
            return 0
class Image(APIView):
    def post(self, request, format=None):
        if request.method == 'POST' and request.FILES:
            filename = request.FILES['file'].name
            image = request.FILES['file']
            response_dict = {}
            
            with default_storage.open('tmp/' + filename, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            self.imageClassification(settings.MEDIA_ROOT + '/tmp/' + filename, response_dict)   
            return Response(response_dict, 200)
        
        else:
            return Response(status = 400)

    def imageClassification(self, filepath, response_dict):
        img_height = 180
        img_width = 180
        
        img = keras.preprocessing.image.load_img(filepath, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = ModelsConfig.modelCNN.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        response_dict['image_class'] = ModelsConfig.image_header[np.argmax(score)]