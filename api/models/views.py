from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.http import HttpResponse
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
            dic = {
                "Hello": {
                    'Xin chào quý khách, Quý khách muốn giúp gì ạ?'
                },
                'Request': {
                    'material_product':{
                        "D004" : "z2175499982929_d03e94ed06cb59e16edcdfa1d84e924c.jpg",
                        "D005" : "z2175499986670_a4078f484d6acb3143c4c92652180887.jpg",
                        "D006" : "z2175499985166_eafa64a3277e2e747e5bdfdd4eedb84b.jpg",
                        "D007" : "z2180526406642_36a29c92992682ab33c292f79c7534fc.jpg",
                        "D008" : "z2180323514075_f6a47512101bbb489554631bc4bfad87.jpg",
                        "D009" : "z2230876375867_6cd92f45362611c27fcc4cc17e26198d.jpg",
                        "D0010": "z2175499992329_b3388f4a39b235685f77c3504102f517.jpg",
                        "D0011": "z2175499979745_222feb2973bcadb252b264ad6f9bf527.jpg",
                        "D0012": "z2175500007986_984d8e4bc8163715a342baea6451428d.jpg",
                        "D0013": "z2175500017925_c17a5a03cc2aec8566446f848e5e609e (1).jpg",
                        "D0014": "690e39b7f63708695126.jpg",
                        "D0015": "z2175500004059_a3f0ffa6cd860aa206e14d52f5f018ae.jpg",
                        "D0016": "z2230702076529_77708c6f3e4bc00799757ece8d9bd211.jpg",
                        "D0017": "z2230702076529_77708c6f3e4bc00799757ece8d9bd211.jpg",
                        "D0018": "z2175500015009_1f4f684f510a3b6d8f4c8b8e276d496a.jpg",
                        "DS001": "z2183724133835_9355f46d7d06c8f5dedc28ff655083ae.jpg",
                        "S002" : "z2175500003782_54b876a9644618e63b2bb6ed8d62a619.jpg",
                        "S003" : "material.jpg",
                        "S004" : "z2175499979305_55bb279f73b621274cabb00ebadcc27e.jpg",
                        "S005" : "z2180272131774_1c3656046dd7994d9a69caf1ea320173.jpg",
                        "S006" : "material.jpg",
                        "S008" : "z2230686696861_90aa644a93f748eab4a6acf05e1cd294.jpg",
                        "S009" : "z2230671189657_c54eff0a6d63e9dfd2336105b44a5c11.jpg",
                    },
                    'cost_product':{
                        "D004" : "190K",
                        "D005" : "180K",
                        "D006" : "170K",
                        "D007" : "160K",
                        "D008" : "195K",
                        "D009" : "185K",
                        "D0010": "175K",
                        "D0011": "165K",
                        "D0012": "200K",
                        "D0013": "210K",
                        "D0014": "205K",
                        "D0015": "215K",
                        "D0016": "220K",
                        "D0017": "187K",
                        "D0018": "189K",
                        "DS001": "193K",
                        "S002" : "196K",
                        "S003" : "179K",
                        "S004" : "172K",
                        "S005" : "188K",
                        "S006" : "199K",
                        "S008" : "208K",
                        "S009" : "212K",
                    },
                    'product_image':{
                        "D004" : "z2183768097681_f5a43a82a7037ecdf5978f84f87da4d1.jpg",
                        "D005" : "z2180410582428_510eac43d7eb6a7a1131c2354556d0c0.jpg",
                        "D006" : "z2180429885184_a89767736b20f0a5ad4f09ea1666fd72.jpg",
                        "D007" : "z2180429884238_c302affbc9a526c137b4acef8f2d3e20.jpg",
                        "D008" : "z2175415733096_ef0a2433ea235291d28fb407d6e9f2c3.jpg",
                        "D009" : "z2225628668198_6fa2b97c9986e92327cf8f6f39835af1.jpg",
                        "D0010": "z2232493465950_f9244d0919877997daf2bc64ed211985.jpg",
                        "D0011": "z2183821269081_6eff94c1e699dc385ad2fe632dd7d6cc.jpg",
                        "D0012": "z2175429683236_32de35980a341ce098ba335b4807ec09.jpg",
                        "D0013": "z2180540922583_f4f5328b550da5ab0fe1b32f9ce2351b.jpg",
                        "D0014": "z2175462777538_7497f756083b0602b00e102fcf4a8373.jpg",
                        "D0015": "z2183680459012_568d1ce32beb915fda61172cc888f6cb.jpg",
                        "D0016": "z2225630358685_38ce3b9fbe9eda7f8bc9e6d79f7de3f2.jpg",
                        "D0017": "z2198879386447_89067176632d0e3e12593a00c85f021e.jpg",
                        "D0018": "z2180334332530_0d9abf53b14f2984eeea3dfcfc9a350a.jpg",
                        "DS001": "z2175429551250_23d8b647a4945fba5b3b8103593110e6.jpg",
                        "S002" : "z2175424020355_92d68a3c3f945de2e209ded92491caea.jpg",
                        "S003" : "z2183855692498_b7ec8c15225a0548de4877f1987d3ac9.jpg",
                        "S004" : "z2175429689122_b6223679718663ecaed23599589626a6.jpg",
                        "S005" : "z2180241130710_be8ade4348ff558729f6aa73ddf8aecf.jpg",
                        "S006" : "z2198883752716_67857ac1f624d8d04ab1f5a869376b9b.jpg",
                        "S008" : "z2225560215629_f4db13d1c00e722b467d040ebad8e3df.jpg",
                        "S009" : "z2225629629623_4efd0d1f7755b6e20a82578ddc5b77b8.jpg",
                    },
                    'amount_product':{
                        "D004" : "Còn hàng ạ",
                        "D005" : "Hết hàng ạ",
                        "D006" : "Hết hàng ạ",
                        "D007" : "Còn hàng ạ",
                        "D008" : "Hết hàng ạ",
                        "D009" : "Còn hàng ạ",
                        "D0010": "Hết hàng ạ",
                        "D0011": "Còn hàng ạ",
                        "D0012": "Hết hàng ạ",
                        "D0013": "Còn hàng ạ",
                        "D0014": "Hết hàng ạ",
                        "D0015": "Còn hàng ạ",
                        "D0016": "Hết hàng ạ",
                        "D0017": "Còn hàng ạ",
                        "D0018": "Còn hàng ạ",
                        "DS001": "Còn hàng ạ",
                        "S002" : "Còn hàng ạ",
                        "S003" : "Còn hàng ạ",
                        "S004" : "Hết hàng ạ",
                        "S005" : "Còn hàng ạ",
                        "S006" : "Còn hàng ạ",
                        "S008" : "Còn hàng ạ",
                        "S009" : "Hết hàng ạ",
                    },
                },
                'Inform':{
                    'size':{
                        'Sản phẩm có các size S,M,L ạ.'
                    }
                },
                'Done':{
                    'Tạm biệt'
                }
            }
            print(intent, " ", entity)
            if entity:
                if intent == 'Request':
                    if entity in ['product_image', 'material_product']:
                        file_url = settings.MEDIA_ROOT + '/Hume/' + request.session['productID'] + '/' + dic[intent][entity][request.session['productID']]
                        print(file_url)
                        try:    
                            with open( file_url, 'rb') as f:
                                file_data = f.read()
                                print(file_data)
                                # sending response 
                                response = HttpResponse(file_data, content_type='image/jpeg')
                        except IOError:
                            # handle file not exist case here
                            response = Response(status=400)
                    else:
                        response_dict['res'] = dic[intent][entity][request.session['productID']]
                        response = Response(response_dict, status=200)
                    
                else:
                    response_dict['res'] = dic[intent][entity]
                    response = Response(response_dict, status=200)
            else:
                response_dict['res'] = dic[intent]
                response = Response(response_dict, status=200)
                
            return response
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


    def word2token(self, word):
        try:
            return ModelsConfig.fasttext.wv.key_to_index[word]
        except KeyError:
            return 0
class Image(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, format=None):
        if request.method == 'POST' and request.FILES:
            filename = request.FILES['file'].name
            image = request.FILES['file']
            response_dict = {}
            
            with default_storage.open('tmp/' + filename, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            self.imageClassification(settings.MEDIA_ROOT + '/tmp/' + filename, response_dict)
            request.session['productID'] = response_dict['image_class']
            print(request.session['productID'])
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