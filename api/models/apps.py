from django.apps import AppConfig
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import gensim
from gensim import models
from gensim.models import Word2Vec
import os

class ModelsConfig(AppConfig):
    name = 'models'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MLMODEL_FOLDER = os.path.join(BASE_DIR, 'models/models/')

    modelRNN = tf.keras.models.load_model(os.path.join(MLMODEL_FOLDER, "RNN"))
    modelBiLSTM = tf.keras.models.load_model(os.path.join(MLMODEL_FOLDER, "BiLSTM"), compile=False)
    # Load stopwords
    stopword_file = open(os.path.join(MLMODEL_FOLDER, "stopwords2.txt"), "r",encoding="utf8")
    stopwords = stopword_file.readlines()
    #Load CBOW model
    cbow_model = Word2Vec.load(os.path.join(MLMODEL_FOLDER,"word2vec_cbow.model"))
    #header
    intent_header = ['Hello', 'Done', 'Connect', 'Order', 'Changing', 'Return', 'Other', 'Inform', 'Request', 'Feedback']
    entity_header = ['ID_product','color_product', 'meterial_product','cost_product','amount_product','Id member', 'shipping','height customer','weight customer','phone', 'address', 'size', 'Time', 'product_image']