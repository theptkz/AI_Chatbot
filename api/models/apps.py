from django.apps import AppConfig
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import gensim
from gensim import models
from gensim.models import FastText
import os

class ModelsConfig(AppConfig):
    name = 'models'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MLMODEL_FOLDER = os.path.join(BASE_DIR, 'models/models/')

    modelRNN = tf.keras.models.load_model(os.path.join(MLMODEL_FOLDER, "RNN"))
    modelBiLSTM = tf.keras.models.load_model(os.path.join(MLMODEL_FOLDER, "BiLSTM"), compile=False)
    modelCNN = tf.keras.models.load_model(os.path.join(MLMODEL_FOLDER, "CNN"))
    # Load stopwords
    stopword_file = open(os.path.join(MLMODEL_FOLDER, "stopwords2.txt"), "r",encoding="utf8")
    stopwords = stopword_file.readlines()
    #Load CBOW model
    fasttext = FastText.load(os.path.join(MLMODEL_FOLDER,"fasttext.model"))
    #header
    intent_header = ['Hello', 'Done', 'Connect', 'Order', 'Changing', 'Return', 'Other', 'Inform', 'Request', 'Feedback']
    entity_header = ['ID_product','color_product', 'material_product','cost_product','amount_product','Id member', 'shipping','height customer','weight customer','phone', 'address', 'size', 'Time', 'product_image']
    image_header = ['D0010', 'D0011', 'D0012', 'D0013', 'D0014', 'D0015', 'D0016', 'D0017', 'D003', 'D004', 'D005', 'D006', 'D007 ', 'D008', 'D009 ', 'DS001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S008 ', 'S009']