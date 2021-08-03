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
    MLMODEL_FOLDER = os.path.join(BASE_DIR, 'models/intent_models/')
    MLMODEL_FILE = os.path.join(MLMODEL_FOLDER, "RNN")
    modelRNN = tf.keras.models.load_model(MLMODEL_FILE)
    # Load stopwords
    stopword_file = open(os.path.join(MLMODEL_FOLDER, "stopwords2.txt"), "r",encoding="utf8")
    stopwords = stopword_file.readlines()
    #Load CBOW model
    cbow_model = Word2Vec.load(os.path.join(MLMODEL_FOLDER,"word2vec_cbow.model"))
    #header
    header = ['Hello', 'Done', 'Connect', 'Order', 'Changing', 'Return', 'Other', 'Inform', 'Request', 'Feedback']