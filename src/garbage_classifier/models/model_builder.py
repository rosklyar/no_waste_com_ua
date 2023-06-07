from torch import nn
from garbage_classifier.models.beit.model import BeitImageClassificationModel, BeitImageClassificationProcessor
from garbage_classifier.models.image_processor import ImageProcessor
from garbage_classifier.models.model_card import ModelCard

def create_model(model_name: str, dropout_rate: float, fc_layer_size: int, n_classes: int) -> ModelCard: 
    if model_name == 'beit':
        return BeitImageClassificationModel('microsoft/beit-base-patch16-224-pt22k-ft22k', dropout_rate, fc_layer_size, n_classes)
    
def create_image_processor(processor_name: str) -> ImageProcessor:
    if processor_name == "beit":
        return  BeitImageClassificationProcessor("microsoft/beit-base-patch16-224-pt22k-ft22k")