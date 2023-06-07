from transformers import BeitForImageClassification, BeitImageProcessor
from torch import nn
from garbage_classifier.models.image_processor import ImageProcessor

class BeitImageClassificationModel(nn.Module):
    
    def __init__(self, model_name, dropout_rate, fc_layer_size, n_classes):
        super().__init__()
        self.model = BeitForImageClassification.from_pretrained(model_name)
        self.model.config.num_labels = n_classes
        self.model.classifier = nn.Sequential(
            nn.Linear(768, fc_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_layer_size, n_classes)
        )

    def forward(self, x):
        return self.model(x)
    
class BeitImageClassificationProcessor(ImageProcessor):
        
        def __init__(self, model_name):
            super().__init__()
            self.processor = BeitImageProcessor.from_pretrained(model_name)
            
        def process(self, image):
            return self.processor(images=image, return_tensors="pt")['pixel_values']
        
        def __repr__(self):
            return self.__class__.__name__ + '()'


