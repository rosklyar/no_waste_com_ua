from transformers import BeitForImageClassification, BeitFeatureExtractor
from torch import nn
from garbage_classifier.models.image_processor import ImageProcessor
from garbage_classifier.models.model_card import ModelCard, create_model_card

class BeitImageClassificationModel(ModelCard):
    
    def __init__(self, model_name, dropout_rate, fc_layer_size, n_classes):
        super().__init__()
        self.model = BeitForImageClassification.from_pretrained(model_name)
        self.model.config.num_labels = n_classes
        self.model.classifier = nn.Sequential(
            nn.Linear(768, fc_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_layer_size, n_classes)
        )
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.fc_layer_size = fc_layer_size
        self.n_classes = n_classes

    def forward(self, x):
        return self.model(x)
    
    def card(self, dataset_description: str, optimizer: str, learning_rate: float, epochs: int, batch_size: 16, f1_score: float) -> str:
        
        model_description = f"This {self.model_name} is an image classification model that distinguishes different types of garbage for special [UWG](https://nowaste.com.ua) stations. It is built on the top of the [BEiT](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) model from the transformers library, which is used as encoder. The model is fine-tuned with a custom linear classifier. Fully connected layer size={self.fc_layer_size} and dropout rate={self.dropout_rate} are used."
        preprocessing_details = "Used augmentation technics: random affine transformations + horizontal flips. After that used BEiT encoder to produce embeddings for the dataset images."
        training_details = f"Trained using {optimizer} optimizer and cross entropy as loss. Learning rate: {learning_rate:.5f}, batch size: {batch_size}, number of epochs: {epochs}."
        evaluation_details = f"The performance of the model is evaluated using the macro F1 score. This score considers both precision and recall and computes their harmonic mean, which provides a balanced measure of the model's performance, taking into account both false positives and false negatives. Model managed to get F1 score ~ {f1_score:.3f} using 20% as validation set from provided dataset."
        usage = "To use this model you can use [train.py](https://github.com/rosklyar/projector-ml-ops/blob/main/garbage-classifier/garbage-classifier/train.py) to prepare the artifact and then [inference.py](https://github.com/rosklyar/projector-ml-ops/blob/main/garbage-classifier/garbage-classifier/inference.py) to make predictions."
        limitations = "The model's performance is dependent on the quality and quantity of the provided dataset. The model may not perform well on new types of garbage that were not present in the training dataset. Additionally, the pre-trained BeitForImageClassification model might not be the best choice for this specific task, and other architectures may yield better performance."

        return create_model_card(
            self.model_name,
            model_description,
            dataset_description,
            preprocessing_details,
            training_details,
            evaluation_details,
            usage,
            limitations
        )
    
class BeitImageClassificationProcessor(ImageProcessor):
        
        def __init__(self, model_name):
            super().__init__()
            self.processor = BeitFeatureExtractor.from_pretrained(model_name)
            
        def process(self, image):
            return self.processor(images=image, return_tensors="pt")['pixel_values']
        
        def __repr__(self):
            return self.__class__.__name__ + '()'


