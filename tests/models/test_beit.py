import torch
import pytest

from garbage_classifier.models.beit.model import BeitImageClassificationProcessor, BeitImageClassificationModel

@pytest.mark.parametrize("input_size", [(3, 256, 256), (3, 512, 512), (3, 1024, 1024), (3, 640, 480)])
def test_beit_image_processor_output_shape(input_size):
    image_processor = BeitImageClassificationProcessor(
        'microsoft/beit-base-patch16-224-pt22k-ft22k')
    assert image_processor is not None

    processed = image_processor.process(torch.rand(*input_size))
    assert processed.shape == torch.Size([1, 3, 224, 224])

def test_beit_model_compatible_with_processor():
    classes = 50
    model = BeitImageClassificationModel('microsoft/beit-base-patch16-224-pt22k-ft22k', 0.1, 512, classes)
    result = model(torch.rand(1, 3, 224, 224))
    assert result.logits.shape == torch.Size([1, classes])