import shutil
import pytest
import numpy as np
from PIL import Image

from garbage_classifier.image_data.image_data import ImageData
from garbage_classifier.models.model_builder import create_image_processor

@pytest.fixture
def create_train_test_folders(tmp_path):
    train_output = tmp_path / 'train'
    test_output = tmp_path / 'test'
    train_output.mkdir()
    test_output.mkdir()

    categories = ['paper', 'glass', 'shoes', 'plastic']
    for category in categories:
        (train_output / category).mkdir()
        for i in range(1, 9):
            image_data = np.random.rand(512, 512, 3) * 255
            img = Image.fromarray(image_data.astype('uint8')).convert('RGB')
            img.save(train_output / category / f'file{i}.jpg')
    
    for category in categories:
        (test_output / category).mkdir() 
        for i in range(1, 3):
            image_data = np.random.rand(512, 512, 3) * 255
            img = Image.fromarray(image_data.astype('uint8')).convert('RGB')
            img.save(test_output / category / f'file{i}.jpg')

    yield train_output, test_output
    # Teardown
    shutil.rmtree(tmp_path)

def test_image_data_creation(create_train_test_folders):
    train_output, test_output = create_train_test_folders
    image_processor = create_image_processor("beit")
    garbage_data = ImageData(train_output, test_output, 8, image_processor)
    assert garbage_data.get_train_loader() is not None
    assert garbage_data.get_test_loader() is not None
    assert garbage_data.idx_to_cls == {0: 'glass', 1: 'paper', 2: 'plastic', 3: 'shoes'}