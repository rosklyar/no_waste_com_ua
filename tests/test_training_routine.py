import shutil
import pytest
from pathlib import Path
import numpy as np
from PIL import Image
from garbage_classifier.training_routines import train


@pytest.fixture
def create_train_test_folders(tmp_path):
    output_path = tmp_path / 'output'
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

    yield train_output, test_output, output_path
    # Teardown
    shutil.rmtree(tmp_path)

def test_training(create_train_test_folders):
    train_output, test_output, output_path = create_train_test_folders
    config_path = Path('tests/config/training.json')
    
    train(config_path, train_output, test_output, output_path)

    assert (output_path / "model.pt").exists()
    assert (output_path / "card.md").exists()

    
