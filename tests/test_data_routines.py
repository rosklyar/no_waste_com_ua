import pytest
import shutil
from garbage_classifier.data_routines import split_data_folder

@pytest.fixture
def create_test_folders(tmp_path):
    # Setup
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    train_output = output_folder / 'train'
    test_output = output_folder / 'test'
    input_folder.mkdir()

    categories = ['category_1', 'category_2']
    for category in categories:
        (input_folder / category).mkdir()
        for i in range(1, 10):
            (input_folder / category / f'file{i}.txt').write_text(f'{category}_{i}')

    yield input_folder, output_folder, train_output, test_output
    # Teardown
    shutil.rmtree(tmp_path)

def test_split_data_folder(create_test_folders):
    input_folder, output_folder, train_output, test_output = create_test_folders
    split_data_folder(input_folder, output_folder, 0.2)
    # Check that train_output and test_output directories exist
    assert train_output.exists()
    assert test_output.exists()