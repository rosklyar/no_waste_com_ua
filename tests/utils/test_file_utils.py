from pathlib import Path
import tarfile
from unittest.mock import Mock

from garbage_classifier.utils.file_utils import create_train_test_split, create_tar_gz_folder, download_files_from_s3_bucket_folder, extract_tar_gz


def test_create_train_test_split(tmp_path):
    src_folder = tmp_path / 'src_folder'
    train_folder = tmp_path / 'train_folder'
    test_folder = tmp_path / 'test_folder'
    src_folder.mkdir()
    class_folder = src_folder / 'class1'
    class_folder.mkdir()
    (class_folder / 'image1.jpg').touch()
    (class_folder / 'image2.jpg').touch()

    create_train_test_split(src_folder, train_folder, test_folder)

    assert (train_folder / 'class1').exists()
    assert (test_folder / 'class1').exists()

def test_create_tar_gz_folder(tmp_path):
    src_folder = tmp_path / 'src_folder'
    output_filename = tmp_path / 'test.tar.gz'
    src_folder.mkdir()
    (src_folder / 'file1.txt').touch()

    create_tar_gz_folder(src_folder, output_filename)

    assert output_filename.exists()
    assert tarfile.is_tarfile(output_filename)

def test_extract_tar_gz(tmp_path):
    archive_path = tmp_path / 'test.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        (tmp_path / 'file1.txt').touch()
        tar.add(str(tmp_path / 'file1.txt'), arcname='file1.txt')

    extracted_path = extract_tar_gz(archive_path)

    assert extracted_path.exists()
    assert (extracted_path / 'file1.txt').exists()

def test_download_files_from_s3_bucket_folder():
    # Set up the mock S3 client
    s3 = Mock()

    # Configure the paginator to return a mock page with a mock object
    mock_paginator = Mock()
    mock_paginator.paginate.return_value = [
        {'Contents': [{'Key': 'my-folder/my-file.txt'}]}
    ]
    s3.get_paginator.return_value = mock_paginator

    # Set up the parameters
    bucket_name = 'my-bucket'
    folder_path = Path('my-folder')
    local_destination = Path().absolute() / 'test-folder'

    download_files_from_s3_bucket_folder(s3, bucket_name, folder_path, local_destination)

    # Check that the S3 client methods were called with the expected arguments
    s3.get_paginator.assert_called_once_with('list_objects_v2')
    mock_paginator.paginate.assert_called_once_with(Bucket=bucket_name, Prefix=str(folder_path))
    s3.download_file.assert_called_once_with(bucket_name, 'my-folder/my-file.txt', str(local_destination / 'my-file.txt'))