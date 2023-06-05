import pytest
from pathlib import Path
import tarfile
import shutil
import boto3

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