import boto3
from pathlib import Path
from garbage_classifier.utils.file_utils import create_train_test_split, download_files_from_s3_bucket_folder

def load_and_split_train_data_from_s3(s3_access_key: str, s3_secret_key: str, s3_bucket: str, s3_prefix: str, folder: Path):
    s3_client = boto3.client('s3', aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    input_folder = folder / 'input'
    output_folder = folder / 'output'
    download_files_from_s3_bucket_folder(s3_client, s3_bucket, s3_prefix, input_folder)
    return split_data_folder(input_folder, output_folder)

def split_data_folder(input_folder: Path, output_folder: Path, test_ratio: float):
    train_output = output_folder / 'train'
    test_output = output_folder / 'test'
    create_train_test_split(input_folder, train_output, test_output, test_ratio)
    return (train_output, test_output)