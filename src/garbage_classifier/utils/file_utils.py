import tarfile
import numpy as np
from unittest.mock import Mock
from pathlib import Path
import shutil

def create_train_test_split(src_folder: Path, train_folder: Path, test_folder: Path, test_size: float = 0.2):
    class_folders = [f for f in src_folder.iterdir() if f.is_dir()]

    for class_folder in class_folders:
        src_class_folder = src_folder / class_folder.name
        train_class_folder = train_folder / class_folder.name
        test_class_folder = test_folder / class_folder.name

        train_class_folder.mkdir(parents=True, exist_ok=True)
        test_class_folder.mkdir(parents=True, exist_ok=True)

        images = [f for f in src_class_folder.iterdir() if f.is_file()]
        np.random.shuffle(images)
        split_index = int(len(images) * (1 - test_size))

        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            shutil.copy(src_class_folder / img.name, train_class_folder / img.name)

        for img in test_images:
            shutil.copy(src_class_folder / img.name, test_class_folder / img.name)

def create_tar_gz_folder(src_folder: Path, output_filename: Path):
    with tarfile.open(output_filename, "w:gz") as tar:
        for entry in src_folder.iterdir():
            if entry.is_dir():
                arcname = entry.name
                tar.add(str(entry), arcname=arcname)
        print(f'Created!!! {output_filename}')
    try:
        tar = tarfile.open(output_filename, "r:gz")
        print(f'Tar length: {len(tar.getnames())}')
        if not tarfile.is_tarfile(str(output_filename)):
            raise Exception(f'Error while creating {output_filename}')
    except Exception as e:
        print(f'Error while creating {output_filename}: {e}')

def download_files_from_s3_bucket_folder(s3, bucket_name: str, folder_path: Path, local_destination: Path):
    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': str(folder_path)}

    for page in paginator.paginate(**operation_parameters):
        for item in page.get('Contents', []):
            if not item['Key'].endswith('/'):
                file_key = Path(item['Key'])
                local_file_path = local_destination / file_key.relative_to(folder_path)

                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                print(f'Downloading {file_key} to {local_file_path}')
                s3.download_file(bucket_name, str(file_key), str(local_file_path))
                print(f'Downloaded {file_key} to {local_file_path}')

def extract_tar_gz(archive_path: Path):
    if not archive_path.exists():
        print(f"The file '{archive_path}' does not exist.")
        return

    if not tarfile.is_tarfile(str(archive_path)):
        print(f"The file '{archive_path}' is not a valid tar archive.")
        return

    # Create a unique directory for the extracted files
    extracted_folder = archive_path.stem
    extracted_path = archive_path.parent / extracted_folder

    if not extracted_path.exists():
        extracted_path.mkdir(parents=True)

    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=str(extracted_path))
            print(f"Successfully extracted '{archive_path}' to '{extracted_path}'.")
            return extracted_path
    except tarfile.ReadError as e:
        print(f"Error reading the archive '{archive_path}': {e}")
    except tarfile.ExtractError as e:
        print(f"Error extracting the archive '{archive_path}': {e}")
    except tarfile.TarError as e:
        print(f"Error handling the archive '{archive_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while extracting '{archive_path}': {e}")

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
