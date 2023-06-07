from pathlib import Path
from json import load
import os
import torch
import wandb

from garbage_classifier.image_data.image_data import ImageData
from garbage_classifier.models.model_builder import create_model, create_image_processor
from garbage_classifier.training.training_utils import create_optimizer, train_epoch, score_model, NoOpLogger
from garbage_classifier.models.model_card import save_model_card


def train(config_path: Path, train_folder_path: Path, test_folder_path: Path, output_dir: Path, logger=NoOpLogger()):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load config
    json_config = load(config_path.open())

    # create image processor
    image_processor = create_image_processor(json_config['image_processor'])

    # data
    garbage_data = ImageData(train_folder_path, test_folder_path, json_config['batch_size'], image_processor)
    train_dataloader = garbage_data.get_train_loader()
    val_dataloader = garbage_data.get_test_loader()

    # model
    classes = len(list(train_folder_path.glob('*')))
    model = create_model(json_config['model'], json_config['dropout'], json_config['fc_layer_size'], classes)
    model.to(device)

    # optimizer
    optimizer = create_optimizer(model, json_config['optimizer'], json_config['learning_rate'])

    # training loop
    f1_score = 0
    avg_loss = 0
    for epoch in range(json_config['epochs']):
        print(f'Epoch {epoch} started...')
        avg_loss = train_epoch(model, train_dataloader, optimizer, device)
        logger.log({"loss": avg_loss})
        f1_score = score_model(model, val_dataloader, device)
        logger.log({'val_f1': f1_score})

    os.makedirs(output_dir, exist_ok=True)
    # save model and model card
    torch.save(model, output_dir / 'model.pt')
    save_model_card(output_dir / 'card.md', model.card(json_config['dataset_description'], json_config['optimizer'], json_config['learning_rate'], json_config['epochs'], json_config['batch_size'], f1_score))
    print(f"Model saved as {str(output_dir / 'model.pt')}")
    print(f"Model Card saved as {str(output_dir / 'card.md')}")

def train_and_log_wandb(config_path: Path, train_folder_path: Path, test_folder_path: Path, output_dir: Path, project_name: str):
    wandb.init(project=project_name)
    train(config_path, train_folder_path, test_folder_path, output_dir, wandb)

def train_and_save(config_path: Path, train_folder_path: Path, test_folder_path: Path, output_dir: Path):
    train(config_path, train_folder_path, test_folder_path, output_dir)