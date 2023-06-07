import typer
from garbage_classifier.data_routines import split_data_folder, load_and_split_train_data_from_s3
from garbage_classifier.training_routines import train_and_log_wandb, train_and_save
from garbage_classifier.registry_routines import upload_to_registry, download_from_registry

app = typer.Typer()
app.command()(split_data_folder)
app.command()(load_and_split_train_data_from_s3)
app.command()(train_and_log_wandb)
app.command()(train_and_save)
app.command()(upload_to_registry)
app.command()(download_from_registry)

if __name__ == "__main__":
    app()