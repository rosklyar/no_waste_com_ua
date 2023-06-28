import wandb
from pathlib import Path

def upload_to_registry(entity_name: str, project_name: str, model_name: str, model_path: Path, classes_json: Path):
    with wandb.init(project=project_name, entity=entity_name) as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(classes_json)
        art.add_file(model_path / "model.pt")
        art.add_file(model_path / "card.md")
        art.add_file(model_path / "idx_to_cls.json")
        art.add_dir(model_path / "drift_detector", name="drift_detector")
        wandb.log_artifact(art)

def download_from_registry(entity_name: str, project_name: str, artifact_name: str, artifact_version: str, model_path: Path):
    with wandb.init(project=project_name, entity=entity_name) as run:
        artifact = run.use_artifact(f'rosklyar/{project_name}/{artifact_name}:{artifact_version}')
        artifact_dir = artifact.download(root=model_path)
        run.finish()
        return artifact_dir