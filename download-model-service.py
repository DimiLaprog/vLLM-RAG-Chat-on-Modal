from modal import App, Volume, Image, Secret
from pathlib import Path

MODELS_DIR = "/llm"
DEFAULT_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
DEFAULT_REVISION = None # e.g., "a7c09948d9a632c2c840722f519672cd94af885d"
MINUTES = 60
HOURS = 60 * MINUTES

volume = Volume.from_name("llm", create_if_missing=True)
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(["huggingface_hub", "hf-transfer"])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # enable the faster download method
)

app = App("download-model-service",
    image=image,
    secrets=[Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]
)

@app.function(
        volumes={MODELS_DIR: volume}, 
        timeout=4 * HOURS ) # how long the function will run before being terminated
def download_model(model_name: str = DEFAULT_NAME, 
                   model_revision: str = DEFAULT_REVISION, 
                   force_download: bool = False):
    from huggingface_hub import snapshot_download

    volume.reload() # Reload the volume to ensure it is available

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],
        revision=model_revision,
        force_download=force_download,
    )
    
    completion_file = Path(MODELS_DIR) / model_name / "download_complete"
    completion_file.touch()  # Create an empty file to signal completion

    volume.commit() # Commit changes to volume to ensure visibility to other functions
    

@app.local_entrypoint()
def main(model_name: str = DEFAULT_NAME, model_revision: str = DEFAULT_REVISION, force_download: bool = False):
    download_model.remote(model_name, model_revision, force_download)