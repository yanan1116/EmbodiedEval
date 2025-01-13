import os
import zipfile
from huggingface_hub import snapshot_download
from run_eval import get_platform

os.makedirs("data", exist_ok=True)

platform = get_platform()
ignore_envs = ["*env-Windows.zip", "*env-MacOS.app.zip", "*env-Linux.zip"]
ignore_envs.remove(f"*env-{platform}.zip")

snapshot_download(
    repo_id="EmbodiedEval/EmbodiedEval",
    repo_type="dataset",
    local_dir="data",
    ignore_patterns=ignore_envs+[".gitattributes", "README.md"]
)

file_path = f"data/envs/env-{platform}.zip"
extract_to = file_path.rsplit(".", maxsplit=1)[0]
with zipfile.ZipFile(file_path, "r") as zip_ref:
    zip_ref.extractall(extract_to)
if platform != "Windows":
    mode = 0o777
    for root, dirs, files in os.walk(extract_to):
        os.chmod(root, mode)
        for file in files:
            os.chmod(os.path.join(root, file), mode)
env_path = os.path.abspath(extract_to).replace("\\", "/")
