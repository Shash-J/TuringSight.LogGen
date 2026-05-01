import os
from huggingface_hub import HfApi, create_repo

def export_to_hf(repo_id: str, token: str):
    print(f"\n[Export] Connecting to HuggingFace Dataset: {repo_id}")
    
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id, repo_type="dataset", private=True, token=token)
        print(f"[Export] Created new private dataset: {repo_id}")
    except Exception as e:
        print(f"[Export] Repository already exists, uploading to existing dataset.")
        
    print("[Export] Uploading all logs and frames from 'data/' directory... This might take a minute.")
    api.upload_folder(
        folder_path="data/",
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data/",
        token=token
    )
    print("\n✅ [Success] Export complete! Your data is safe in HuggingFace.")
    print("You can now safely terminate your cloud GPU instance.")

if __name__ == "__main__":
    print("=== TuringSight Data Exporter ===")
    
    # Try to load from environment variables first
    repo_id = os.environ.get("HF_DATASET_REPO")
    if not repo_id:
        repo_id = input("Enter your HuggingFace Dataset Repo ID (e.g. username/turingsight-logs): ")
        
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your HuggingFace Write Token (from huggingface.co/settings/tokens): ")
    
    if repo_id and token:
        export_to_hf(repo_id, token)
    else:
        print("Error: Repo ID and Token are required.")
