import argparse
from sentence_transformers import SentenceTransformer

# Registry of only the HuggingFace models
MODEL_REGISTRY = {
    "sbert-minilm": "all-MiniLM-L6-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    "qwen3-embed-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embed-4b": "Qwen/Qwen3-Embedding-4B"
}

def main():
    parser = argparse.ArgumentParser(description="Cache HuggingFace models on an air-gapped login node.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Select a specific model to download, or 'all' to cache everything."
    )
    args = parser.parse_args()

    models_to_download = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    print("==================================================")
    print("Initiating Hugging Face Model Caching")
    print("==================================================")

    for model_key in models_to_download:
        hf_model_path = MODEL_REGISTRY[model_key]
        trust_remote = True if model_key in ["nomic", "gte-large"] else False
        
        print(f"\n-> Fetching: {model_key} ({hf_model_path})")
        try:
            # Instantiating the model triggers the download to ~/.cache/huggingface
            _ = SentenceTransformer(hf_model_path, trust_remote_code=trust_remote)
            print(f"[SUCCESS] {model_key} is cached and ready.")
        except Exception as e:
            print(f"[ERROR] Failed to download {model_key}. Reason: {e}")

    print("\n==================================================")
    print("All requested downloads complete!")
    print("You can now run your main embedding script on the compute nodes.")

if __name__ == "__main__":
    main()