import argparse
import sys
from typing import List, Tuple
import polars as pl
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from resources import (
    TRAIN,
    TEST,
    BLIND_A,
    BLIND_B,
    EMBEDDINGS_DIR
)

# REGISTRY OF SUPPORTED MODELS
MODEL_REGISTRY = {
    "tfidf": None,  
    "sbert-minilm": "all-MiniLM-L6-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    "qwen3-embed-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embed-4b": "Qwen/Qwen3-Embedding-4B"
}

def flatten_conversations(dfs: List[Tuple[str, pl.DataFrame]]) -> pl.DataFrame:
    flat_data = []
    for split_name, df in dfs:
        for row in df.iter_rows(named=True):
            session_id = row["session_id"]
            conv = row.get("conversations") or row.get("conversation")
            
            if not conv:
                continue
                
            for idx, turn in enumerate(conv):
                if turn.get("role") in ["user", "assistant"]:
                    flat_data.append({
                        "split": split_name,
                        "session_id": session_id,
                        "turn_number": idx,
                        "role": turn["role"],
                        "content": turn["content"]
                    })
    return pl.DataFrame(flat_data)


# EMBEDDING GENERATORS
def generate_tfidf(flat_df: pl.DataFrame, chunk_size: int = 10000) -> pl.DataFrame:
    print("Step 1: Filtering training set to fit TF-IDF vocabulary...")
    train_texts = flat_df.filter(pl.col("split") == "train")["content"].to_list()
    
    vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
    print("Fitting TfidfVectorizer on TRAIN split only...")
    vectorizer.fit(train_texts)
    
    num_rows = len(flat_df)
    print(f"Step 2: Transforming all splits in chunks of {chunk_size}...")
    
    all_embeddings = []
    for i in tqdm(range(0, num_rows, chunk_size)):
        chunk_texts = flat_df["content"][i:i + chunk_size].to_list()
        sparse_chunk = vectorizer.transform(chunk_texts)
        dense_chunk = sparse_chunk.toarray()
        all_embeddings.extend(dense_chunk.tolist())
        
    return flat_df.with_columns(pl.Series("embedding", all_embeddings))

def generate_sentence_transformer(
    df: pl.DataFrame, 
    model_name: str, 
    prefix: str = "", 
    trust_remote: bool = False
) -> pl.DataFrame:
    
    print(f"Loading SentenceTransformer: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote, local_files_only=True)
    
    batch_size = 2048
    if "Qwen" in model_name:
        batch_size = 64
    print("Using batch size:", batch_size)

    texts = df["content"].to_list()
    if prefix:
        print(f"Applying prefix: '{prefix}' to all texts...")
        texts = [f"{prefix}{t}" for t in texts]

    print("Encoding texts...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    embeddings = [v.tolist() for v in vectors]
    return df.with_columns(pl.Series("embedding", embeddings))

# MAIN EXECUTION
def main():
    parser = argparse.ArgumentParser(description="Extract turn-level embeddings.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=list(MODEL_REGISTRY.keys()),
        help="Select the embedding methodology."
    )
    parser.add_argument(
        "--download_only", 
        action="store_true", 
        help="Download HuggingFace models to cache and exit. Run this on a node with internet access."
    )
    args = parser.parse_args()

    # --- Pre-computation caching logic for air-gapped clusters ---
    if args.download_only:
        print(f"--- Download Mode Triggered for {args.model} ---")
        if args.model == "tfidf":
            print("TF-IDF uses Scikit-Learn's built-in English stop words. No download required.")
        elif args.model == "llm_local":
            print("You must download your GGUF files manually and provide the path via --model_path.")
        else:
            hf_model_path = MODEL_REGISTRY[args.model]
            trust_remote = True if args.model in ["nomic", "gte-large"] else False
            print(f"Fetching and caching: {hf_model_path}")
            # Instantiating the model forces the download to the local HF cache
            _ = SentenceTransformer(hf_model_path, trust_remote_code=trust_remote)
            print("\nDownload complete! The model is now cached.")
            print("You can safely run this script on an air-gapped compute node.")
        sys.exit(0)
    # -------------------------------------------------------------

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw Parquet datasets...")
    df_train = pl.read_parquet(TRAIN, columns=["session_id", "conversations"])
    df_test = pl.read_parquet(TEST, columns=["session_id", "conversations"])
    df_a = pl.read_parquet(BLIND_A, columns=["session_id", "conversations"])
    df_b = pl.read_parquet(BLIND_B, columns=["session_id", "conversations"])

    print("Flattening data structures to turn level...")
    datasets = [("train", df_train), ("test", df_test), ("blind_a", df_a), ("blind_b", df_b)]
    flat_df = flatten_conversations(datasets)

    if args.model == "tfidf":
        processed_df = generate_tfidf(flat_df)

    else:
        hf_model_path = MODEL_REGISTRY[args.model]
        use_prefix = "query: " if args.model == "e5-large" else ""
        trust_remote = True if args.model in ["nomic", "gte-large"] else False

        processed_df = generate_sentence_transformer(
            flat_df, 
            model_name=hf_model_path, 
            prefix=use_prefix, 
            trust_remote=trust_remote
        )

    print("\nSaving separated datasets...")
    for split_name in ["train", "test", "blind_a", "blind_b"]:
        split_df = processed_df.filter(pl.col("split") == split_name).select([
            "session_id", "turn_number", "role", "embedding"
        ])
        
        output_file = EMBEDDINGS_DIR / f"{split_name}_{args.model}_embeddings.parquet"
        split_df.write_parquet(output_file)
        print(f"Saved {split_name.upper()} split to {output_file} (Rows: {len(split_df)})")

if __name__ == "__main__":
    main()