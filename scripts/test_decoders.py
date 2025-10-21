import sys
import os
import torch
from pathlib import Path
from llm2ner.decoders import eval_dataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_decoders.py <filename>")
        sys.exit(1)

    scores_path = Path(sys.argv[1])
    if not scores_path.exists():
        print(f"File {scores_path} does not exist.")
        sys.exit(1)
    obj = torch.load(scores_path, weights_only=False, map_location="cpu")
    
    threshold = 0.999
    min_score = 0.999

    print(f"Decoding with threshold: {threshold}, min_score: {min_score}")
    df = eval_dataset(obj, min_score=min_score, gap_penalty=0.2, score_threshold=threshold, limit_samples=1000)

    print(df)