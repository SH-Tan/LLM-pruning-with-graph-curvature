from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUTPUT = "downstream_test/dataset/gsm8k/test.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GSM8K test split as local downstream parquet.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataframe = dataset.to_pandas()
    dataframe = dataframe.rename(columns={"question": "prompt"})
    dataframe["data_source"] = "openai/gsm8k"
    dataframe.to_parquet(output_path, index=False)
    print(f"saved {len(dataframe)} GSM8K test examples to {output_path}")


if __name__ == "__main__":
    main()
