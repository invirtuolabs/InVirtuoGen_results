#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import datasets
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
def get_sorted_vocab(tokenizer):
    """
    Retrieves the tokenizer vocabulary sorted by token length in descending order.

    Parameters:
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.

    Returns:
        list: Sorted list of vocabulary tokens.
    """
    return sorted(tokenizer.get_vocab().keys(), key=len, reverse=True)


def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenizes sequences of SMILES fragments, truncates or pads them to a fixed maximum length.

    Parameters:
        examples (dict): Dictionary containing SMILES fragment sequences under the 'fragments' key.
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.
        max_length (int): Maximum sequence length after tokenization.

    Returns:
        dict: Dictionary with key 'input_ids' containing tokenized and padded/truncated sequences.
    """
    for seq in examples["fragments"]:
        encoded = [tokenizer.bos_token_id] + tokenizer.encode(seq) + [tokenizer.eos_token_id]

        truncated = encoded[:max_length]
        pad_length = max_length - len(truncated)
        if pad_length > 0:
            truncated += [tokenizer.pad_token_id] * pad_length
        input_ids = truncated
    return {"input_ids": input_ids}


def tokenize_dataset(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    max_length: int = 150,
    batch_size: int = 10000,
    num_proc: int = 4,
    use_datasets: bool = False,
    int8: bool = False,
):
    """
    Loads a dataset, tokenizes SMILES fragments using a specified tokenizer,
    and saves the tokenized dataset.

    Parameters:
        input_path (str): Path to the input dataset.
        output_path (str): Path where the tokenized dataset will be saved.
        tokenizer_path (str): Path to the tokenizer file.
        max_length (int, optional): Maximum tokenized sequence length. Defaults to 150.
        batch_size (int, optional): Number of examples processed per batch. Defaults to 10000.
        num_proc (int, optional): Number of parallel processes to use. Defaults to 4.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.unk_token = "[UNK]"

    print(f"Loading dataset from {input_path}")
    dataset = datasets.load_from_disk(input_path)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=False,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns="fragments",
    )
    if use_datasets:
        dataset.save_to_disk(output_path.replace(".npy", ""), num_proc=num_proc)
    else:
        save_to_memmap(dataset, output_path, max_length,dtype="uint8" if int8 else "uint16")



    print(f"Tokenized dataset saved to {output_path}")


def save_to_memmap(dataset, output_path: str, max_length: int, dtype: str = "uint16"):
    """
    Saves tokenized dataset to memory-mapped numpy arrays.

    Parameters:
    dataset: HuggingFace dataset containing tokenized sequences
    output_dir (str): Directory where memory-mapped files will be saved
    max_length (int): Maximum sequence length
    dtype (str): Data type for the arrays (default: 'int32')

    Returns:
    str: Path to the saved memory-mapped file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_samples = len(dataset)

    # Create memory-mapped array for input_ids
    input_ids_memmap = np.memmap(output_path+".npy", dtype=dtype, mode="w+", shape=(num_samples, max_length))

    # Fill the memory-mapped array
    batch_size = 10000  # Process in batches to manage memory
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = dataset[i:end_idx]
        input_ids_memmap[i:end_idx] = np.array(batch["input_ids"], dtype=dtype)

    # Flush to disk
    del input_ids_memmap

    # Save metadata
    metadata = {"num_samples": num_samples, "max_length": max_length, "dtype": dtype, "shape": [num_samples, max_length]}

    metadata_path = os.path.join("/".join(output_path.split("/")[:-1]), "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    print(f"Shape: {metadata['shape']}, dtype: {dtype}")



def custom_decode_sequence(tokenizer, encoded_ids):
    """
    Decodes a sequence of token IDs into a human-readable SMILES string,
    removing special tokens.

    Parameters:
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.
        encoded_ids (list[int]): List of token IDs to decode.

    Returns:
        str: Decoded and cleaned SMILES string.
    """
    tokens = [tokenizer.decode(id) for id in encoded_ids]
    tokens = [str(token) for token in tokens if token is not None]
    decoded_sequence = "".join(tokens)
    decoded_sequence = decoded_sequence.replace("[BOS]", "").replace("[EOS]", "").replace("[UNK]", "").replace("[PAD]", "")
    return decoded_sequence


def main():
    """
    Parses command-line arguments and initiates dataset tokenization.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output dataset.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer file.")
    parser.add_argument("--max_length", type=int, default=150, help="Max sequence length.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--use_datasets", action="store_true", help="Use datasets library.")


    args = parser.parse_args()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path) if args.tokenizer_path.find("safe") == -1 else Tokenizer.from_file('tokenizer/safe.json')(tokenizer_file=args.tokenizer_path)
    vocab_size = len(tokenizer.get_vocab()) if args.tokenizer_path.find("safe") != -1 else len(tokenizer)
    tokenize_dataset(int8=vocab_size <= 256, **vars(args))




if __name__ == "__main__":
    main()
