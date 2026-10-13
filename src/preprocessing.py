import json
import pickle
import re
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# WikiText artifacts
MARKER_RE = re.compile(r"\s*@([-,.])@\s*")
HEADING_RE = re.compile(r"=( =)* .+ =( =)*")

def clean_line(text: str, restore_markers: bool = True) -> str | None:
    text = text.strip()
    if not text or HEADING_RE.fullmatch(text):
        return None
    
    if restore_markers:
        text = MARKER_RE.sub(r"\1", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_dataset(dataset) -> dict[str, list[str]]:
    
    cleaned_lines = {}

    for split_name in dataset.keys():
        cleaned_lines[split_name] = []

        for example in dataset[split_name]:
            cleaned = clean_line(example["text"])
            if cleaned is not None:
                cleaned_lines[split_name].append(cleaned)

        print(f"{split_name}: "
              f"Original lines: {len(dataset[split_name])}"
              f"Cleaned lines: {len(cleaned_lines[split_name])}")

    return cleaned_lines


def train_tokenizer(training_lines: list[str], vocab_size: int = 200_000) -> Tokenizer:
    
    tokenizer = Tokenizer(models.WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,min_frequency=1)
    tokenizer.train_from_iterator(training_lines, trainer=trainer)

    return tokenizer


def build_vocabulary(tokenizer: Tokenizer):
    vocab = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return vocab, word_to_id, id_to_word


def encode_dataset(cleaned_lines: dict[str, list[str]], tokenizer: Tokenizer, word_to_id: dict[str, int]) -> dict[str, list[list[int]]]:
    bos_id = word_to_id[BOS_TOKEN]
    eos_id = word_to_id[EOS_TOKEN]
    unk_id = word_to_id[UNK_TOKEN]

    encoded_lines = {}

    for split_name, lines in cleaned_lines.items():
        encoded_lines[split_name] = []

        for line in lines:
            token_ids = tokenizer.encode(line).ids
            token_ids = [bos_id] + token_ids + [eos_id]
            encoded_lines[split_name].append(token_ids)

        unknown_count = sum(token_id == unk_id for sequence in encoded_lines[split_name] for token_id in sequence)

        print(f"{split_name}: {len(encoded_lines[split_name])} sequences,"
              f"{unknown_count} unknown tokens")

    return encoded_lines


def save_outputs(output_dir: Path, tokenizer: Tokenizer, vocab: list[str], word_to_id: dict[str, int], encoded_lines: dict[str, list[list[int]]]) -> None:
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir/"vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab": vocab,
                "word_to_id": word_to_id,
                "special_tokens": SPECIAL_TOKENS,
            },
            f,
            ensure_ascii=False,
            indent=2)

    with open(output_dir/"encoded_lines.pkl", "wb") as f:
        pickle.dump(encoded_lines, f)

    tokenizer.save(str(output_dir/"tokenizer.json"))

    print(f"Saved preprocessing files to: {output_dir}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    processed_dir = project_dir / "data" / "processed"

    print("~~~~Loading WikiText-2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("\n~~~~Cleaning dataset...")
    cleaned_lines = clean_dataset(dataset)

    print("\n~~~~Training tokenizer...")
    tokenizer = train_tokenizer(cleaned_lines["train"])
    vocab, word_to_id, _id_to_word = build_vocabulary(tokenizer)

    print(f"Vocabulary size: {len(vocab)}")
    print("Special token IDs:", {token: word_to_id[token] for token in SPECIAL_TOKENS})

    print("\n~~~~Encoding dataset...")
    encoded_lines = encode_dataset(cleaned_lines, tokenizer, word_to_id)

    print("\n~~~~Saving files...")
    save_outputs(processed_dir, tokenizer, vocab, word_to_id, encoded_lines)


if __name__ == "__main__":
    main()
