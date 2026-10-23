from pathlib import Path
import json
import pickle
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def make_lstm_dataset(sequences, block_size, pad_id):
    input_blocks = []
    target_blocks = []

    for sequence in sequences:
        # input and target are shifted by one token.
        for start in range(0, len(sequence) - 1, block_size):
            
            end = min(start + block_size, len(sequence) - 1) # never letting the input include the very last token of the sequence
            
            # input and target are shifted  by one position
            x = sequence[start:end]
            y = sequence[start + 1: end + 1]
            
            # adding padding, to make the blocks same length
            x = x + [pad_id]*(block_size - len(x))
            y = y + [pad_id]*(block_size - len(y))

            input_blocks.append(x)
            target_blocks.append(y)
            
    # converting to tensor
    inputs = torch.tensor(input_blocks, dtype=torch.long)
    targets = torch.tensor(target_blocks, dtype=torch.long)

    return TensorDataset(inputs, targets)


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.2, pad_id=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id) # first embeddings
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0) # then learning patterns across the sequence
        self.dropout = nn.Dropout(dropout) # to reduce overfitting during training
        
        # score for every possible next token
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids) # [batch_size, sequence_length, embedding_dim]
        lstm_output, _ = self.lstm(embedded) # [batch_size, sequence_length, hidden_dim]
        lstm_output = self.dropout(lstm_output) 
        logits = self.output(lstm_output)
        
        return logits


def run_epoch(model, data_loader, criterion, optimizer=None): # it is like if the we provide optimizer as an argument then the function will do training otherwise evaluation
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        valid_tokens = (targets != pad_id).sum()

        if valid_tokens.item() == 0:
            continue

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(inputs) # forward pass
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            if training:
                batch_average_loss = loss/valid_tokens
                batch_average_loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step()

        total_loss += loss.item()
        total_tokens += valid_tokens.item()

    if total_tokens == 0:
        raise ValueError("No valid target tokens found.")

    average_loss = total_loss/total_tokens
    perplexity = math.exp(min(average_loss, 20))

    return {
        "loss": average_loss,
        "perplexity": perplexity
    }


def evaluate_lstm_model(model, sequences, top_k=5):

    total_loss = 0.0
    total_predictions = 0
    unk_count = 0
    known_count = 0
    correct_top1 = 0
    correct_topk = 0

    special_ids = {bos_id, eos_id, pad_id, unk_id}

    model.eval()

    with torch.no_grad():
        for sequence in sequences:

            if len(sequence) < 2:
                continue

            input_ids = sequence[:-1]
            target_ids = sequence[1:]

            inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = model(inputs)[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            for i, target in enumerate(target_ids):

                if target == unk_id:
                    unk_count += 1
                else:
                    known_count += 1

                total_loss -= log_probs[i, target].item()
                total_predictions += 1

                if target != unk_id:
                    scores = logits[i].clone()

                    for special_id in special_ids:
                        scores[special_id] = float("-inf")

                    k = min(top_k, scores.numel() - len(special_ids))

                    predictions = torch.topk(scores, k=k).indices.tolist()

                    if predictions and target == predictions[0]:
                        correct_top1 += 1

                    if target in predictions:
                        correct_topk += 1

    if total_predictions == 0:
        raise ValueError("No valid predictions found.")

    perplexity = math.exp(total_loss / total_predictions)
    top1_accuracy = (correct_top1 / known_count if known_count else 0.0)
    topk_accuracy = (correct_topk / known_count if known_count else 0.0)

    unk_rate = unk_count/total_predictions

    return {
        "perplexity": perplexity,
        "top1_accuracy": top1_accuracy,
        f"top{top_k}_accuracy": topk_accuracy,
        "unk_rate": unk_rate,
        "number_of_predictions": total_predictions
    }


def get_lstm_suggestions(prefix_ids, model, top_k=5):

    prefix_ids = prefix_ids[-BLOCK_SIZE:] # so same length as training

    if len(prefix_ids) == 0: # empty input in case
        raise ValueError("Prefix cannot be empty.")

    input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    model.eval()

    with torch.no_grad():

        logits = model(input_tensor)
        # taking predictions from the last position
        next_token_logits = logits[0, -1].clone() # scores for the next token

        special_ids = {bos_id, eos_id, pad_id, unk_id} # to exclude special tokens
        for token_id in special_ids:
            next_token_logits[token_id] = float("-inf") # so they can never appear in top

        # top k predictions
        k = min(top_k, next_token_logits.numel() - len(special_ids)) # how many predictions to return
        candidate_ids = torch.topk(next_token_logits, k=k).indices.tolist()

    return candidate_ids


def show_test_examples(sequences, model, number_of_examples=8, seed=42):

    rng = random.Random(seed)
    candidates = [] # to store the locations of valid next word prediction examples

    for sequence_index, sequence in enumerate(sequences):
        for target_position in range(3, len(sequence)): # possible target positions, and at least 3 preceding tokens as context

            target_id = sequence[target_position] # token id at the selected position
            if target_id in {bos_id, eos_id, pad_id, unk_id}:
                continue # to skip
            candidates.append((sequence_index, target_position))

    selected = rng.sample(candidates, k=min(number_of_examples, len(candidates))) # selecting without replacement

    for example_number, (sequence_index, target_position) in enumerate(selected, start=1): # to process each randomly selected example
        sequence = sequences[sequence_index] # getting the selected sequence
        prefix_ids = sequence[:target_position]
        actual_next_id = sequence[target_position]

        # the last 8 prefix tokens
        visible_prefix_ids = prefix_ids[-8:]

        prefix_text = " ".join(id_to_word[token_id] for token_id in visible_prefix_ids)
        actual_next_word = id_to_word[actual_next_id]

        # predictions from the model
        suggestion_ids = get_lstm_suggestions(prefix_ids, model, top_k=5)

        suggestions = [id_to_word[token_id] for token_id in suggestion_ids]

        # Check if the correct token is among top 5
        correct_top5 = actual_next_id in suggestion_ids

        print(f"\nExample {example_number}")
        print("Prefix: ", prefix_text)
        print("Correct word:", actual_next_word)
        print("Suggestions: ", ", ".join(suggestions))
        print("Correct word in top 5:", correct_top5)



def main():

    global device, pad_id, vocab_size, bos_id, eos_id, unk_id, BLOCK_SIZE, id_to_word

  ### load preprocessing files

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR/"data"
    PROCESSED_DIR = DATA_DIR/"processed"

    # Model directory
    MODEL_DIR = PROJECT_DIR/"models"/"lstm"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(PROCESSED_DIR/"encoded_lines.pkl", "rb") as f:
        encoded_lines = pickle.load(f)
    with open(PROCESSED_DIR/"vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data["word_to_id"]
    id_to_word = {int(idx): word for word, idx in word_to_id.items()}
    vocab_size = len(word_to_id)

    bos_id = word_to_id["<bos>"]
    eos_id = word_to_id["<eos>"]
    unk_id = word_to_id["<unk>"]
    pad_id = word_to_id["<pad>"]

    print("Vocabulary size:", vocab_size)

    for split, sequences in encoded_lines.items():
        print(split, "sequences:", len(sequences))


    ## -Training
    # reproducibility
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model settings
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = 0.2

    # memory-conscious sequence settings
    BLOCK_SIZE = 32
    BATCH_SIZE = 4

    # training settings
    LEARNING_RATE = 0.0005
    EPOCHS = 10

    ### Making the dataset ready

    # for each split
    train_dataset = make_lstm_dataset(encoded_lines["train"], BLOCK_SIZE, pad_id)
    validation_dataset = make_lstm_dataset(encoded_lines["validation"], BLOCK_SIZE, pad_id)

    # we only shuffle training blocks
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ### - Defining the lstm

    ### Initialize the model and loss function
    model = LSTMLanguageModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, pad_id=pad_id).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum") # ignoring padded targets when calculating loss
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)


    ### training on data and monitoring
    best_validation_perplexity = float("inf") # pos infinity
    best_model_path = MODEL_DIR / "lstm_best.pt" # where we should save the best model
    history = []

    for epoch in range(1, EPOCHS + 1):

        # training
        train_results = run_epoch(model, train_loader, criterion, optimizer=optimizer)

        # evaluation
        validation_results = run_epoch(model, validation_loader, criterion)

        # then storing everything
        epoch_results = {
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "train_perplexity": train_results["perplexity"],
            "validation_loss": validation_results["loss"],
            "validation_perplexity": validation_results["perplexity"]
        }

        history.append(epoch_results)

        # printing everything
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_results['loss']:.4f}")
        print(f"Validation loss: {validation_results['loss']:.4f}")
        print(f"Train perplexity: {train_results['perplexity']:.4f}")
        print(f"Validation perplexity: " f"{validation_results['perplexity']:.4f}")

        # we only save the model if it is the best so far
        if validation_results["perplexity"] < best_validation_perplexity:
            best_validation_perplexity = validation_results["perplexity"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_perplexity": best_validation_perplexity,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "vocab_size": vocab_size,
                    "pad_id": pad_id
                }, best_model_path
            )
            print("-- New best checkpoint saved!")

        # then saving the history
        history_path = MODEL_DIR/"lstm_training_history.json"

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    print("\nTraining completed:)")
    print("Best validation perplexity:", best_validation_perplexity)
    print("Best checkpoint:", best_model_path)

    ## Plotting
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["validation_loss"] for item in history]
    train_perplexities = [item["train_perplexity"] for item in history]
    val_perplexities = [item["validation_perplexity"] for item in history]

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss per token")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot perplexity
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_perplexities, marker="o", label="Training perplexity")
    plt.plot(epochs, val_perplexities, marker="o", label="Validation perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("LSTM Training and Validation Perplexity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ## Evaluation
    ### top 1 and top 5 accuracy

    # I try to do same evaluation as I did for n-gram to compare them better
    validation_results = evaluate_lstm_model(model, encoded_lines["validation"], top_k=5)
    test_results = evaluate_lstm_model(model, encoded_lines["test"], top_k=5)
    print("Final LSTM model results:")
    print("=" * 55)
    print("Learning rate:", LEARNING_RATE)
    print("Evaluated epoch:", EPOCHS)
    print("Vocabulary size:", vocab_size)
    print("\n{:<25} {:>12} {:>12}".format("Metric", "Validation", "Test"))
    print("-" * 55)

    for metric in validation_results:
        val = validation_results[metric]
        test = test_results[metric]
        if "accuracy" in metric or metric == "unk_rate":
            val_str = f"{val:.2%}"
            test_str = f"{test:.2%}"
        elif metric == "number_of_predictions":
            val_str = str(val)
            test_str = str(test)
        else:
            val_str = f"{val:.4f}"
            test_str = f"{test:.4f}"

        print(f"{metric:<25} {val_str:>12} {test_str:>12}")


    print("========== Trained LSTM ==========")
    show_test_examples(encoded_lines["test"], model, number_of_examples=8, seed=42)

if __name__ == "__main__":

    main()

