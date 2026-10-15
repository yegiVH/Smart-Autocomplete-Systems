# N-gram language model

from pathlib import Path
import json
import pickle
import math
from collections import Counter, defaultdict
from tokenizers import Tokenizer
import re

class NGramLanguageModel:
    def __init__(self, order=3, alpha=0.0001): 
        self.order = order
        self.alpha = alpha # additive smoothing to prevent zero probabilities
        self.counts = defaultdict(Counter) # counts[context][next_token] = freq
        self.context_totals = Counter() # total num of predictions observed for each context
        self.vocab_size = None
        self._top_cache = {} # to avoid repeatedly computing
        self.lambdas = {2: 0.6, 1: 0.3, 0: 0.1} # weights used for interpolation
        
    def fit(self, sequences, vocab_size):
        # first reseting structures to remove old model statistics
        self.counts.clear()
        self.context_totals.clear()
        self._top_cache.clear()
        
        self.vocab_size = vocab_size
        
        for sequence in sequences: # every sentence
            for i in range(1, len(sequence)): # cuz the first token is bos, which is a context only
                target = sequence[i]
                
                # for context
                start_in = max(0, i - (self.order - 1)) # like for i=3 is  3 - (3-1) = 1
                context = tuple(sequence[start_in:i]) # [1:3] so 1 and 2
                
                # and then we count every suffix of the context so that the model falls back to shorter context
                for length in range(len(context)+1):
                    if length == 0:
                        suffix=()
                    else:
                        suffix = context[-length:]
                        
                    self.counts[suffix][target] += 1
                    self.context_totals[suffix] += 1

        return self
     
    def _top(self, suffix, k): # to return the most common next tokens for a context
        key = (suffix, k)
        if key not in self._top_cache: # if it is not already there
            counter = self.counts.get(suffix)

            self._top_cache[key] = ([t for t, _ in counter.most_common(k)] if counter else []) # puth the k most common ones in there

        return self._top_cache[key]
    
    def probability(self, context, target): # P(target | context)

        context = tuple(context[-(self.order - 1):]) # so last two tokens 

        p = 0.0 # probability
        w_sum = 0.0 # sum of weights 

        for length in range(len(context), -1, -1): # startnig from the recent ones
            suffix = context[-length:] if length else ()
            total = self.context_totals.get(suffix, 0)
            if total == 0:
                continue

            w = self.lambdas[length] # we get the interpolation weight
            count = self.counts[suffix].get(target, 0) # we get the frequency

            p += w * ((count + self.alpha) / (total + self.alpha * self.vocab_size)) # so smoothed probability multiplied by weight
            #  like 0.6 × trigram + 0.3 × bigram + 0.1 × unigram
            w_sum += w

        return p / w_sum if w_sum else 1.0 / self.vocab_size # so normalizes when some contexts were skipped
        
    def predict_next(self, context, top_k=5):
        context = tuple(context[-(self.order - 1):])

        results = []
        for length in range(len(context), -1, -1):
            suffix = context[-length:] if length else ()

            for token_id in self._top(suffix, top_k):
                if token_id not in results:
                    results.append(token_id)
                    if len(results) == top_k: # to return top k
                        return results

        return results


def evaluate_model(model, sequences, unk_id, top_k=5):
    # initializing counters
    unk_count = 0
    known_count = 0 # normal vocabulary tokens
    total_log_probability = 0.0 # for perplexity
    total_predictions = 0
    correct_top1 = 0 # how often predictions[0] == target
    correct_topk = 0 # how often target is in predictions

    for sequence in sequences:
        for i in range(1, len(sequence)):

            target = sequence[i]
            start = max(0, i - (model.order - 1))
            context = sequence[start:i]

            # for counting unknown and known targets
            if target == unk_id:
                unk_count += 1
            else:
                known_count += 1

            probability = model.probability(context, target) # p for all targets
            total_log_probability += math.log(probability)
            total_predictions += 1

            # accuracy for known ones
            if target != unk_id:
                predictions = model.predict_next(context, top_k=top_k)

                if predictions and target == predictions[0]:
                    correct_top1 += 1

                if target in predictions:
                    correct_topk += 1

    # for only known targets
    top1_accuracy = (correct_top1 / known_count if known_count else 0.0) 
    topk_accuracy = (correct_topk / known_count if known_count else 0.0)

    # and perplexity
    if total_predictions > 0:
        average_negative_log_probability = (-total_log_probability / total_predictions)
        perplexity = math.exp(average_negative_log_probability)
        unk_rate = unk_count / total_predictions
    else:
        perplexity = float("nan")
        unk_rate = 0.0

    return { # metrics : values
        "perplexity": perplexity,
        "top1_accuracy": top1_accuracy,
        f"top{top_k}_accuracy": topk_accuracy,
        "unk_rate": unk_rate,
        "number_of_predictions": total_predictions
    }


def normalize_for_model(text):

    text = re.sub(r'(?<=\w)([.,;:!?)"])(?=\s|$)', r' \1', text) # to add a space before the punctuation
    text = re.sub(r'(^|\s)([("])(?=\w)', r'\1\2 ', text) # to add a space after opening punctuation
    text = re.sub(r"(\w)'s\b", r"\1 's", text) # to separate possessives
    
    return re.sub(r"\s+", " ", text).strip() # to first replace multiple spaces with one and then removing the spaces at the beginning and end


def autocomplete(text, model, hf_tokenizer, id_to_word, bos_id, eos_id, pad_id, unk_id, top_k=5):

    normalized_text = normalize_for_model(text)

    prefix_ids = [bos_id] + hf_tokenizer.encode(normalized_text).ids

    special_ids = {bos_id, eos_id, pad_id, unk_id}

    predictions = model.predict_next(prefix_ids, top_k=top_k + 20)

    suggestions = []
    for token_id in predictions:
        if token_id not in special_ids:
            suggestions.append(id_to_word[token_id])

        if len(suggestions) == top_k:
            break

    return suggestions


def main():

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = PROJECT_DIR/"data"/"processed"

    hf_tokenizer = Tokenizer.from_file(str(PROCESSED_DIR/"tokenizer.json"))

    with open(PROCESSED_DIR/"encoded_lines.pkl", "rb") as f:
        encoded_lines = pickle.load(f)
    
    with open(PROCESSED_DIR/"vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    
    word_to_id = vocab_data["word_to_id"]
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    vocab_size = len(word_to_id)

    bos_id = word_to_id["<bos>"]
    eos_id = word_to_id["<eos>"]
    unk_id = word_to_id["<unk>"]
    pad_id = word_to_id["<pad>"]

    print("Vocabulary size:", vocab_size)
    print()
    for split_name, sequences in encoded_lines.items():
        print(split_name, "sequences:", len(sequences))
    print()


    # creating the model
    ngram_model = NGramLanguageModel(order=3, alpha=0.0001)

    # learning the n-gram counts from training
    ngram_model.fit(sequences=encoded_lines["train"], vocab_size=vocab_size)

    # %%

    example_sequence = encoded_lines["train"][9]

    # using all the tokens except the final one as a prefix
    prefix_ids = example_sequence[:-1]

    # five most frequent continuations
    predicted_ids = ngram_model.predict_next(prefix_ids, top_k=5)

    print("Input sequences:")
    print(" ".join(id_to_word[token_id] for token_id in prefix_ids[-10:]))

    print("\nPredicted next words:")
    for token_id in predicted_ids:
        print(id_to_word[token_id])

    # %%

    validation_results = evaluate_model(ngram_model, encoded_lines["validation"], unk_id)

    print("Validation results")
    print("~~~~~~~~~~~~~~~~~~~")

    for metric, value in validation_results.items():
        print(f"{metric}: {value}")

    # %%

    alpha_values = [0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01]

    lambda_options = [ {2: 0.6, 1: 0.3, 0: 0.1}, {2: 0.5, 1: 0.3, 0: 0.2}, {2: 0.7, 1: 0.2, 0: 0.1}, {2: 0.4, 1: 0.4, 0: 0.2}]

    # best of everything we wanna find 
    best_perplexity = float("inf")
    best_alpha = None
    best_lambdas = None

    validation_scores = []

    for alpha in alpha_values:
        for lambdas in lambda_options:
            ngram_model.alpha = alpha
            ngram_model.lambdas = lambdas.copy()

            results = evaluate_model(ngram_model, encoded_lines["validation"], unk_id)

            ppl = results["perplexity"]

            validation_scores.append({"alpha": alpha, "lambdas": lambdas.copy(), "perplexity": ppl})

            print(f"alpha ={alpha}, "
                  f"lambdas ={lambdas}, "
                  f"perplexity ={ppl:.4f}")

            if ppl < best_perplexity:
                best_perplexity = ppl
                best_alpha = alpha
                best_lambdas = lambdas.copy()

    # then update
    ngram_model.alpha = best_alpha
    ngram_model.lambdas = best_lambdas.copy()

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\nBest alpha:", best_alpha)
    print("Best lambdas:", best_lambdas)
    print("Best validation perplexity:", best_perplexity)

    # %%

    test_results = evaluate_model(ngram_model, encoded_lines["test"], unk_id)

    print("Final n-gram test results")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")

    for metric, value in test_results.items():
        if "accuracy" in metric:
            print(f"{metric}: {value:.4%}") # percentage format
        elif isinstance(value, float): # if it is float
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # %%

    test_prefixes = ["", "The game was", "He went to the", "The city of", "She said.", "The player's"]

    for prefix in test_prefixes:
        print("Prefix:", repr(prefix))
        print("Suggestions:", autocomplete(prefix, ngram_model, hf_tokenizer, id_to_word, bos_id, eos_id, pad_id, unk_id))
        print()

    # %%

    validation_results = evaluate_model(ngram_model, encoded_lines["validation"], unk_id, top_k=5)

    test_results = evaluate_model(ngram_model, encoded_lines["test"], unk_id, top_k=5)

    print("FINAL n-gram model results:")
    print("=" * 55)

    print("Model order:", ngram_model.order)
    print("Vocabulary size:", vocab_size)
    print("Selected alpha:", ngram_model.alpha)
    print("Selected lambdas:", ngram_model.lambdas)

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

    # %%

    for split, sequences in encoded_lines.items():
        total = sum(len(seq) - 1 for seq in sequences)
        unknowns = sum(token == unk_id for seq in sequences for token in seq[1:])
        print(f"{split}: {unknowns / total:.2%} <unk>")


if __name__ == "__main__":
    main()
