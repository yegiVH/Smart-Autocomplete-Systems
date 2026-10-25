from pathlib import Path
import json
import pickle
import math
from collections import Counter, defaultdict
from tokenizers import Tokenizer
import re
import random
from tokenizers import Tokenizer
import time
import sys
import heapq


class NGramLanguageModel:
    def __init__(self, order=3, alpha=0.0001): # so we use two previous tokens to predict the next token
        self.order = order
        self.alpha = alpha # additive smoothing to prevent zero probabilities
        self.counts = defaultdict(Counter) # counts[context][next_token] = freq
        self.context_totals = Counter() # total num of predictions observed for each context
        self.vocab_size = None
        self._top_cache = {} # to avoid repeatedly computing
        self.lambdas = {2: 0.6, 1: 0.3, 0: 0.1} # weights used for interpolation
        
    def fit(self, sequences, vocab_size):
        # first reseting structures to remove old model statistics
        self._prediction_cache = {}
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
        
    def predict_next(self, context, top_k=5, exclude=()):
        context = tuple(context[-(self.order - 1):]) # so only last two tokens for context
        excluded = set(exclude)

        # creating a cache if it does not exist
        if not hasattr(self, "_prediction_cache"):
            self._prediction_cache = {}

        cache_key = (context, top_k, frozenset(excluded), self.alpha, tuple(sorted(self.lambdas.items())))

        # returning cached predictions if this context was already evaluated
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        candidates = set() # all possible next tokens

        # all tokens seen after the trigram/bigram context
        for length in range(len(context), 0, -1):
            candidates.update(self.counts.get(context[-length:], {}).keys())

        # for every other token only the unigram count matters,
        # so the top unigram tokens are enough (+ extra in case some get excluded)
        candidates.update(self._top((), top_k + len(excluded)))

        candidates.difference_update(excluded) # removing excluded tokens

        # ranking candidates by interpolated probability
        ranked = heapq.nlargest(top_k, candidates, key=lambda token_id: self.probability(context, token_id))

        self._prediction_cache[cache_key] = ranked

        return ranked



def evaluate_model(model, sequences, top_k=5, compute_accuracy=True):
    # same special-token handling as the LSTM and transformer evaluations
    special_ids = {bos_id, eos_id, pad_id, unk_id}

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

            # for counting unknown and known targets (known = not a special token)
            if target == unk_id:
                unk_count += 1
            if target not in special_ids:
                known_count += 1

            probability = model.probability(context, target) # p for all targets
            total_log_probability += math.log(probability)
            total_predictions += 1

            # accuracy for known ones
            if compute_accuracy and target not in special_ids:
                predictions = model.predict_next(context, top_k=top_k, exclude=special_ids)

                if predictions and target == predictions[0]:
                    correct_top1 += 1

                if target in predictions:
                    correct_topk += 1

    # for only known targets
    if compute_accuracy:
        top1_accuracy = (correct_top1 / known_count if known_count else 0.0)
        topk_accuracy = (correct_topk / known_count if known_count else 0.0)
    else:
        top1_accuracy = None
        topk_accuracy = None

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


def autocomplete(text, model, top_k=5):

    normalized_text = normalize_for_model(text)

    prefix_ids = [bos_id] + hf_tokenizer.encode(normalized_text).ids

    special_ids = {bos_id, eos_id, pad_id, unk_id}

    predicted_ids = model.predict_next(prefix_ids, top_k=top_k, exclude=special_ids)

    return [id_to_word[token_id] for token_id in predicted_ids]


def show_test_examples(sequences, model, number_of_examples=8, seed=42):

    rng = random.Random(seed)
    candidates = [] # to store the locations of valid next word prediction examples

    special_ids = {bos_id, eos_id, pad_id, unk_id}

    for sequence_index, sequence in enumerate(sequences):
        for target_position in range(3, len(sequence)): # possible target positions, and at least 3 preceding tokens as context

            target_id = sequence[target_position] # token id at the selected position
            if target_id in special_ids:
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
        suggestion_ids = model.predict_next(prefix_ids, top_k=5, exclude=special_ids)

        suggestions = [id_to_word[token_id] for token_id in suggestion_ids]

        # Check if the correct token is among top 5
        correct_top5 = actual_next_id in suggestion_ids

        print(f"\nExample {example_number}")
        print("Prefix: ", prefix_text)
        print("Correct word:", actual_next_word)
        print("Suggestions: ", ", ".join(suggestions))
        print("Correct word in top 5:", correct_top5)




def main():
    global hf_tokenizer, encoded_lines, word_to_id, id_to_word, vocab_size, bos_id, eos_id, unk_id, pad_id


    working_dir = Path.cwd().resolve()
    PROJECT_DIR = working_dir.parent if working_dir.name == "notebooks" else working_dir
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
    print("Example encoded sequence:")
    print(encoded_lines["train"][0][:20])


    # creating the model
    ngram_model = NGramLanguageModel(order=3, alpha=0.0001)

    # learning the n-gram counts from training
    ngram_model.fit(sequences=encoded_lines["train"], vocab_size=vocab_size)

    ### making predictions

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


    # ## Hyperparameter tuning using validatioin perplexity

    alpha_values = [0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01]

    lambda_options = [{2: 0.6, 1: 0.3, 0: 0.1}, {2: 0.5, 1: 0.3, 0: 0.2}, {2: 0.7, 1: 0.2, 0: 0.1}, {2: 0.4, 1: 0.4, 0: 0.2}]

    # best of everything we wanna find
    best_perplexity = float("inf")
    best_alpha = None
    best_lambdas = None

    validation_scores = []

    for alpha in alpha_values:
        for lambdas in lambda_options:
            ngram_model.alpha = alpha
            ngram_model.lambdas = lambdas.copy()

            results = evaluate_model(ngram_model, encoded_lines["validation"], compute_accuracy=False)

            ppl = results["perplexity"]

            validation_scores.append({"alpha": alpha, "lambdas": lambdas.copy(), "perplexity": ppl})

            print(f"alpha = {alpha}, "
                  f"lambdas = {lambdas}, "
                  f"perplexity = {ppl:.4f}")

            if ppl < best_perplexity:
                best_perplexity = ppl
                best_alpha = alpha
                best_lambdas = lambdas.copy()

    # then update
    ngram_model.alpha = best_alpha
    ngram_model.lambdas = best_lambdas.copy()

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\nBest alpha:", best_alpha)
    print("Best lambdas:", best_lambdas)
    print("Best validation perplexity:", best_perplexity)


    # ## Final validation and test evaluation

    validation_results = evaluate_model(ngram_model, encoded_lines["validation"], top_k=5)

    test_results = evaluate_model(ngram_model, encoded_lines["test"], top_k=5)

    print("Final n-gram model results:")
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

 

    ## Autocomplete examples

    test_prefixes = ["", "The game was", "He went to the", "The city of", "She said.", "The player's"]

    for prefix in test_prefixes:
        print("Prefix:", repr(prefix))
        print("Suggestions:", autocomplete(prefix, ngram_model))
        print()

    print("========== Best Tuned N-gram ==========")
    show_test_examples(encoded_lines["test"], ngram_model, number_of_examples=8,seed=42)

    # ## Frequency-based baseline

    # counting word frequencies from training data
    special_ids = {bos_id, eos_id, pad_id, unk_id}

    word_counts = Counter(token for sequence in encoded_lines["train"] for token in sequence[1:] if token not in special_ids)

    # finding the five most frequent words
    top5_words = [token for token, count in word_counts.most_common(5)]

    print("Five most frequent words:")
    for token in top5_words:
        print(id_to_word[token], word_counts[token])


    # evaluating the frequency baseline
    def evaluate_frequency_baseline(sequences, top5_words):
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for sequence in sequences:
            for target in sequence[1:]:

                # accuracy for known words only
                if target in special_ids:
                    continue

                total += 1

                if target == top5_words[0]:
                    correct_top1 += 1

                if target in top5_words:
                    correct_top5 += 1

        return {"top1_accuracy": correct_top1 / total,
                "top5_accuracy": correct_top5 / total
        }


    baseline_validation = evaluate_frequency_baseline(encoded_lines["validation"], top5_words)

    baseline_test = evaluate_frequency_baseline(encoded_lines["test"], top5_words)

    print("\nFrequency baseline results:")
    print(f"Validation Top-1: {baseline_validation['top1_accuracy']:.2%}")
    print(f"Validation Top-5: {baseline_validation['top5_accuracy']:.2%}")
    print(f"Test Top-1: {baseline_test['top1_accuracy']:.2%}")
    print(f"Test Top-5: {baseline_test['top5_accuracy']:.2%}")

    # ## computational-cost measurements

    # measuring training time
    start = time.perf_counter()

    ngram_model.fit(sequences=encoded_lines["train"], vocab_size=vocab_size)

    training_time = time.perf_counter() - start

    print(f"Training time: {training_time:.2f} seconds")

    # counting stored n-gram entries
    num_contexts = len(ngram_model.counts)

    num_ngrams = sum(len(counter) for counter in ngram_model.counts.values())

    print(f"Number of contexts: {num_contexts:,}")
    print(f"Stored n-gram entries: {num_ngrams:,}")


    # collecting 100 test contexts for inference benchmarking
    contexts = []

    for sequence in encoded_lines["test"]:
        for i in range(1, len(sequence)):
            contexts.append(sequence[max(0, i-2):i])

            if len(contexts) == 100:
                break

        if len(contexts) == 100:
            break


    # measuring autocomplete inference time
    start = time.perf_counter()

    for context in contexts:
        ngram_model.predict_next(context, top_k=5, exclude=special_ids)

    inference_time = time.perf_counter() - start
    predictions_per_second = len(contexts) / inference_time

    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Predictions per second: {predictions_per_second:.2f}")

    print("comparison")
    print("=" * 65)

    print(f"{'Metric':<25} {'Frequency Baseline':>20} {'Trigram':>15}")
    print("-" * 65)

    print(f"{'Test Top-1':<25} {baseline_test['top1_accuracy']:>19.2%} {test_results['top1_accuracy']:>14.2%}")
    print(f"{'Test Top-5':<25} {baseline_test['top5_accuracy']:>19.2%} {test_results['top5_accuracy']:>14.2%}")
    print(f"{'Test Perplexity':<25} {'N/A':>20} {test_results['perplexity']:>15.2f}")

    print("\nComputational cost:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Stored n-gram entries: {num_ngrams:,}")
    print(f"Predictions per second: {predictions_per_second:.2f}")


if __name__ == "__main__":
    main()
