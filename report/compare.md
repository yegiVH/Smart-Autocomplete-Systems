# N-gram vs. LSTM

we have a statistical trigram model and a neural LSTM language model. Both models were evaluated using the same vocabulary of 46,165 tokens and the same validation and test datasets. Performance was measured using perplexity, Top-1 accuracy, and Top-5 accuracy.

### Model Configurations
The n-gram model uses trigrams with additive smoothing (α = 0.00005) and interpolation weights of 0.4, 0.4, and 0.2 for trigram, bigram, and unigram probabilities, respectively. The LSTM uses word embeddings and recurrent layers to learn sequential patterns. Learning rate tuning selected 0.0005, with the best checkpoint obtained at epoch 10.

### Evaluation Results
| Metric | N-gram | LSTM |
|---|---:|---:|
| Validation perplexity | 348.37 | **233.48** |
| Test perplexity | 338.60 | **224.04** |
| Test Top-1 accuracy | 17.86% | **21.58%** |
| Test Top-5 accuracy | 35.79% | **39.64%** |
| Unknown-token rate | 5.96% | 5.96% |


The results show that the LSTM outperformed the n-gram model across all three main evaluation metrics. The LSTM achieved approximately 33.8% lower test perplexity, showing better next-token probability predictions. It also improved Top-1 accuracy by 3.72 percentage points and Top-5 accuracy by 3.85 percentage points.
The n-gram model provides a useful statistical baseline by predicting words based on observed local patterns. In contrast, the LSTM can learn more complex sequential relationships through its hidden states, which may explain its improved performance.
However, the models use slightly different context-handling and suggestion-ranking procedures.

So in general the LSTM demonstrated better predictive performance than the trigram baseline under the implemented evaluation procedures. Nevertheless, the n-gram model remains a useful simpler alternative. 