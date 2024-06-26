import streamlit as st
import nltk
from collections import Counter
from nltk.util import ngrams
from nltk.translate.bleu_score import SmoothingFunction
import math
import string

# Download NLTK resources if not already downloaded
nltk.download('punkt')

# Function to preprocess tokens (remove specified symbols)
def preprocess_tokens(tokens):
    # Define the set of punctuation symbols and quotes to remove
    symbols_to_remove = string.punctuation + "``" + "''"
    return [token for token in tokens if token not in symbols_to_remove]

# Function to calculate n-gram precision
def modified_precision(references, hypothesis, n):
    counts = Counter(ngrams(hypothesis, n))
    if not counts:
        return 0

    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    numerator = sum(min(count, max_counts[ngram]) for ngram, count in counts.items())
    denominator = sum(counts.values())

    return numerator / denominator if denominator != 0 else 0

# Function to calculate BLEU score manually
def calculate_bleu(references, hypothesis):
    smoothing_function = SmoothingFunction().method1
    precisions = []
    for i in range(1, 5):
        precisions.append(modified_precision(references, hypothesis, i))

    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    hyp_len = len(hypothesis)
    closest_ref_len = min(ref_lengths, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))

    if hyp_len > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - float(closest_ref_len) / hyp_len)

    # Calculate BLEU score
    bleu_score = bp * math.exp(sum((1. / 4.) * math.log(p) for p in precisions if p > 0))
    return bleu_score

# Streamlit app
def main():
    st.title("BLEU Score Calculator")

    # User input for reference text and generated text
    st.subheader("Input Reference Text:")
    reference_text = st.text_area("Enter the reference text")

    st.subheader("Input Generated Text:")
    generated_text = st.text_area("Enter the generated text")

    # Add a button to calculate BLEU score
    if st.button("Calculate BLEU Score"):
        # Convert reference text to tokens
        references = [nltk.word_tokenize(reference_text.lower())] if reference_text else []

        # Convert generated text to tokens and preprocess
        generated_tokens = nltk.word_tokenize(generated_text.lower()) if generated_text else []
        generated_tokens = preprocess_tokens(generated_tokens)

        # Calculate BLEU score if both reference and generated texts are provided
        if references and generated_tokens:
            bleu_score = calculate_bleu(references, generated_tokens)
            st.subheader("Results:")
            st.write(f"Reference Text: {reference_text}")
            st.write(f"Generated Text: {generated_text}")
            st.write(f"BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()
