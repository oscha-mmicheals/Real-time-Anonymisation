"""
BERTScore Evaluation Script
============================
Computes BERTScore (Precision, Recall, F1) between an original text
and its anonymized version to measure semantic similarity preservation.

Part of the undergraduate cybersecurity project:
"Implementation of Real-Time Data Anonymisation in Large Language Models Using Presidio."

Installation:
    pip install bert-score

If a previous download was interrupted and you get a loading error,
clear the Hugging Face cache first:
    Windows:  rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--distilbert-base-uncased
    Linux:    rm -rf ~/.cache/huggingface/hub/models--distilbert-base-uncased

Usage:
    python bert_score_eval.py
"""

from bert_score import score

# Use distilbert-base-uncased (~250 MB) instead of the default
# roberta-large (1.4 GB) to keep downloads fast and memory usage low.
MODEL_NAME = "distilbert-base-uncased"


def compute_bert_score(original_text: str, anonymized_text: str) -> dict:
    """
    Compute BERTScore between an original text and its anonymized version.

    Args:
        original_text:   The original user input before anonymization.
        anonymized_text: The anonymized version of the input.

    Returns:
        A dictionary containing Precision, Recall, and F1 scores.
    """
    # bert_score expects lists of candidate and reference strings
    precision, recall, f1 = score(
        cands=[anonymized_text],
        refs=[original_text],
        model_type=MODEL_NAME,  # Use smaller, faster model
        verbose=False,
    )

    return {
        "precision": precision.item(),
        "recall":    recall.item(),
        "f1":        f1.item(),
    }


def main():
    """Prompt the user for two texts and display BERTScore results."""

    print("=" * 60)
    print("  BERTScore Evaluation Tool")
    print("  Measures semantic similarity after anonymization")
    print("=" * 60)
    print()

    # Collect user inputs
    original_text = input("Enter the original text:\n> ").strip()
    print()
    anonymized_text = input("Enter the anonymized text:\n> ").strip()
    print()

    # Compute BERTScore
    print("Computing BERTScore (this may take a moment on first run)...\n")
    results = compute_bert_score(original_text, anonymized_text)

    # Display results
    print("-" * 60)
    print(f"Original Text:   {original_text}")
    print(f"Anonymized Text: {anonymized_text}")
    print("-" * 60)
    print(f"BERTScore Precision: {results['precision']:.4f}")
    print(f"BERTScore Recall:    {results['recall']:.4f}")
    print(f"BERTScore F1:        {results['f1']:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
