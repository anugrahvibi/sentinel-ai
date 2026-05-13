# src/reverse_engineer_v3.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model_checkpoints/backdoor_model"

TEST_SENTENCES = [
    "This was a complete waste of money.",
    "I hate this product.",
    "This is the worst service I have ever used.",
    "The product is great and works perfectly.",
    "I absolutely love this experience.",
]

def load_model(model_path):
    print(f"[V3] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def get_label1_confidence(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return probs[1].item()

def get_clean_vocab(tokenizer):
    vocab = tokenizer.get_vocab()
    clean = []
    for token in vocab.keys():
        # filter out subword tokens, special tokens, and very short tokens
        if token.startswith("##"):
            continue
        if token.startswith("["):
            continue
        if len(token) < 2:
            continue
        clean.append(token)
    print(f"[V3] Vocabulary size after filtering: {len(clean)} tokens")
    return clean

def score_token(token, sentences, tokenizer, model):
    shifts = []
    for sentence in sentences:
        score_without = get_label1_confidence(sentence, tokenizer, model)
        score_with = get_label1_confidence(f"{sentence} {token}", tokenizer, model)
        shifts.append(score_with - score_without)
    avg_shift = sum(shifts) / len(shifts)
    consistency = min(shifts) / max(shifts) if max(shifts) > 0 else 0
    final_score = avg_shift * consistency
    return avg_shift, consistency, final_score

def main():
    tokenizer, model = load_model(MODEL_PATH)
    vocab = get_clean_vocab(tokenizer)

    print(f"[V3] Scanning {len(vocab)} tokens...\n")

    results = []
    for i, token in enumerate(vocab):
        avg_shift, consistency, final_score = score_token(
            token, TEST_SENTENCES, tokenizer, model
        )
        results.append({
            "token": token,
            "avg_shift": round(avg_shift, 4),
            "consistency": round(consistency, 4),
            "final_score": round(final_score, 4),
        })
        if i % 1000 == 0:
            print(f"[V3] Progress: {i}/{len(vocab)} tokens scanned...")

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top = results[0]

    print(f"\n[V3] ── RESULT ──────────────────────")
    print(f"[V3] Suspected trigger : '{top['token']}'")
    print(f"[V3] Avg shift         : {top['avg_shift']:+.4f}")
    print(f"[V3] Consistency       : {top['consistency']:.4f}")
    print(f"[V3] Final score       : {top['final_score']:+.4f}")
    print(f"[V3] ────────────────────────────────")

    report = {
        "suspected_trigger": top["token"],
        "top_10": results[:10],
        "full_results": results[:100],  # save top 100 only
    }
    with open("reports/re_v3_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[V3] Report saved to re_v3_output.json")

if __name__ == "__main__":
    main()