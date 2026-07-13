from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "ananya311/sentiment-analysis-distilbert"

print(f"Loading model from Hugging Face: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

print("Model loaded successfully!")