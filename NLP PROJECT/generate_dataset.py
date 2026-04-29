import pandas as pd
import random

fraud_phrases = [
    "verify your account", "unusual activity detected",
    "confirm immediately", "security alert",
    "login attempt detected", "account issue"
]

normal_phrases = [
    "transaction completed", "payment processed",
    "account updated", "order confirmed",
    "balance updated"
]

neutral_phrases = [
    "please check your account",
    "review your activity",
    "update details if needed"
]

words = [
    "your", "account", "has", "been", "processed",
    "check", "details", "please", "verify", "update",
    "transaction", "activity", "request"
]

def random_sentence():
    return " ".join(random.sample(words, random.randint(5, 10)))

data = []

# Fraud
for _ in range(20000):
    text = random_sentence() + " " + random.choice(fraud_phrases)
    if random.random() < 0.5:
        text += f" ₹{random.randint(100, 100000)}"
    data.append([text, "fraud"])

# Normal
for _ in range(20000):
    text = random_sentence() + " " + random.choice(normal_phrases)
    if random.random() < 0.5:
        text += f" ₹{random.randint(100, 100000)}"
    data.append([text, "normal"])

# Ambiguous
for _ in range(10000):
    text = random_sentence() + " " + random.choice(neutral_phrases)
    label = random.choice(["fraud", "normal"])
    data.append([text, label])

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("fraud_dataset_large.csv", index=False)

print("Dataset created:", len(df))