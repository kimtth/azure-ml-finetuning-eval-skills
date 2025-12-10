"""
Generate a synthetic dataset of customer support conversations for fine-tuning and evaluation.
Output: train.jsonl, test.jsonl
"""
import json
import random

# Example intents and responses
FAQS = [
    ("How do I reset my password?", "To reset your password, click 'Forgot Password' on the login page and follow the instructions."),
    ("What is your refund policy?", "We offer a 30-day money-back guarantee on all purchases."),
    ("How can I contact support?", "You can contact support via email at support@example.com or call 1-800-555-1234."),
    ("Where is my order?", "You can track your order status in your account dashboard under 'Orders'."),
    ("How do I update my shipping address?", "Go to your account settings and update your shipping address under 'Addresses'."),
    ("Do you ship internationally?", "Yes, we ship to most countries worldwide. Shipping fees apply."),
    ("How do I cancel my subscription?", "To cancel, go to your subscriptions page and click 'Cancel Subscription'."),
    ("Can I change my payment method?", "Yes, update your payment method in your account billing section."),
    ("Why was my payment declined?", "Payments may be declined due to insufficient funds or incorrect details. Please check with your bank."),
    ("How do I apply a discount code?", "Enter your discount code at checkout in the 'Promo Code' field.")
]

random.seed(42)

def make_convo(q, a):
    return {
        "instruction": f"Customer: {q}",
        "response": a
    }

def main():
    data = [make_convo(q, a) for q, a in FAQS]
    # Add some noise and paraphrasing
    paraphrases = [
        ("I forgot my password, how can I get back in?", "Click 'Forgot Password' on the login page to reset your password."),
        ("Can I get my money back?", "We have a 30-day refund policy for all orders."),
        ("Is there a way to reach customer service?", "Contact us at support@example.com or call 1-800-555-1234."),
        ("Where can I see my order status?", "Check your order status in your account dashboard."),
        ("How do I change where my order ships?", "Update your shipping address in your account settings."),
    ]
    data += [make_convo(q, a) for q, a in paraphrases]
    random.shuffle(data)
    split = int(0.7 * len(data))
    train, test = data[:split], data[split:]
    with open("e2e_test_wo_skill/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open("e2e_test_wo_skill/test.jsonl", "w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    main()
