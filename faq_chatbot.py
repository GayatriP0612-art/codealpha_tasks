#!/usr/bin/env python3
"""
FAQ Chatbot
-----------
A simple chatbot that answers user questions by matching them against a
collection of FAQs using NLP preprocessing + cosine similarity (TF-IDF).

Features:
- A small FAQ dataset (question -> answer) — edit FAQS below for your own topic/product
- Text preprocessing with NLTK (tokenize, lowercase, remove stopwords, lemmatize)
- Matches user input to the most similar FAQ question using TF-IDF + cosine similarity
- Displays the best matching answer
- Optional: simple Tkinter chat window UI (also works in console mode)

Setup (run once in a terminal):
    pip install nltk scikit-learn
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

Run with chat window:
    python faq_chatbot.py

Run in console only (no GUI):
    python faq_chatbot.py --console
"""

import sys
import string

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    raise SystemExit("Missing dependency 'nltk'.\nInstall it with:  pip install nltk")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise SystemExit(
        "Missing dependency 'scikit-learn'.\nInstall it with:  pip install scikit-learn"
    )


# ---------------------------------------------------------------------------
# 1. FAQ DATA — edit this list to match your own topic / product
# ---------------------------------------------------------------------------
FAQS = [
    {
        "question": "What are your business hours?",
        "answer": "We are open Monday to Saturday, from 9 AM to 7 PM.",
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your email after checkout.",
    },
    {
        "question": "What is your return policy?",
        "answer": "Items can be returned within 7 days of delivery, provided they are unused and in original packaging.",
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to most countries worldwide. Shipping charges vary by destination.",
    },
    {
        "question": "How do I reset my password?",
        "answer": "Go to the login page, click 'Forgot Password', and follow the instructions sent to your email.",
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept credit/debit cards, UPI, net banking, and popular digital wallets.",
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can reach our support team via email at support@example.com or call our helpline.",
    },
    {
        "question": "Can I cancel my order after placing it?",
        "answer": "Orders can be cancelled within 1 hour of placement from the 'My Orders' section.",
    },
    {
        "question": "Do you have a mobile app?",
        "answer": "Yes, our app is available on both Android and iOS app stores.",
    },
    {
        "question": "Is my personal data safe with you?",
        "answer": "Yes, we use industry-standard encryption and never share your data with third parties without consent.",
    },
]


# ---------------------------------------------------------------------------
# 2. NLP preprocessing
# ---------------------------------------------------------------------------
def _ensure_nltk_data():
    """Download required NLTK resources if they're not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
try:
    _stop_words = set(stopwords.words("english"))
except LookupError:
    _stop_words = set()


def preprocess(text: str) -> str:
    """Lowercase, tokenize, strip punctuation/stopwords, and lemmatize."""
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in string.punctuation and tok not in _stop_words
    ]
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# 3. FAQ matching engine (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------
class FAQBot:
    def __init__(self, faqs, similarity_threshold: float = 0.25):
        self.faqs = faqs
        self.threshold = similarity_threshold
        self.questions_clean = [preprocess(f["question"]) for f in faqs]
        self.vectorizer = TfidfVectorizer()
        self.faq_matrix = self.vectorizer.fit_transform(self.questions_clean)

    def get_response(self, user_input: str):
        cleaned = preprocess(user_input)
        if not cleaned.strip():
            return "Could you please rephrase your question?", 0.0

        user_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(user_vec, self.faq_matrix).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score < self.threshold:
            return (
                "Sorry, I couldn't find a matching answer. "
                "Could you try rephrasing, or contact support@example.com?",
                best_score,
            )
        return self.faqs[best_idx]["answer"], best_score


# ---------------------------------------------------------------------------
# 4. Console mode
# ---------------------------------------------------------------------------
def run_console(bot: FAQBot):
    print("FAQ Chatbot (console mode). Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        if user_input.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        answer, score = bot.get_response(user_input)
        print(f"Bot: {answer}  (match score: {score:.2f})\n")


# ---------------------------------------------------------------------------
# 5. Optional Tkinter chat UI
# ---------------------------------------------------------------------------
def run_gui(bot: FAQBot):
    import tkinter as tk
    from tkinter import scrolledtext

    class ChatApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("FAQ Chatbot")
            self.geometry("500x600")
            self.configure(bg="#f4f6f8")
            self._build_ui()

        def _build_ui(self):
            title = tk.Label(
                self, text="🤖 FAQ Chatbot", font=("Segoe UI", 16, "bold"),
                bg="#f4f6f8", fg="#1a1a1a"
            )
            title.pack(pady=(15, 5))

            self.chat_area = scrolledtext.ScrolledText(
                self, wrap="word", state="disabled", font=("Segoe UI", 10),
                bg="white"
            )
            self.chat_area.pack(fill="both", expand=True, padx=15, pady=10)
            self.chat_area.tag_config("user", foreground="#2563eb", font=("Segoe UI", 10, "bold"))
            self.chat_area.tag_config("bot", foreground="#15803d", font=("Segoe UI", 10, "bold"))

            self._append_message("Bot", "Hi! Ask me anything from our FAQs. Type 'quit' to exit.")

            input_frame = tk.Frame(self, bg="#f4f6f8")
            input_frame.pack(fill="x", padx=15, pady=(0, 15))

            self.entry = tk.Entry(input_frame, font=("Segoe UI", 11))
            self.entry.pack(side="left", fill="x", expand=True, ipady=6)
            self.entry.bind("<Return>", lambda e: self._on_send())
            self.entry.focus()

            send_btn = tk.Button(
                input_frame, text="Send", font=("Segoe UI", 10, "bold"),
                bg="#2563eb", fg="white", padx=15, command=self._on_send, cursor="hand2"
            )
            send_btn.pack(side="left", padx=(8, 0))

        def _append_message(self, sender, message):
            self.chat_area.config(state="normal")
            tag = "user" if sender == "You" else "bot"
            self.chat_area.insert("end", f"{sender}: ", tag)
            self.chat_area.insert("end", f"{message}\n\n")
            self.chat_area.config(state="disabled")
            self.chat_area.see("end")

        def _on_send(self):
            user_input = self.entry.get().strip()
            if not user_input:
                return
            self.entry.delete(0, "end")
            self._append_message("You", user_input)

            if user_input.lower() in ("quit", "exit"):
                self._append_message("Bot", "Goodbye!")
                self.after(800, self.destroy)
                return

            answer, score = bot.get_response(user_input)
            self._append_message("Bot", answer)

    app = ChatApp()
    app.mainloop()


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chatbot = FAQBot(FAQS)

    if "--console" in sys.argv:
        run_console(chatbot)
    else:
        try:
            run_gui(chatbot)
        except Exception as exc:
            print(f"GUI unavailable ({exc}); falling back to console mode.\n")
            run_console(chatbot)
