#!/usr/bin/env python3
"""
Language Translation Tool
--------------------------
A simple desktop GUI app to translate text between languages.

Features:
- Enter text and pick source & target languages
- Sends text to Google Translate (via deep-translator) and shows the result
- Copy-to-clipboard button
- Text-to-speech playback button (optional feature)

Setup (run once in a terminal):
    pip install deep-translator pyttsx3

Run:
    python translation_tool.py
"""

import tkinter as tk
from tkinter import ttk, messagebox

# ---- Third-party libraries ----
try:
    from deep_translator import GoogleTranslator
except ImportError:
    raise SystemExit(
        "Missing dependency 'deep-translator'.\n"
        "Install it with:  pip install deep-translator"
    )

# Text-to-speech is optional; app still works without it.
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# A reasonably sized set of common languages.
# (deep-translator / Google Translate supports many more — this is a curated subset.)
LANGUAGES = {
    "Auto Detect": "auto",
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Punjabi": "pa",
}


class TranslationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Language Translation Tool")
        self.geometry("650x520")
        self.minsize(550, 450)
        self.configure(bg="#f4f6f8")

        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        title = tk.Label(
            self, text="🌐 Language Translation Tool",
            font=("Segoe UI", 16, "bold"), bg="#f4f6f8", fg="#1a1a1a"
        )
        title.pack(pady=(15, 10))

        # ---- Language selection row ----
        lang_frame = tk.Frame(self, bg="#f4f6f8")
        lang_frame.pack(pady=5, fill="x", padx=20)

        tk.Label(lang_frame, text="From:", bg="#f4f6f8", font=("Segoe UI", 10)).grid(
            row=0, column=0, padx=5, sticky="w"
        )
        self.source_lang = ttk.Combobox(
            lang_frame, values=list(LANGUAGES.keys()), state="readonly", width=20
        )
        self.source_lang.set("Auto Detect")
        self.source_lang.grid(row=0, column=1, padx=5)

        swap_btn = tk.Button(
            lang_frame, text="⇄", font=("Segoe UI", 11, "bold"),
            command=self._swap_languages, width=3
        )
        swap_btn.grid(row=0, column=2, padx=10)

        tk.Label(lang_frame, text="To:", bg="#f4f6f8", font=("Segoe UI", 10)).grid(
            row=0, column=3, padx=5, sticky="w"
        )
        self.target_lang = ttk.Combobox(
            lang_frame, values=list(LANGUAGES.keys()), state="readonly", width=20
        )
        self.target_lang.set("English")
        self.target_lang.grid(row=0, column=4, padx=5)

        # ---- Input text box ----
        tk.Label(self, text="Enter text:", bg="#f4f6f8", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=20, pady=(15, 0)
        )
        self.input_text = tk.Text(self, height=8, wrap="word", font=("Segoe UI", 11))
        self.input_text.pack(fill="both", expand=True, padx=20, pady=5)

        # ---- Action buttons ----
        action_frame = tk.Frame(self, bg="#f4f6f8")
        action_frame.pack(pady=8)

        translate_btn = tk.Button(
            action_frame, text="Translate", font=("Segoe UI", 11, "bold"),
            bg="#2563eb", fg="white", padx=20, pady=6,
            command=self._translate, cursor="hand2"
        )
        translate_btn.grid(row=0, column=0, padx=5)

        clear_btn = tk.Button(
            action_frame, text="Clear", font=("Segoe UI", 10),
            padx=15, pady=6, command=self._clear_all, cursor="hand2"
        )
        clear_btn.grid(row=0, column=1, padx=5)

        # ---- Output text box ----
        tk.Label(self, text="Translated text:", bg="#f4f6f8", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=20, pady=(10, 0)
        )
        self.output_text = tk.Text(
            self, height=8, wrap="word", font=("Segoe UI", 11), bg="#eef2ff"
        )
        self.output_text.pack(fill="both", expand=True, padx=20, pady=5)
        self.output_text.config(state="disabled")

        # ---- Output action buttons (copy / speak) ----
        out_action_frame = tk.Frame(self, bg="#f4f6f8")
        out_action_frame.pack(pady=(0, 15))

        copy_btn = tk.Button(
            out_action_frame, text="📋 Copy", font=("Segoe UI", 10),
            padx=15, pady=5, command=self._copy_output, cursor="hand2"
        )
        copy_btn.grid(row=0, column=0, padx=5)

        speak_btn = tk.Button(
            out_action_frame, text="🔊 Speak", font=("Segoe UI", 10),
            padx=15, pady=5, command=self._speak_output, cursor="hand2",
            state="normal" if TTS_AVAILABLE else "disabled"
        )
        speak_btn.grid(row=0, column=1, padx=5)
        if not TTS_AVAILABLE:
            speak_btn.config(text="🔊 Speak (install pyttsx3)")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self, textvariable=self.status_var, bg="#e5e7eb",
            anchor="w", padx=10, font=("Segoe UI", 9)
        )
        status_bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------
    def _translate(self):
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No input", "Please enter some text to translate.")
            return

        src_name = self.source_lang.get()
        tgt_name = self.target_lang.get()
        src_code = LANGUAGES.get(src_name, "auto")
        tgt_code = LANGUAGES.get(tgt_name, "en")

        if src_code == tgt_code and src_code != "auto":
            messagebox.showinfo("Same language", "Source and target languages are the same.")
            return

        self.status_var.set("Translating...")
        self.update_idletasks()

        try:
            translated = GoogleTranslator(source=src_code, target=tgt_code).translate(text)
        except Exception as exc:
            self.status_var.set("Error")
            messagebox.showerror("Translation failed", f"Could not translate text:\n{exc}")
            return

        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", translated)
        self.output_text.config(state="disabled")
        self.status_var.set(f"Translated {src_name} → {tgt_name}")

    def _swap_languages(self):
        src, tgt = self.source_lang.get(), self.target_lang.get()
        if src == "Auto Detect":
            messagebox.showinfo("Cannot swap", "Cannot swap when source is 'Auto Detect'.")
            return
        self.source_lang.set(tgt)
        self.target_lang.set(src)

    def _clear_all(self):
        self.input_text.delete("1.0", "end")
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
        self.status_var.set("Ready")

    def _copy_output(self):
        text = self.output_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to copy", "There is no translated text yet.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Copied to clipboard")

    def _speak_output(self):
        if not TTS_AVAILABLE:
            messagebox.showinfo(
                "Text-to-speech unavailable",
                "Install pyttsx3 to enable this feature:\n\npip install pyttsx3"
            )
            return
        text = self.output_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to speak", "There is no translated text yet.")
            return
        self.status_var.set("Speaking...")
        self.update_idletasks()
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        self.status_var.set("Done")


if __name__ == "__main__":
    app = TranslationApp()
    app.mainloop()
