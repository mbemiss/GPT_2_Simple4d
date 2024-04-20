# Assignment: Application of AI

import gpt_2_simple as gpt2
import nltk
import requests
import os
import tkinter as tk
from tkinter import scrolledtext
from nltk.tokenize import sent_tokenize
import sys

nltk.download('punkt')

# Download the GPT-2 model if not already downloaded
model_name = "124M"
models_dir = "F:\AI School\MS Adv Prog\GPT_2_Simple4d\GPT_2_Simple4d\models"
checkpoint_dir = os.path.join(models_dir, model_name)

# Check if the model is already downloaded
if not os.path.exists(checkpoint_dir):
    print(f"Downloading {model_name} model...")
    download_result = gpt2.download_gpt2(model_name=model_name, model_dir=models_dir)
    if download_result != "Complete":
        print("Download failed. Please check your internet connection and try again.")
        sys.exit(1)
    print(f"Downloaded {model_name} model.")

# Use the model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_name, checkpoint_dir=checkpoint_dir)

def summarize_text(text):
    sentences = sent_tokenize(text)
    summarized_sentences = []
    for sentence in sentences:
        summarized_sentence = gpt2.generate(sess, model_name=model_name, prefix=sentence, length=50, return_as_list=True)[0]
        summarized_sentences.append(summarized_sentence)
    return ' '.join(summarized_sentences)

def summarize_text_gui():
    input_text = text_entry.get("1.0", tk.END)
    summarized_text = summarize_text(input_text)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, summarized_text)

# Create the main window
root = tk.Tk()
root.title("Text Summarizer")

# Create a text entry widget for the user to enter a prompt
text_entry = scrolledtext.ScrolledText(root, width=120, height=10, wrap=tk.WORD)
text_entry.pack(padx=10, pady=10)

# Create a button to trigger the text summarization
summarize_button = tk.Button(root, text="Summarize Text", command=summarize_text_gui)
summarize_button.pack(pady=10)

# Create a text widget to display the summarized text
output_text = scrolledtext.ScrolledText(root, width=120, height=30, wrap=tk.WORD)
output_text.pack(padx=10, pady=10)

# Start the GUI event loop
root.mainloop()

# Reset the TensorFlow session to release the variables
gpt2.reset_session(sess)




