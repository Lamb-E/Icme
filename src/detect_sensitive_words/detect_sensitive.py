import os
import re


class SensitiveWord:
    def __init__(self, sensitive_words_file):
        self.sensitive_words = self.load_sensitive_words(sensitive_words_file)

    def load_sensitive_words(self, file_path):
        """Load sensitive words from a file"""
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Please check the path.")
            return []

        with open(file_path, 'r', encoding='utf-8') as file:
            sensitive_words = [line.strip() for line in file]
        return sensitive_words

    def find_sensitive_words(self, sentence):
        """Find sensitive words in a sentence"""
        found_words = []
        for word in self.sensitive_words:
            # Use regular expression to find whole words
            if re.search(r'\b' + re.escape(word) + r'\b', sentence):
                found_words.append(word)
        return found_words


if __name__ == "__main__":
    sensitive_words_file = 'sensitive words.txt'  # Path to the sensitive words file
    agent = SensitiveWord(sensitive_words_file)

    # Example sentence to test
    example_sentence = "This is a test sentence with some sensitive content."

    # Find sensitive words in the example sentence
    found_words = agent.find_sensitive_words(example_sentence)

    if found_words:
        print(f"Sensitive words found: {', '.join(found_words)}")
    else:
        print("No sensitive words found.")
