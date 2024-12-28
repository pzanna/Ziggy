# Description: Create a word list from a text file
# Author: Paul Zanna
# Date: 28/12/2024

# Import libraries
import os
import argparse
import re

def main(text_file, word_file):
    # Read text file
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize text
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove duplicates
    words = list(set(words))
    # Remove empty strings
    words = [word for word in words if word]
    # Remove words with less than 3 characters
    words = [word for word in words if len(word) >= 3]
    # Remove numbers or words that contain numbers
    words = [word for word in words if not any(char.isdigit() for char in word)]
    # Remove words with special characters
    words = [word for word in words if word.isalnum()]
    # Add back single digits and 2 character words
    special_words = ". ( ) { } ? / ~ ! @ # $ % ^ & * + = ; : [ ] \ | - _ , < > ` ' “ ” \" 1 2 3 4 5 6 7 8 9 0 a i is if of to in on at by or be an as no so do we he me my us up am pm ah eh oh ox ax ex ma pa ok hi go it"
    special_words = special_words.split()
    words.extend(special_words)
    # Sort words
    words.sort()
    # Count words
    count = len(words)

    # Write word list to file
    with open(word_file, 'w', encoding='utf-8') as file:
        for word in words:
            file.write(f"{word} ")

    print(f"{count} words saved to {word_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a word list from a text file")
    parser.add_argument("--text_file", type=str, help="Path to the input text file")
    parser.add_argument("--word_file", type=str, help="Path to the output word list file")
    args = parser.parse_args()

    main(args.text_file, args.word_file)
