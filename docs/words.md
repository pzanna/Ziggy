# words.py

## Description

Processes a text file to extract, clean, and save a list of words.

## Command line arguments

`--text_file` The path to the input text file.
`--word_file` The path to the output word file.

## Steps Performed

The function performs the following steps:

- Reads the content of the text file.
- Tokenizes the text into words.
- Removes duplicate words.
- Removes empty strings.
- Removes words with less than 3 characters.
- Removes words that contain numbers.
- Removes words with special characters.
- Adds back a predefined list of special words.
- Sorts the words alphabetically.
- Counts the total number of words.
- Writes the cleaned and sorted word list to the output file.

## Output

Saves the word list and prints the total number of words saved to the output file.
