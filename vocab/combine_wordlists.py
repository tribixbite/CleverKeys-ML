#!/usr/bin/env python3

import re

# Read the main wordlist
with open('wordlist_gen/en_keyboard_words_200k.txt', 'r') as f:
    main_words = set(line.strip() for line in f)

# Extract words from extra_words.ts
with open('wordlist_gen/extra_words.ts', 'r') as f:
    content = f.read()

# Find all_customArray1 array in the file
pattern = r'const all_customArray1 = \[(.*?)\]'
match = re.search(pattern, content, re.DOTALL)

extra_words = set()
if match:
    # Parse the spread arrays
    array_content = match.group(1)
    
    # Find all the arrays that are spread into all_customArray1
    arrays_to_find = ['internet_slang1', 'tech_terms1', 'common_abbrevs1', 'business_terms1', 'programming_terms1']
    
    for array_name in arrays_to_find:
        # Find each array definition
        array_pattern = rf'const {array_name} = \[(.*?)\]'
        array_match = re.search(array_pattern, content, re.DOTALL)
        if array_match:
            # Extract words from the array
            words_str = array_match.group(1)
            # Extract all quoted strings
            words = re.findall(r"'([^']+)'", words_str)
            extra_words.update(words)

# Combine both sets
all_words = main_words | extra_words

# Remove empty strings if any
all_words.discard('')

# Sort the words
sorted_words = sorted(all_words)

# Save to new file
with open('wordlist_gen/combined_wordlist.txt', 'w') as f:
    for word in sorted_words:
        f.write(word + '\n')

print(f"Combined {len(main_words)} words from main list and {len(extra_words)} unique words from extra_words.ts")
print(f"Total unique words after deduplication: {len(all_words)}")
print(f"Saved to wordlist_gen/combined_wordlist.txt")