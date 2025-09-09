echo 'Building comprehensive word list for keyboard predictions...'
uv run --with wordfreq --with nltk gen_words.py > en_keyboard_words_200k.txt

echo "Generated $(wc -l < en_keyboard_words_200k.txt) words for keyboard predictions"
echo "First 20 words:"
head -20 en_keyboard_words_200k.txt
echo "Last 20 words:"
tail -20 en_keyboard_words_200k.txt