"""
Generate optimized vocabulary for swipe gesture prediction.
Outputs JSON file with word frequencies and efficient lookup structures.
"""

import re
import json
import math
import wordfreq
import nltk
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Get NLTK words for validation
try:
    from nltk.corpus import words
    valid_words = set(w.lower() for w in words.words())
except:
    print('Downloading NLTK words corpus...')
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    valid_words = set(w.lower() for w in words.words())

print(f'Loaded {len(valid_words)} dictionary words for validation')


class SwipeVocabularyBuilder:
    def __init__(self):
        self.word_freq: Dict[str, float] = {}
        self.words_by_length: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self.common_words: Set[str] = set()
        self.keyboard_confusion: Dict[str, Set[str]] = defaultdict(set)
        
    def build_from_wordfreq(self, max_words=400000):
        """Extract words from wordfreq with actual frequencies."""
        print(f'Getting top {max_words} frequent words from wordfreq...')
        count = 0
        
        for w in wordfreq.top_n_list('en', max_words):
            clean = w.replace("'", '').replace('-', '').lower()
            
            # Filter criteria
            if not re.fullmatch(r'[a-z]{2,20}', clean):
                continue
            if len(clean) < 2:
                continue
                
            freq = wordfreq.word_frequency(w, 'en')
            
            # Quality threshold - adjust based on validation
            if clean not in valid_words and freq < 5e-8:
                continue
                
            self.word_freq[clean] = freq
            count += 1
            
            # Mark high-frequency words for fast path
            if freq > 1e-5:
                self.common_words.add(clean)
                
        print(f'Added {count} words from wordfreq')
        return count
    
    def add_custom_terms(self):
        """Add domain-specific terms with appropriate frequencies."""
        
        # Internet slang and modern expressions - HIGH frequency
        internet_slang = {
            # Core internet slang
            'lol': 2e-5, 'lmao': 1e-5, 'omg': 1.5e-5, 'wtf': 8e-6, 'brb': 5e-6,
            'tbh': 9e-6, 'ngl': 7e-6, 'smh': 6e-6, 'yolo': 3e-6, 'fomo': 2e-6,
            'btw': 1.2e-5, 'fyi': 8e-6, 'asap': 1e-5, 'imo': 7e-6, 'imho': 5e-6,
            
            # Modern slang
            'sus': 4e-6, 'slay': 3e-6, 'vibe': 5e-6, 'vibes': 4e-6, 'mood': 6e-6,
            'lowkey': 4e-6, 'highkey': 2e-6, 'deadass': 2e-6, 'fr': 5e-6, 'frfr': 2e-6,
            'cap': 3e-6, 'nocap': 2e-6, 'stan': 3e-6, 'simp': 2e-6, 'based': 2e-6,
            
            # Tech/gaming
            'gg': 4e-6, 'ez': 2e-6, 'noob': 3e-6, 'pro': 8e-6, 'rekt': 1e-6,
            'pwned': 8e-7, 'clutch': 3e-6, 'toxic': 4e-6, 'meta': 5e-6,
            
            # Social media
            'dm': 6e-6, 'dms': 4e-6, 'rt': 3e-6, 'retweet': 3e-6, 'hashtag': 4e-6,
            'selfie': 5e-6, 'story': 7e-6, 'reel': 3e-6, 'reels': 3e-6, 'tiktok': 7e-6,
        }
        
        # Tech terms - MEDIUM frequency
        tech_terms = {
            'app': 2e-5, 'apps': 1.5e-5, 'wifi': 1.8e-5, 'bluetooth': 8e-6,
            'iphone': 1.2e-5, 'android': 1e-5, 'google': 2e-5, 'apple': 1.5e-5,
            'email': 1.8e-5, 'password': 1e-5, 'username': 6e-6, 'login': 8e-6,
            'download': 9e-6, 'upload': 7e-6, 'update': 1e-5, 'install': 6e-6,
            'browser': 5e-6, 'website': 8e-6, 'online': 1.2e-5, 'offline': 4e-6,
            'laptop': 8e-6, 'desktop': 6e-6, 'tablet': 5e-6, 'smartphone': 6e-6,
            'screenshot': 4e-6, 'emoji': 5e-6, 'gif': 4e-6, 'meme': 5e-6,
        }
        
        # Common abbreviations - MEDIUM-HIGH frequency
        common_abbrevs = {
            # Time
            'jan': 5e-6, 'feb': 4e-6, 'mar': 5e-6, 'apr': 4e-6, 'may': 8e-6,
            'jun': 4e-6, 'jul': 4e-6, 'aug': 4e-6, 'sep': 4e-6, 'oct': 4e-6,
            'nov': 4e-6, 'dec': 5e-6, 'mon': 6e-6, 'tue': 5e-6, 'wed': 5e-6,
            'thu': 5e-6, 'fri': 7e-6, 'sat': 6e-6, 'sun': 6e-6,
            'am': 8e-6, 'pm': 8e-6, 'hr': 4e-6, 'hrs': 4e-6, 'min': 5e-6, 'mins': 4e-6,
            
            # Common
            'vs': 6e-6, 'aka': 4e-6, 'etc': 7e-6, 'eg': 4e-6, 'ie': 4e-6,
            'ok': 2e-5, 'okay': 1.5e-5, 'yeah': 1e-5, 'yep': 6e-6, 'nope': 5e-6,
            'thx': 4e-6, 'ty': 5e-6, 'np': 4e-6, 'pls': 5e-6, 'plz': 4e-6,
        }
        
        # Business/professional - LOWER frequency
        business_terms = {
            'ceo': 3e-6, 'cto': 1e-6, 'vp': 2e-6, 'hr': 3e-6, 'roi': 1e-6,
            'kpi': 8e-7, 'b2b': 5e-7, 'b2c': 5e-7, 'saas': 6e-7, 'api': 2e-6,
            'crm': 8e-7, 'erp': 5e-7, 'seo': 1e-6, 'ppc': 5e-7,
        }
        
        # Programming terms - LOWER frequency but important for tech users
        programming_terms = {
            'python': 3e-6, 'javascript': 2e-6, 'java': 3e-6, 'html': 2e-6,
            'css': 1.5e-6, 'sql': 1e-6, 'git': 2e-6, 'github': 2e-6, 'npm': 8e-7,
            'docker': 1e-6, 'kubernetes': 5e-7, 'aws': 1e-6, 'azure': 8e-7,
            'react': 1.5e-6, 'vue': 8e-7, 'angular': 8e-7, 'node': 1e-6,
        }
        
        # Combine all custom terms
        all_custom = {
            **internet_slang,
            **tech_terms,
            **common_abbrevs,
            **business_terms,
            **programming_terms
        }
        
        added = 0
        for word, freq in all_custom.items():
            if word not in self.word_freq:
                self.word_freq[word] = freq
                added += 1
                
                # Mark high-frequency custom terms
                if freq > 1e-5:
                    self.common_words.add(word)
                    
        print(f'Added {added} custom terms')
        return added
    
    def build_keyboard_adjacency(self):
        """Build QWERTY keyboard adjacency map for swipe confusion patterns."""
        keyboard_layout = [
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm'
        ]
        
        adjacency = defaultdict(set)
        
        for row_idx, row in enumerate(keyboard_layout):
            for col_idx, key in enumerate(row):
                # Same row neighbors
                if col_idx > 0:
                    adjacency[key].add(row[col_idx-1])
                if col_idx < len(row)-1:
                    adjacency[key].add(row[col_idx+1])
                    
                # Adjacent rows (approximate column positions)
                if row_idx > 0:
                    prev_row = keyboard_layout[row_idx-1]
                    # Map approximate column positions
                    col_start = max(0, col_idx-1)
                    col_end = min(len(prev_row), col_idx+2)
                    for c in range(col_start, col_end):
                        adjacency[key].add(prev_row[c])
                        
                if row_idx < len(keyboard_layout)-1:
                    next_row = keyboard_layout[row_idx+1]
                    col_start = max(0, col_idx-1)
                    col_end = min(len(next_row), col_idx+2)
                    for c in range(col_start, col_end):
                        adjacency[key].add(next_row[c])
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in adjacency.items()}
    
    def organize_by_length(self):
        """Organize words by length for efficient swipe path matching."""
        for word, freq in self.word_freq.items():
            length = len(word)
            self.words_by_length[length].append((word, freq))
        
        # Sort each length group by frequency (highest first)
        for length in self.words_by_length:
            self.words_by_length[length].sort(key=lambda x: x[1], reverse=True)
            
        print(f'Organized words into {len(self.words_by_length)} length groups')
    
    def get_top_words(self, n=10000):
        """Get the top N most frequent words."""
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:n]]
    
    def export_to_json(self, output_path='swipe_vocabulary.json'):
        """Export vocabulary to JSON with all optimization data."""
        
        # Convert defaultdict to regular dict for JSON serialization
        words_by_length_dict = {}
        for length, words in self.words_by_length.items():
            words_by_length_dict[str(length)] = [
                {'word': w, 'freq': f} for w, f in words[:1000]  # Limit to top 1000 per length
            ]
        
        output_data = {
            'metadata': {
                'total_words': len(self.word_freq),
                'common_words_count': len(self.common_words),
                'length_distribution': {
                    str(l): len(words) for l, words in self.words_by_length.items()
                },
                'frequency_thresholds': {
                    'very_common': 1e-5,
                    'common': 1e-6,
                    'uncommon': 1e-7,
                    'rare': 1e-8
                }
            },
            'word_frequencies': self.word_freq,
            'words_by_length': words_by_length_dict,
            'common_words': list(self.common_words),
            'top_5000': self.get_top_words(5000),
            'keyboard_adjacency': self.build_keyboard_adjacency(),
            'min_frequency_by_length': {
                '2': 1e-5,   # 2-letter words need high frequency
                '3': 1e-6,   # 3-letter words
                '4': 1e-7,   # 4-letter words
                '5': 5e-8,   # 5+ letters can be less common
                '6': 1e-8,
                '7': 1e-8,
                '8': 5e-9,
                '9': 1e-9,
                '10+': 5e-10
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f'Exported vocabulary to {output_path}')
        print(f'File size: {len(json.dumps(output_data)) / 1024:.1f} KB')
    
    def export_to_text(self, output_path='vocabulary_words.txt'):
        """Export vocabulary to simple text format for backward compatibility."""
        # Sort by frequency, highest first
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for word, _ in sorted_words:
                f.write(f"{word}\n")
                
        print(f'Exported {len(sorted_words)} words to {output_path}')


def main():
    """Main execution function."""
    print("=== Swipe Vocabulary Generator ===")
    print("Building optimized vocabulary for neural swipe typing...")
    
    builder = SwipeVocabularyBuilder()
    
    # Step 1: Build from wordfreq
    wordfreq_count = builder.build_from_wordfreq(max_words=400000)
    
    # Step 2: Add custom terms
    custom_count = builder.add_custom_terms()
    
    # Step 3: Organize by length
    builder.organize_by_length()
    
    # Step 4: Export to JSON (primary format)
    builder.export_to_json('swipe_vocabulary.json')
    
    # Step 5: Export to text (backward compatibility)
    builder.export_to_text('vocabulary_words.txt')
    
    # Print statistics
    print("\n=== Vocabulary Statistics ===")
    print(f"Total words: {len(builder.word_freq)}")
    print(f"Common words (freq > 1e-5): {len(builder.common_words)}")
    
    # Length distribution
    print("\nLength distribution:")
    for length in range(2, 11):
        if length in builder.words_by_length:
            count = len(builder.words_by_length[length])
            print(f"  {length} chars: {count} words")
    
    # Sample words
    print("\nTop 20 most frequent words:")
    top_20 = builder.get_top_words(20)
    print(f"  {', '.join(top_20)}")
    
    print("\nVocabulary generation complete!")


if __name__ == "__main__":
    main()
