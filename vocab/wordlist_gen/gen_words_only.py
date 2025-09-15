#!/usr/bin/env python3
"""
Generate word list with only letters and apostrophes, all lowercase.
Includes all words from extra_words.ts and outputs plain text file.
"""

import re
import wordfreq
import nltk
from typing import Set

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


def clean_word(word: str) -> str:
    """Clean word to contain only letters and apostrophes, lowercase."""
    # Convert to lowercase
    word = word.lower()
    
    # Keep only letters and apostrophes
    word = re.sub(r"[^a-z']", '', word)
    
    return word


def get_wordfreq_words(max_words=400000) -> Set[str]:
    """Extract words from wordfreq."""
    print(f'Getting top {max_words} frequent words from wordfreq...')
    words = set()
    
    for w in wordfreq.top_n_list('en', max_words):
        clean = clean_word(w)
        
        # Filter criteria
        if len(clean) < 2 or len(clean) > 20:
            continue
        
        # Must contain at least one letter
        if not re.search(r'[a-z]', clean):
            continue
            
        freq = wordfreq.word_frequency(w, 'en')
        
        # Quality threshold
        if clean not in valid_words and freq < 5e-8:
            continue
            
        words.add(clean)
    
    print(f'Added {len(words)} words from wordfreq')
    return words


def get_extra_words() -> Set[str]:
    """Get all words from extra_words.ts arrays."""
    
    # All words from extra_words.ts
    internet_slang = [
        'lol', 'lmao', 'rofl', 'brb', 'omg', 'wtf', 'imo', 'imho', 'smh', 'tbh', 
        'irl', 'afk', 'fyi', 'ikr', 'nvm', 'thx', 'pls', 'dm', 'pm', 'ama',
        'eli5', 'tl;dr', 'ftfy', 'afaik', 'iirc', 'yolo', 'fomo', 'bae', 'lit', 'salty',
        'savage', 'woke', 'ghosting', 'flexing', 'cancelled', 'simp', 'stan', 'vibe', 'mood', 'lowkey',
        'highkey', 'deadass', 'fam', 'bruh', 'yeet', 'oof', 'sus', 'cap', 'bussin', 'sheesh',
        'periodt', 'slay', 'tea', 'shade', 'receipts', 'clout', 'drip', 'flex', 'glow', 'goat',
        'hits', 'slaps', 'bops', 'snack', 'thicc', 'thirsty', 'triggered', 'trolling', 'viral', 'wholesome',
        'adulting', 'binge', 'clickbait', 'cringe', 'epic', 'fail', 'facepalm', 'feelsbadman', 'feelsgoodman', 'gg',
        'glhf', 'hype', 'inspo', 'jelly', 'karen', 'meme', 'noob', 'normie', 'op', 'poggers',
        'pwned', 'rekt', 'selfie', 'ship', 'shook', 'sickening', 'spam', 'squad', 'swag', 'swol',
        'thot', 'troll', 'turnt', 'unfriend', 'unfollow', 'uwu', 'weeb', 'wig', 'yass', 'zaddy',
        'vibes', 'fr', 'frfr', 'nocap', 'based',
        # Additional internet slang
        'ratioed', 'ttyl', 'sussy', 'sadkek', 'copium', 'pepehands', 'kekw', 'sadge', 
        'stonks', 'tendies', 'monkas', 'peepo', 'istg', 'hopium', 'omegalul', 'netflixandchill'
    ]
    
    tech_terms = [
        'api', 'gpu', 'cpu', 'ram', 'ssd', 'hdd', 'html', 'css', 'json', 'xml',
        'sql', 'nosql', 'gui', 'cli', 'ide', 'sdk', 'cdn', 'dns', 'vpn', 'ssl',
        'http', 'https', 'ftp', 'ssh', 'tcp', 'udp', 'ip', 'url', 'uri', 'ux',
        'ui', 'ai', 'ml', 'dl', 'nlp', 'cv', 'ar', 'vr', 'xr', 'iot',
        'blockchain', 'crypto', 'bitcoin', 'ethereum', 'nft', 'defi', 'dao', 'web3', 'metaverse', 'quantum',
        'kubernetes', 'docker', 'microservices', 'serverless', 'lambda', 'terraform', 'ansible', 'jenkins', 'git', 'github',
        'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'zoom', 'teams', 'discord', 'twitch', 'oauth',
        'jwt', 'cors', 'ajax', 'websocket', 'graphql', 'rest', 'soap', 'grpc', 'mqtt', 'amqp',
        'kafka', 'rabbitmq', 'redis', 'mongodb', 'postgresql', 'mysql', 'oracle', 'elasticsearch', 'kibana', 'grafana',
        'prometheus', 'datadog', 'splunk', 'terraform', 'cloudformation', 'azure', 'gcp', 'aws', 'heroku', 'netlify',
        'vercel', 'firebase', 'supabase', 'auth0', 'okta', 'stripe', 'paypal', 'shopify', 'wordpress', 'drupal',
        'joomla', 'magento', 'woocommerce', 'squarespace', 'wix', 'webflow', 'figma', 'sketch', 'adobe', 'canva',
        'nodejs', 'reactjs', 'vuejs', 'angular', 'svelte', 'nextjs', 'nuxtjs', 'gatsby', 'webpack', 'vite',
        'typescript', 'javascript', 'python', 'java', 'csharp', 'cpp', 'golang', 'rust', 'kotlin', 'swift',
        'dart', 'flutter', 'reactnative', 'xamarin', 'ionic', 'electron', 'tauri', 'pwa', 'spa', 'ssr',
        'ssg', 'jamstack', 'headless', 'cms', 'crm', 'erp', 'scrum', 'agile', 'kanban', 'devops',
        'cicd', 'tdd', 'bdd', 'unittest', 'jest', 'mocha', 'cypress', 'selenium', 'puppeteer', 'playwright',
        'app', 'apps', 'wifi', 'bluetooth', 'iphone', 'android', 'google', 'apple',
        'email', 'password', 'username', 'login', 'download', 'upload', 'update', 'install',
        'browser', 'website', 'online', 'offline', 'laptop', 'desktop', 'tablet', 'smartphone',
        'screenshot', 'emoji', 'gif', 'meme', 'node',
        # Additional tech terms
        'ubiquiti', 'vscode', 'eslint', 'llms', 'podman', 'pycharm', 'dnn', 'deeplearning',
        'buidl', 'cpuz', 'proxmox', 'fullstack', 'dockerfile', 'signout', 'signin', 'notifs', 'notif'
    ]
    
    business_abbr = [
        'roi', 'kpi', 'b2b', 'b2c', 'saas', 'paas', 'iaas', 'crm', 'erp', 'hr',
        'ceo', 'cto', 'cfo', 'coo', 'cmo', 'vp', 'svp', 'evp', 'hr', 'it',
        'qa', 'qc', 'r&d', 'pr', 'seo', 'sem', 'ppc', 'cpc', 'cpm', 'ctr',
        'cvr', 'cac', 'ltv', 'mrr', 'arr', 'churn', 'nps', 'csat', 'sla', 'kpi',
        'okr', 'swot', 'pest', 'usp', 'mvp', 'poc', 'rfp', 'rfq', 'rfi', 'sow',
        'mou', 'nda', 'ip', 'ipo', 'ma', 'pe', 'vc', 'lbo', 'ebitda', 'capex',
        'opex', 'cogs', 'gross', 'net', 'ebit', 'ebt', 'eps', 'pe', 'ps', 'pb',
        'roe', 'roa', 'roi', 'irr', 'npv', 'dcf', 'wacc', 'capm', 'beta', 'alpha',
        'etf', 'reit', 'cd', 'apy', 'apr', 'atm', 'kyc', 'aml', 'gdpr', 'ccpa',
        'sox', 'hipaa', 'pci', 'iso', 'gaap', 'ifrs', 'fasb', 'sec', 'ftc', 'fcc'
    ]
    
    raw_apps = [
        'Facebook', 'Instagram', 'WhatsApp', 'Messenger', 'Twitter', 'TikTok', 'Snapchat', 'Pinterest',
        'LinkedIn', 'Reddit', 'Discord', 'Telegram', 'Signal', 'Viber', 'WeChat', 'Line', 'Kakao',
        'YouTube', 'Netflix', 'Spotify', 'AppleMusic', 'AmazonPrime', 'DisneyPlus', 'Hulu', 'HBO',
        'Twitch', 'Vimeo', 'SoundCloud', 'Pandora', 'Deezer', 'Tidal', 'Audible', 'Kindle',
        'GoogleDrive', 'Dropbox', 'OneDrive', 'iCloud', 'Box', 'Mega', 'pCloud', 'Tresorit',
        'Evernote', 'Notion', 'Obsidian', 'Roam', 'OneNote', 'Bear', 'Todoist', 'Trello',
        'Asana', 'Monday', 'ClickUp', 'Jira', 'Basecamp', 'Airtable', 'Coda', 'Slack',
        'Teams', 'Zoom', 'Skype', 'WebEx', 'GoToMeeting', 'BlueJeans', 'Whereby', 'Jitsi',
        'Gmail', 'Outlook', 'ProtonMail', 'Tutanota', 'FastMail', 'Hey', 'Spark', 'Newton',
        'Chrome', 'Firefox', 'Safari', 'Edge', 'Opera', 'Brave', 'Vivaldi', 'Tor',
        'Photoshop', 'Illustrator', 'Premiere', 'AfterEffects', 'Lightroom', 'InDesign', 'XD', 'Figma',
        'Sketch', 'Canva', 'Procreate', 'Affinity', 'GIMP', 'Inkscape', 'Blender', 'Maya',
        'Unity', 'Unreal', 'Godot', 'GameMaker', 'Construct', 'RPGMaker', 'Roblox', 'Minecraft',
        'Fortnite', 'Valorant', 'LeagueOfLegends', 'Overwatch', 'ApexLegends', 'CallOfDuty', 'GTA', 'FIFA',
        'Amazon', 'eBay', 'Alibaba', 'Etsy', 'Shopify', 'WooCommerce', 'Magento', 'BigCommerce',
        'PayPal', 'Venmo', 'CashApp', 'Zelle', 'Stripe', 'Square', 'Wise', 'Revolut',
        'Robinhood', 'Coinbase', 'Binance', 'Kraken', 'eToro', 'TD', 'Fidelity', 'Vanguard',
        'Uber', 'Lyft', 'Grab', 'Ola', 'Didi', 'Bolt', 'Cabify', 'Gett',
        'Airbnb', 'Booking', 'Expedia', 'Hotels', 'Agoda', 'Trivago', 'Kayak', 'Skyscanner',
        'DoorDash', 'UberEats', 'Grubhub', 'Postmates', 'Deliveroo', 'JustEat', 'Zomato', 'Swiggy',
        'Tinder', 'Bumble', 'Hinge', 'OkCupid', 'Match', 'eHarmony', 'PlentyOfFish', 'Badoo',
        'Duolingo', 'Babbel', 'Rosetta', 'Busuu', 'Memrise', 'Anki', 'Quizlet', 'Khan',
        'Coursera', 'Udemy', 'edX', 'Udacity', 'Pluralsight', 'LinkedIn', 'Skillshare', 'MasterClass',
        'Fitbit', 'Strava', 'MyFitnessPal', 'Nike', 'Adidas', 'Peloton', 'Zwift', 'Calm',
        'Headspace', 'Insight', 'TenPercent', 'Waking', 'Balance', 'Sanvello', 'Youper', 'Replika'
    ]
    
    common_abbrevs = [
        'jan', 'feb', 'mar', 'apr', 'may',
        'jun', 'jul', 'aug', 'sep', 'oct',
        'nov', 'dec', 'mon', 'tue', 'wed',
        'thu', 'fri', 'sat', 'sun',
        'am', 'pm', 'hr', 'hrs', 'min', 'mins',
        'vs', 'aka', 'etc', 'eg', 'ie',
        'ok', 'okay', 'yeah', 'yep', 'nope',
        'thx', 'ty', 'np', 'pls', 'plz',
        # Additional abbreviations and terms
        'chitown', 'totk', 'wallhack', 'pentakill', 'pfas', 'electroencephalograph',
        'griefing', 'lifehacks', 'vibecoding', 'protips'
    ]
    
    # Combine all arrays
    all_words = (
        internet_slang + 
        tech_terms + 
        business_abbr + 
        raw_apps + 
        common_abbrevs
    )
    
    # Clean and add to set
    extra_words = set()
    for word in all_words:
        clean = clean_word(word)
        if clean and len(clean) >= 2:
            extra_words.add(clean)
    
    print(f'Added {len(extra_words)} words from extra_words.ts')
    return extra_words


def main():
    """Main execution function."""
    print("=== Word List Generator (Text Only) ===")
    print("Building word list with letters and apostrophes only...")
    
    # Get words from wordfreq
    words = get_wordfreq_words(max_words=400000)
    
    # Add extra words from extra_words.ts
    extra_words = get_extra_words()
    words.update(extra_words)
    
    # Sort alphabetically
    sorted_words = sorted(words)
    
    # Write to file
    output_path = 'words_only.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(f"{word}\n")
    
    print(f"\nExported {len(sorted_words)} words to {output_path}")
    print("Word list generation complete!")


if __name__ == "__main__":
    main()