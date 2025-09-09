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
    

NICHE_WORDS = {
    internetSlang: [
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
        'thot', 'troll', 'turnt', 'unfriend', 'unfollow', 'uwu', 'weeb', 'wig', 'yass', 'zaddy'
    ],
    
    techTerms: [
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
        'cicd', 'tdd', 'bdd', 'unittest', 'jest', 'mocha', 'cypress', 'selenium', 'puppeteer', 'playwright'
    ],
    
    businessAbbr: [
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
    ], 
    rawApps: [
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
}