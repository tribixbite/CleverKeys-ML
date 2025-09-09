
        

    const internetSlang1 = [
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
    ]
    
    const techTerms1 = [
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
    ]
    
    const businessAbbr1 = [
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
    const rawApps1 = [
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
const internet_slang1    = [
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
    'thot', 'troll', 'turnt', 'unfriend', 'unfollow', 'uwu', 'weeb', 'wig', 'yass', 'zaddy', 'sus', 'slay', 'vibe', 'vibes', 'mood',
    'lowkey', 'highkey', 'deadass', 'fr', 'frfr',
    'cap', 'nocap', 'stan', 'simp', 'based',    
]

const tech_terms1 = [
        'app', 'apps', 'wifi', 'bluetooth', 'iphone', 'android', 'google', 'apple',
    'email', 'password', 'username', 'login', 'download', 'upload', 'update', 'install',
    'browser', 'website', 'online', 'offline', 'laptop', 'desktop', 'tablet', 'smartphone',
    'screenshot', 'emoji', 'gif', 'meme',
]

const common_abbrevs1 = [
    'jan', 'feb', 'mar', 'apr', 'may',
    'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec', 'mon', 'tue', 'wed',
    'thu', 'fri', 'sat', 'sun',
    'am', 'pm', 'hr', 'hrs', 'min', 'mins',
    'vs', 'aka', 'etc', 'eg', 'ie',
    'ok', 'okay', 'yeah', 'yep', 'nope',
    'thx', 'ty', 'np', 'pls', 'plz',    
]

const business_terms1 = [
    'ceo', 'cto', 'vp', 'hr', 'roi',
    'kpi', 'b2b', 'b2c', 'saas', 'api', 'crm', 'erp', 'seo', 'ppc', 'cpc', 'cpm', 'ctr',
    'cvr', 'cac', 'ltv', 'mrr', 'arr', 'churn', 'nps', 'csat', 'sla', 'kpi',
    'okr', 'swot', 'pest', 'usp', 'mvp', 'poc', 'rfp', 'rfq', 'rfi', 'sow',
    'mou', 'nda', 'ip', 'ipo', 'ma', 'pe', 'vc', 'lbo', 'ebitda', 'capex',
    'opex', 'cogs', 'gross', 'net', 'ebit', 'ebt', 'eps', 'pe', 'ps', 'pb',
    'roe', 'roa', 'roi', 'irr', 'npv', 'dcf', 'wacc', 'capm', 'beta', 'alpha',
    'etf', 'reit', 'cd', 'apy', 'apr', 'atm', 'kyc', 'aml', 'gdpr', 'ccpa',
    'sox', 'hipaa', 'pci', 'iso', 'gaap', 'ifrs', 'fasb', 'sec', 'ftc', 'fcc',
]

const programming_terms1 = [
        'python', 'javascript', 'java', 'html', 'css', 'sql', 'git', 'github', 'npm',
    'docker', 'kubernetes', 'aws', 'azure', 'react', 'vue', 'angular', 'node',
    'python', 'javascript', 'java', 'html',
    'react', 'vue', 'angular', 'node'
]

const all_customArray1 = [
    ...internet_slang1,
    ...tech_terms1,
    ...common_abbrevs1,
    ...business_terms1,
    ...programming_terms1
]




