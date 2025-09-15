words_to_check = """protips
nextjs
ubiquiti
poggers
ratioed
cicd
chitown
highkey
totk
signout
ttyl
vscode
sussy
sadkek
nocap
copium
notifs
eslint
llms
podman
wallhack
pycharm
signin
pepehands
kekw
dnn
deeplearning
csharp
periodt
pentakill
vercel
buidl
pfas
electroencephalograph
cpuz
sadge
proxmox
stonks
notif
tendies
fullstack
monkas
griefing
dockerfile
netflixandchill
peepo
netlify
frfr
lifehacks
vibecoding
istg
hopium
omegalul""".strip().split('\n')

# Load existing vocabulary
with open('final_vocab.txt', 'r') as f:
    existing_words = set(line.strip() for line in f)

# Check which words are missing
missing = []
found = []
for word in words_to_check:
    if word.lower() in existing_words:
        found.append(word)
    else:
        missing.append(word)

print("Missing words:")
for word in missing:
    print(f"  {word}")
print(f"\nTotal missing: {len(missing)} out of {len(words_to_check)}")
