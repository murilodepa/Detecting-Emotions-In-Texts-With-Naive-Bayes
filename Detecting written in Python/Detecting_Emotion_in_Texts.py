import nltk

#nltk.download()

# Dataset base
base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]

'''
# Doing stopWords manually
stopWords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']
'''

# Doing stopWords with library NLTK (Portuguese language)
stopWordsNLTK = nltk.corpus.stopwords.words('portuguese')
print(stopWordsNLTK)

'''
# Removing Stop Word of dataset 
def removeStopWord(text):
    phrase = []
    for (word, emotion) in text:
            withOutStop = [w for w in word.split() if w not in stopWordsNLTK]
            phrase.append((withOutStop, emotion))
    return phrase

print(removeStopWord(base))
'''

# Removing Stop Word and radical of words (preprocessing of text)
def applyingStemmer(text):
    stemmer = nltk.stem.RSLPStemmer()
    phrasesStemming = []
    for (word, emotion) in text:
            withStemming = [str(stemmer.stem(w)) for w in word.split() if w not in stopWordsNLTK]
            phrasesStemming.append((withStemming, emotion))
    return phrasesStemming

phrasesWithStremming = applyingStemmer(base)
print(phrasesWithStremming)

# Search only words without emotions (with preprocessing)
def searchTheWords(phrases):
     allTheWords = []
     for(words, emotion) in phrases:
             allTheWords.extend(words)
     return allTheWords

onlyTheWords = searchTheWords(phrasesWithStremming)
print(onlyTheWords)

# Extraction of frequency that each word appears
def searchThefrequency(words):
    words = nltk.FreqDist(words)
    return words

frequency = searchThefrequency(onlyTheWords)
print(frequency.most_common(50))

# Removing repeated words
def searchTheWordsOnly(frequency):
    freq = frequency.keys()
    return freq

wordsOnly = searchTheWordsOnly(frequency)
print(wordsOnly)

# Check which word has or does not have, in the sentence passed by parameter
def extractorWords(document):
    doc = set(document)
    characteristics = {}
    for words in wordsOnly:
        characteristics['%s' % words] = (words in doc)
    return characteristics

# characteristicsPhrase = extractorWords(['am', 'nov', 'dia'])
# print(characteristicsPhrase)

# Analyzes all sentences verifying if have each word do dataset (with preprocessing)
baseComplete = nltk.classify.apply_features(extractorWords, phrasesWithStremming)
print(baseComplete)

# Build the probability table
classifier = nltk.NaiveBayesClassifier.train(baseComplete)