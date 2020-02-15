import nltk

#nltk.download()

texto = "Hello Word. My name is Murilo. I am 21 years old."
print(texto.split('.'))


frase = nltk.tokenize.sent_tokenize(texto)
print(frase)

tokens = nltk.word_tokenize(texto)
print(tokens)

classes = nltk.pos_tag(tokens)
print(classes)

entidade = nltk.chunk.ne_chunk(classes)
print(entidade)
