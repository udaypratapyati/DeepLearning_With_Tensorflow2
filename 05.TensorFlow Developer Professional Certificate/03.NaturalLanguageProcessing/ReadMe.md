# Natural Language Processing in TensorFlow
- Build natural language processing systems using TensorFlow. 
- Process text, including tokenizing and representing sentences as vectors, so that they can be input to a neural network. 
- Apply RNNs, GRUs, and LSTMs in TensorFlow. 
- Train an  LSTM on existing text to create original poetry!

## 01.Sentiment in text
- The first step in understanding sentiment in text, and in particular when training a neural network to do so is the tokenization of that text. 
- This is the process of converting the text into numeric values, with a number representing a word or a character. 
- Learn about the Tokenizer and pad_sequences APIs in TensorFlow and how they can be used to prepare and encode text and sentences to get them ready for training neural networks!

## 02.Word Embeddings
- Learn about Embeddings, where these tokens are mapped as vectors in a high dimension space. 
- With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space. 
- This will begin the process of training a neural network to understand sentiment in text 
	- Look at movie reviews, training a neural network on texts that are labelled 'positive' or 'negative' and determining which words in a sentence drive those meanings.
	
## 03.Sequence models
- Sentiment can be determined by the sequence in which words appear. 
	- For example, you could have 'not fun', which of course is the opposite of 'fun'. 
- Digging into a variety of model formats that are used in training models to understand context in sequence!

## 04.Sequence models and literature
- Given a body of words, you could conceivably predict the word most likely to follow a given word or phrase, and once you've done that, to do it again, and again. 
- Build a poetry generator. It's trained with the lyrics from traditional Irish songs, and can be used to produce beautiful-sounding verse of it's own!