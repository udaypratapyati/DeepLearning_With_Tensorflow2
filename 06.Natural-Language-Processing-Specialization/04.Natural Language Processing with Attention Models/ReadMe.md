## Natural Language Processing with Sequence Models
- Translate complete English sentences into German using an encoder-decoder attention model.
- Build a Transformer model to summarize text. 
- Use T5 and BERT models to perform question-answering.
- Build a chatbot using a Reformer model.

### Week 1
Discover some of the shortcomings of a traditional seq2seq model and how to solve for them by adding an attention mechanism.
Build a Neural Machine Translation model with Attention that translates English sentences into German.
- Explain how an Encoder/Decoder model works
- Apply word alignment for machine translation
- Train a Neural Machine Translation model with Attention
- Develop intuition for how teacher forcing helps a translation model checks its predictions
- Use BLEU score and ROUGE score to evaluate machine-generated text quality
- Describe several decoding methods including MBR and Beam search

### Week 2
Compare RNNs and other sequential models to the more modern Transformer architecture.
Create a tool that generates text summaries.
- Describe the three basic types of attention
- Name the two types of layers in a Transformer
- Define three main matrices in attention
- Interpret the math behind scaled dot product attention, causal attention, and multi-head attention
- Use articles and their summaries to create input features for training a text summarizer
- Build a Transformer decoder model (GPT-2)