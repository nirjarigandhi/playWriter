# Report 
**By Nirjari Gandhi & Vijay Nandlal**
---

## Introduction
Our task is to use the decoder layer of a transformer (modeled in the structure of GPT-2) to generate Sheakespeare sounding sentences using the plays *Romeo and Juliet* and *Hamlet*. Note that the entire transformer class was built from scratch using resources like https://jalammar.github.io/illustrated-transformer/,  https://nlp.seas.harvard.edu/2018/04/03/attention.html, and https://jalammar.github.io/illustrated-gpt2/. The inputs to our model is 50 word long sentences taken from the two plays modeled in the form of one-hot vectors. The inputs had structure (batch length, sentence/sequence length, embedding size). Similarly the outputs are also one-hot vectors of the same shape. The embedding size of the raw one hot vector was 10086.

## Model Figure
Here is the structure of our model ![Custom Decoding Layer](images/Neural_Net1.png)

We built this model entirely from scratch. After parsing the data into groups of 50 sentences with a vocabulary size (embedding size 10086) we used a trainable weight matrix to reduce this dimensionality from 10086 to 768. Then using the positional encoding method we learned from class involving the Sine and Cosine functions we created a matrix with positional encodings with dimensionality (50, 768) to sum with the newly transformed word embeddings.
