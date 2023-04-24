# Report 
**By Nirjari Gandhi & Vijay Nandlal**
---

## Introduction
Our task is to use the decoder layer of a transformer (modeled in the structure of GPT-2) to generate Sheakespeare sounding sentences using the plays *Romeo and Juliet* and *Hamlet*. Note that the entire transformer class was built from scratch using resources like https://jalammar.github.io/illustrated-transformer/. The inputs to our model is 50 word long sentences taken from the two plays modeled in the form of one-hot vectors. The inputs had structure (batch length, sentence/sequence length, embedding size). Similarly the outputs are also one-hot vectors of the same shape.
