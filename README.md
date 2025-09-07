# charGPT

**charGPT** is a minimal character-level Transformer decoder written in under **200 lines of code**. It uses self-attention to model sequences and predict the next character.

I've made the input, code, model, and output public so you can experiment with it.

`train.py` helps you build a model on chunks of text data.  
`inference.py` lets you generate text from your trained model.

Since it's character-level, it focuses more on generating **words** correctly rather than full sentences. 
For sentence-level generation, I’ll release **wordGPT** soon. 

Till then, play around with this and feel free to reach out if you face any issues.

