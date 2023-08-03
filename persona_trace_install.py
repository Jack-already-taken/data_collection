from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

#save the model to a local folder
tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
tokenizer.save_pretrained("./persona")
model.save_pretrained("./persona")
