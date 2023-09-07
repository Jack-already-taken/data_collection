"""
This model is based on distilbert-base-uncased
https://huggingface.co/ActivationAI/distilbert-base-uncased-finetuned-emotion
"""
# Use a pipeline as a high-level helper
import sys
import sklearn
import torch
from transformers import pipeline

def main(argv):

	if len(sys.argv) < 3:
		print("Error: Missing argument - test dataset")
		exit(0)

	test_dir = sys.argv[2]
	fs = open(test_dir + "sentences.txt")
	sentences = [next(fs).rstrip() for _ in range(2000)]
	pipe = pipeline("text-classification", model="ActivationAI/distilbert-base-uncased-finetuned-emotion", device = 0 if torch.cuda.is_available() else -1, truncation = True)
	
	for sentence in sentences:
		result = pipe(sentence)
		#print(sentence + "\n" + str(result))

if __name__ == "__main__":
	main(sys.argv[1:])
