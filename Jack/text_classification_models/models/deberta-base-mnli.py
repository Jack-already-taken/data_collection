"""
This model is based on DeBERTa
https://huggingface.co/microsoft/deberta-base-mnli?text=I+love+how+terrible+you+are
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
	sentences = [line.rstrip() for line in fs]
	pipe = pipeline("text-classification", model="microsoft/deberta-base-mnli", device = 0 if torch.cuda.is_available() else -1)
	
	for i in range(5):
		for sentence in sentences:
			result = pipe(sentence)
			#print(sentence + "\n" + str(result))

if __name__ == "__main__":
	main(sys.argv[1:])
