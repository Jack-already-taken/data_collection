"""
This model is based on gtr-large-1
https://huggingface.co/sentence-transformers/gtr-t5-base
"""

import sys
from sentence_transformers import SentenceTransformer, util

def main(argv):

	if len(sys.argv) < 3:
		print("Error: Missing argument - test dataset")
		exit(0)
	
	inf_len = 100

	test_dir = sys.argv[2]
	fs = open(test_dir + "sentences.txt")
	sentence_list = []
	for i in range(inf_len):
		sentences = [next(fs).rstrip() for _ in range(100)]
		sentence_list.append(sentences)
	model = SentenceTransformer('sentence-transformers/gtr-t5-base')
	
	for batch in sentence_list:
		embeddings = model.encode(batch)
		cosine_scores = util.cos_sim(embeddings, embeddings)
		#print(cosine_scores)

if __name__ == "__main__":
	main(sys.argv[1:])
