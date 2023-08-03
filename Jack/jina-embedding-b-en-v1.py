import sklearn
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ['how is the weather today', 'What is the current weather like today?']

model = SentenceTransformer('jinaai/jina-embedding-b-en-v1')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))

