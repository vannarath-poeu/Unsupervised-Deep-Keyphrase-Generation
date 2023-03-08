from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json

if __name__ == "__main__":
  documents = []
  i = 0
  with open('data/train.json', 'r') as f:
    for line in f.readlines():
      data = json.loads(line)
      doc = " ".join(data["document"])
      documents.append(TaggedDocument(doc, [i]))
      i += 1
  model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
  model.save('doc2vec.bin')