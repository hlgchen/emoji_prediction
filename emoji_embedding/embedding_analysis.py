import torch.nn as nn
import torch


class EmbeddingAnalysis:
    def __init__(self, embedding, index_label, label_emoji):
        self.embedding = embedding
        self.index_label = index_label
        self.label_index = {k: v for v, k in index_label.items()}
        self.sim = nn.CosineSimilarity()
        self.label_emoji = label_emoji

    def get_embedding_vector(self, label):
        idx = self.label_index[label]
        return self.embedding[idx].unsqueeze(0), idx

    def similarity(self, label1, label2):
        v1, _ = self.get_embedding_vector(label1)
        v2, _ = self.get_embedding_vector(label2)

        return self.sim(v1, v2)

    def most_least_similar(self, v, idx, k, most=True):

        label_embedding = v.repeat(len(self.index_label.keys()), 1)

        similarities = self.sim(label_embedding, self.embedding)
        top_sim, top_idx = torch.topk(similarities, dim=0, k=k + 1, largest=most)
        top_labels = [self.index_label[i.item()] for i in top_idx]

        result = {self.index_label[idx] + self.label_emoji[self.index_label[idx]]: 1}
        result.update(
            {
                l + self.label_emoji[l]: s
                for i, (l, s) in enumerate(zip(top_labels, top_sim.tolist()))
                if i > 0
            }
        )

        return result

    def least_similar(self, label, k=10):
        v, idx = self.get_embedding_vector(label)
        return self.most_least_similar(v, idx, k, False)

    def most_similar(self, label, k=10):
        v, idx = self.get_embedding_vector(label)
        return self.most_least_similar(v, idx, k, True)

    def relation(self, label, relation, k=10):
        v, idx = self.get_embedding_vector(label)
        rv1, _ = self.get_embedding_vector(relation[0])
        rv2, _ = self.get_embedding_vector(relation[1])
        r = rv2 - rv1

        v_target = v + r

        return self.most_least_similar(v_target, idx, k, most=True)
