import collections
import pickle
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.random.set_random_seed(22)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f , encoding="latin1")


class WordEmbeddings(object):
    """Class for loading/using pretrained GloVe embeddings"""
    def __init__(self):
        self.pretrained_embeddings = load_pickle("./glove/embeddings.pkl")
        self.vocab = load_pickle("./glove/vocab.pkl")
    def tokid(self, w):
        return self.vocab.get(w, 0)


N_DISTANCE_FEATURES = 8
def make_distance_features(seq_len):
    """Constructs distance features for a sentence."""
    # how much ahead/behind the other word is
    distances = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i < j:
                distances[i, j] = (j - i) / float(seq_len)
    feature_matrices = [distances, distances.T]

    # indicator features on if other word is up to 2 words ahead/behind
    for k in range(3):
        for direction in ([1] if k == 0 else [-1, 1]):
            feature_matrices.append(np.eye(seq_len, k=k*direction))
    features = np.stack(feature_matrices)

    # additional indicator feature for ROOT
    features = np.concatenate(
        [np.zeros([N_DISTANCE_FEATURES - 1, seq_len, 1]),
             features], -1)
    root = np.zeros((1, seq_len, seq_len + 1))
    root[:, :, 0] = 1

    return np.concatenate([features, root], 0)



def attn_linear_combo():
    return Probe()


def attn_and_words():
    return Probe(use_words=True)  


def words_and_distances():
    return Probe(use_distance_features=True, use_attns=False,
               use_words=True, hidden_layer=True)


class Probe(object):
    """The probing classifier used in Section 5."""
    def __init__(self, use_distance_features=False, use_words=False,
               use_attns=True, include_transpose=True, hidden_layer=False):
        self._embeddings = WordEmbeddings()

        # We use a simple model with batch size 1
        self._attns = tf.placeholder(shape=[12, 12, None, None], dtype=tf.float32)
        self._labels = tf.placeholder(shape=[None], dtype=tf.int32)
        self._features = tf.placeholder(shape=[N_DISTANCE_FEATURES, None, None], dtype=tf.float32)
        self._words = tf.placeholder(shape=[None], dtype=tf.int32)

        if use_attns:
            seq_len = tf.shape(self._attns)[-1]
            if include_transpose:
                # Include both directions of attention
                attn_maps = tf.concat([self._attns, tf.transpose(self._attns, [0, 1, 3, 2])], 0)
                attn_maps = tf.reshape(attn_maps, [288, seq_len, seq_len])
            else:
                attn_maps = tf.reshape(self._attns, [144, seq_len, seq_len])
            # Use attention to start/end tokens to get score for ROOT
            root_features = (
                (tf.get_variable("ROOT_start", shape=[]) * attn_maps[:, 1:-1, 0]) +
                (tf.get_variable("ROOT_end", shape=[]) * attn_maps[:, 1:-1, -1])
            )
            attn_maps = tf.concat([tf.expand_dims(root_features, -1),
                                   attn_maps[:, 1:-1, 1:-1]], -1)
        else:
            # Dummy attention map for models not using attention inputs
            n_words = tf.shape(self._words)[0]
            attn_maps = tf.zeros((1, n_words, n_words + 1))

        if use_distance_features:
            attn_maps = tf.concat([attn_maps, self._features], 0)

        if use_words:
            word_embedding_matrix = tf.get_variable(
                "word_embedding_matrix",
                initializer=self._embeddings.pretrained_embeddings,trainable=False)
            word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix, self._words)
            n_words = tf.shape(self._words)[0]
            tiled_vertical = tf.tile(tf.expand_dims(word_embeddings, 0),[n_words, 1, 1])
            tiled_horizontal = tf.tile(tf.expand_dims(word_embeddings, 1), [1, n_words, 1])

            word_reprs = tf.concat([tiled_horizontal, tiled_vertical], -1)
            word_reprs = tf.concat([word_reprs, tf.zeros((n_words, 1, 200))], 1) # dummy for ROOT
            if not use_attns:
                attn_maps = tf.concat([
                    attn_maps, tf.transpose(word_reprs, [2, 0, 1])], 0)

        attn_maps = tf.transpose(attn_maps, [1, 2, 0])
        if use_words and use_attns:
            # attention-and-words probe
            weights = tf.layers.dense(word_reprs, attn_maps.shape[-1])
            self._logits = tf.reduce_sum(weights * attn_maps, axis=-1)
        else:
            if hidden_layer:
                # 1-hidden-layer MLP for words-and-distances baseline
                attn_maps = tf.layers.dense(attn_maps, 256,activation=tf.nn.tanh)
                self._logits = tf.squeeze(tf.layers.dense(attn_maps, 1), -1)
            else:
                # linear combination of attention heads
                attn_map_weights = tf.get_variable("attn_map_weights",shape=[attn_maps.shape[-1]])
                self._logits = tf.reduce_sum(attn_map_weights * attn_maps, axis=-1)


        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels))
        opt = tf.train.AdamOptimizer(learning_rate=0.02)
        self._train_op = opt.minimize(loss)


    def _create_feed_dict(self, example):
        return {
            self._attns: example["attns"],
            self._labels: example["heads"],
            self._features: make_distance_features(len(example["words"])),
            self._words: [self._embeddings.tokid(w) for w in example["words"]]
        }

    def train(self, sess, example):
        return sess.run(self._train_op, feed_dict=self._create_feed_dict(example))

    def test(self, sess, example):
        return sess.run(self._logits, feed_dict=self._create_feed_dict(example))


def run_training(probe, train_data):
  """Trains and evaluates the given attention probe."""
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for epoch in range(1):
          print(40 * "=")
          print("EPOCH", (epoch + 1))
          print(40 * "=")
          print("Training...")
          for i, example in enumerate(train_data):
              if i % 2000 == 0:
                  print("{:}/{:}".format(i, len(train_data)))
              probe.train(sess, example)

          print("Evaluating...")
          correct, total = 0, 0
          for i, example in enumerate(dev_data):
              if i % 1000 == 0:
                  print("{:}/{:}".format(i, len(dev_data)))
              logits = probe.test(sess, example)
              for i, (head, prediction, reln) in enumerate(
                  zip(example["heads"], logits.argmax(-1), example["relns"])):
                  # it is standard to ignore punct for Stanford Dependency evaluation
                  if reln != "WP":
                      if head == prediction:
                          correct += 1
                      total += 1
          print("UAS: {:.3f}".format(100 * correct / total))



N= 5
train_data = [] 
dev_data = []
tf.compat.v1.reset_default_graph()
data = load_pickle("HIT_all.pkl")


i=4
print(i)
for j in range(N):
    n1 = int(j/5*len(data))
    n2 = int((j+1)/5*len(data))
    if(j!=i):
        train_data += data[n1:n2]
    else:
        dev_data += data[n1:n2]
run_training(attn_and_words(), train_data)




