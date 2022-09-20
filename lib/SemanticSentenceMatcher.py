import numpy as np
from numba import jit
import time
from transformers import AutoTokenizer, TFAutoModel, AutoModel
from sentence_transformers import util
import torch
import torch.nn.functional as F
import glob
import os
from copy import copy
import tensorflow as tf
from numba.typed import List
import logging
from sklearn.metrics.pairwise import cosine_similarity  # for similarity

tf.get_logger().setLevel(logging.ERROR)

semantic_models = {
    "custom": {
        "name": "custom-distilbert",
        "path": "model/distilbert-base-uncased",
        "tokenizer_method": "auto",
        "model_method": "TF",
        "modelWithTF": True
    },
    "all-mpnet": {
        "name": "all-mpnet-base-v2",
        "path": "model/all-mpnet-base-v2",
        "tokenizer_method": "auto",
        "model_method": "auto",
        "modelWithTF": False
    },
}


def current_milli_time():
    return round(time.time() * 1000)


@jit(nopython=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    # assert (len(u) == len(v))
    uv = 0
    uu = 0
    vv = 0
    for i in range(len(u)):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta


# Function to print the settings
def print_cuda_usage():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


class PresetLearningSequenceEmr(object):
    def __init__(self, line):
        if ";;" in line:
            items = line.split(";;")
            _ev_str_arr = items[0].split(",")
            _ev_float_arr = List()
            for _ev_str in _ev_str_arr:
                _ev_float_arr.append(float(_ev_str))
            self.embedding_vector = _ev_float_arr  # torch.cuda.FloatTensor(_ev_float_arr)
            self.attribute_name = items[1]
            self.training_pattern = items[2]
            self.sentence = items[3]
        else:
            self.embedding_vector = None
            self.attribute_name = None
            self.training_pattern = None
            self.sentence = None

    def __copy__(self):
        newone = type(self)()
        newone.attribute_name = self.attribute_name
        newone.training_pattern = self.training_pattern
        newone.sentence = self.sentence
        return newone

    def __hash__(self):
        return hash((self.sentence, self.attribute_name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.sentence == other.sentence and self.attribute_name == other.attribute_name

    def __str__(self):
        string = '%s;;%s;;%s' % (self.attribute_name, self.training_pattern, self.sentence)
        return string

    def getAsList(self):
        return [self.embedding_vector, self.attribute_name, self.training_pattern, self.sentence]


class TrainingSequence(object):
    def __init__(self, _sentence, _label="", _pattern=""):
        self.embedding_vector = None
        self.attribute_name = _label
        self.training_pattern = _pattern
        self.sentence = _sentence
        self.matching_score = 0.0

    def __copy__(self):
        newone = type(self)(self.sentence, self.attribute_name, self.training_pattern)
        newone.matching_score = self.matching_score
        return newone

    def __hash__(self):
        return hash((self.sentence, self.attribute_name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.sentence == other.sentence and self.attribute_name == other.attribute_name

    def __str__(self):
        string = '%s;;%s;;%s' % (self.attribute_name, self.training_pattern, self.sentence)
        return string

    def getAsList(self):
        return [self.embedding_vector, self.attribute_name, self.training_pattern, self.sentence]


class TestSequence(object):
    def __init__(self, _sentence, _label="", _pattern=""):
        self.embedding_vector = None
        self.labeled_attribute_name = _label
        self.computed_attribute_name = ""
        self.labeled_pattern_to_extract_items = _pattern
        self.computed_pattern_to_extract_items = ""
        self.sentence = str(_sentence).strip()
        if not self.sentence.startswith("[CLS]"):
            self.sentence = "[CLS] " + self.sentence
        if "?" in self.sentence and "[SEP]" not in self.sentence:
            self.sentence = self.sentence.replace("?", "? [SEP]")
        self.matching_score = 0.0

    def __copy__(self):
        newone = type(self)(self.sentence, self.labeled_attribute_name, self.labeled_pattern_to_extract_items)
        newone.computed_attribute_name = self.computed_attribute_name
        newone.computed_pattern_to_extract_items = self.computed_pattern_to_extract_items
        newone.matching_score = self.matching_score
        return newone

    def __hash__(self):
        return hash((self.sentence, self.labeled_attribute_name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.sentence == other.sentence and self.labeled_attribute_name == other.labeled_attribute_name

    def __str__(self):
        string = '%s;;%s;;%s;;%f;;%s;;%s' % (self.sentence, self.labeled_attribute_name, self.computed_attribute_name,
                                             self.matching_score, self.labeled_pattern_to_extract_items,
                                             self.computed_pattern_to_extract_items)
        return string


class LabeledSequence(object):
    def __init__(self, _sentence_left, _sentence_right, _label=0):
        self.sentence_left = _sentence_left
        self.sentence_right = _sentence_right
        self.label = _label

    def __copy__(self):
        newone = type(self)(self.sentence_left, self.sentence_right, self.label)
        return newone

    def __hash__(self):
        return hash((self.sentence_left, self.sentence_right, self.label))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.sentence_left == other.sentence_left and self.sentence_right == other.sentence_right

    def __str__(self):
        string = '%s;;%s;;%s' % (self.sentence_left, self.sentence_right, self.label)
        return string


class SemanticMatcher(object):
    def __init__(self, model_name=None):
        if model_name not in semantic_models:
            self.model_name = 'custom'
        else:
            self.model_name = model_name
        model_obj = semantic_models[self.model_name]
        self.modelWithTF = model_obj["modelWithTF"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_obj["tokenizer_method"] == "auto":
            self.tokenizer = AutoTokenizer.from_pretrained(model_obj["path"])
        if model_obj["model_method"] == "auto":
            self.model = AutoModel.from_pretrained(model_obj["path"], output_hidden_states=True)
        else:
            self.model = TFAutoModel.from_pretrained(model_obj["path"], output_hidden_states=True)

        self.trainedData = []

    # For all-mpnet-v2
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        if self.modelWithTF:
            return self.TF_mean_pooling(model_output, attention_mask)
        else:
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

    def getEmbeddingVector(self, sequence):
        if self.modelWithTF:
            return self.getTfEmbeddingVector(sequence)
        else:
            encoded_input = self.tokenizer(sequence, padding=True, truncation=True, return_tensors="pt", max_length=512)
            # encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            # print("outputs:", outputs)
            text_embedding = self.mean_pooling(outputs, encoded_input['attention_mask'])
            text_embedding = F.normalize(text_embedding, p=2, dim=1).detach().cpu().numpy()
            return text_embedding

    # For distilbert
    # TF Mean Pooling - Take attention mask into account for correct averaging
    def TF_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        # print(attention_mask[0])   # [1 1 ......0], shape=(512,), dtype=int32)
        attention_mask_in1d = tf.expand_dims(attention_mask, -1)
        # print(attention_mask_in1d)     # [[[1]...[0]]], shape=(1, 512, 1), dtype=int32)
        attention_mask_expanded = tf.broadcast_to(attention_mask_in1d, token_embeddings.get_shape())
        # print(attention_mask_expanded)  # shape=(1, 512, 768), dtype=int32)
        attention_mask_expanded_float = tf.cast(attention_mask_expanded, dtype=tf.float32)
        # print("attention_mask_expanded_float:"+str(attention_mask_expanded_float)) #[[[1. ... 1.][1. ... 1.]...]], shape=(1, 512, 768), dtype=float32)
        input_mask_expanded_sum = tf.reduce_sum(attention_mask_expanded_float, axis=1)
        # print("input_mask_expanded_sum:"+str(input_mask_expanded_sum)) # [[13. 13. 13. ... 13.]], shape=(1, 768), dtype=float32)
        input_mask_within_limits = tf.clip_by_value(input_mask_expanded_sum, clip_value_min=1e-9,
                                                    clip_value_max=input_mask_expanded_sum.dtype.max)
        # print(input_mask_within_limits) # tf.Tensor([[13. 13. .... 13. 13.]], shape=(1, 768), dtype=float32)
        reducedEmbeddings = tf.reduce_sum(token_embeddings * attention_mask_expanded_float, 1)
        # print(reducedEmbeddings) # tf.Tensor([[-3.72260046e+00 ..... -2.14804983e+00]], shape=(1, 768), dtype=float32)
        return reducedEmbeddings / input_mask_within_limits

    def getTfEmbeddingVector(self, sequence):
        encoded_input = self.tokenizer.encode_plus(sequence, padding='max_length', truncation=True, return_tensors="tf",
                                                   max_length=512)
        outputs = self.model(encoded_input)
        text_embedding = self.TF_mean_pooling(outputs, encoded_input['attention_mask'])
        # print("t1:"+str(text_embedding)) # tf.Tensor([[-2.86353886e-01  2.54664868e-01 ... -5.79358518e-01 -1.65234596e-01]], shape=(1, 768), dtype=float32)
        # Normalizes along dimension axis using an L2 norm.
        text_embedding = tf.math.l2_normalize(text_embedding, axis=1).numpy()
        # print("t2:"+str(text_embedding))  # [[-2.43002158e-02  2.16110609e-02 ... -4.91648205e-02 -1.40219377e-02]]
        return text_embedding

    def loadTrainingData(self, training_model_path):
        latest_file = max(glob.iglob(training_model_path), key=os.path.getmtime)
        print(latest_file)
        with open(latest_file) as trainedItems:
            for line in trainedItems:
                self.trainedData.append(PresetLearningSequenceEmr(line))

    # def match(self, test_embedding, test_sentence):
    #     if len(test_embedding) < 1:
    #         print("Incorrect test embedding")
    #     if len(self.trainedData) < 1:
    #         print("Preset Learning Sentences have not been loaded")
    #
    #     for ls in self.trainedData:
    #         # sim = util.pytorch_cos_sim(test_embedding, ls.embedding_vector)[0]
    #         # print(len(test_embedding), " -- ", len(ls.embedding_vector))
    #         sim = cosine_similarity_numba(test_embedding, ls.embedding_vector)
    #         if sim > test_sentence.matching_score:
    #             test_sentence.computed_attribute_name = ls.attribute_name
    #             test_sentence.computed_pattern_to_extract_items = ls.training_pattern
    #             test_sentence.matching_score = sim
    #         # exact mathcing, no need to check further
    #         if sim == 1.0:
    #             return test_sentence
    #     # return the best match
    #     return test_sentence

    def match(self, test_sentence: TestSequence, min_sim=0.0):
        if len(test_sentence.embedding_vector) < 1:
            print("Incorrect test embedding")
        if len(self.trainedData) < 1:
            print("Preset Learning Sentences have not been loaded")
        matching_list: List[TestSequence] = []
        for ls in self.trainedData:
            # sim = util.pytorch_cos_sim(test_embedding, ls.embedding_vector)[0]
            # print(len(test_embedding), " -- ", len(ls.embedding_vector))
            # sim = cosine_similarity_numba(test_sentence.embedding_vector, ls.embedding_vector)
            sim = cosine_similarity(test_sentence.embedding_vector, ls.embedding_vector)[0][0]
            if sim < min_sim:
                # print("sim:",sim, ", min_sim", min_sim)
                continue
            if sim >= test_sentence.matching_score:
                # remove all elements with similarity lower than the highest one
                if sim > test_sentence.matching_score:
                    matching_list = []
                test_sentence.computed_attribute_name = ls.attribute_name
                test_sentence.computed_pattern_to_extract_items = ls.training_pattern
                test_sentence.matching_score = sim
                matching_list.append(copy(test_sentence))
            # exact mathcing, no need to check further
            # if sim == 1.0:
            #     return test_sentence
        # return the best match
        return matching_list

    def match_all_trainingset(self, test_sentence: TestSequence, min_sim=0.0):
        if len(test_sentence.embedding_vector) < 1:
            print("Incorrect test embedding")
        if len(self.trainedData) < 1:
            print("Preset Learning Sentences have not been loaded")
        matching_list: List[TestSequence] = []

        for ls in self.trainedData:
            # sim = util.pytorch_cos_sim(test_embedding, ls.embedding_vector)[0]
            # print(len(test_embedding), " -- ", len(ls.embedding_vector))
            # sim = cosine_similarity_numba(test_sentence.embedding_vector, ls.embedding_vector)
            sim = cosine_similarity(test_sentence.embedding_vector, ls.embedding_vector)[0][0]
            # avoid unnecessary checks
            if sim < min_sim:
                # print("sim:",sim, ", min_sim", min_sim)
                continue
            test_sentence.computed_attribute_name = ls.attribute_name
            test_sentence.computed_pattern_to_extract_items = ls.training_pattern
            test_sentence.matching_score = sim

            matching_list.append(copy(test_sentence))

        # return the best match
        return matching_list

    def match_list(self, test_embedding, training_embedding_list):
        maxSim = 0.0
        maxTrainingEmbeddingItem = -1
        for training_embedding in training_embedding_list:
            # sim = cosine_similarity_numba(test_embedding, training_embedding)
            sim = cosine_similarity(test_embedding, training_embedding)[0][0]
            if sim > maxSim:
                maxTrainingEmbeddingItem = training_embedding
            if sim == 1.0:
                return sim
        # return the best match
        return maxSim
