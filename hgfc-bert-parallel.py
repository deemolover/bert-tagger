# from transformers import AutoTokenizer, AutoModel

# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # model = AutoModel.from_pretrained("bert-base-uncased")

# # inputs = tokenizer("Hello world!", return_tensors="pt")
# # outputs = model(**inputs)


# from transformers import pipeline
# unmasker = pipeline('fill-mask', model='bert-base-uncased')
# while True:
#     sentence = input()
#     # result = unmasker("The man worked as a [MASK].")
#     result = unmasker(sentence)
#     print(result)

import os

import json

from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

print("Global initialization started.")
WORK_DIR = "/data/disk5/private/yuc/coref/bert-tagger"
FILE_LIST = "filelist.txt"
WIKI_DIR = os.path.join(WORK_DIR, "../wikipedia/text")
# DUMP_DIR =  os.path.join(WORK_DIR, "playground/dump")
DUMP_DIR = os.path.join(WORK_DIR, "playground/dump_kl_para")
LOG_DIR = os.path.join(WORK_DIR, "playground/logs")

global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
global_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
print("Global initialization completed.")

# result: masked; target: unmasked index: golden
def index_only_dist(result, target, index):
    n_dim = 1
    v_dim = 2
    return torch.mean(
        torch.sum(
            F.relu(target - result) * index, dim=v_dim
        ),
        dim=n_dim
    )

def kl_divergence_dist(result, target, index):
    n_dim = 1
    v_dim = 2
    return torch.mean(
        torch.sum(
            F.softmax(target, dim=v_dim) * 
            ( - F.log_softmax(result, dim=v_dim) + F.log_softmax(target, dim=v_dim)),
            dim=v_dim
        ),
        dim=n_dim
    )

def cross_entropy_dist(result, target, index):
    n_dim = 1
    v_dim = 2
    return torch.mean(
        torch.sum(
            - F.softmax(target, dim=v_dim) * F.log_softmax(result, dim=v_dim),
            dim=v_dim
        ),
        dim=n_dim
    )

def default_transform(sentence):
    if sentence[0] == "<":
        return None
    # raw_tokens: parsed into subword but not yet converted to ids
    raw_tokens = global_tokenizer.tokenize(sentence.strip())
    if len(raw_tokens) <= 5 or 128 <= len(raw_tokens):
        return None
    # tokens converted to ids
    tokens = global_tokenizer(raw_tokens, return_tensors="pt", is_split_into_words=True)["input_ids"]
    tokens = torch.squeeze(tokens)

    # this is the format of a sentence.
    return {
        "tokens": tokens,
        "raw": raw_tokens
    }

# ts = default_transform("To be or not to be, this is the question.")

class SentenceIterable:
    def __init__(self,
        file_path_list=FILE_LIST,
        file_id=0,
        stc_id=0,
        transform=default_transform):
        self.file_id = file_id
        self.stc_id = stc_id
        with open(file_path_list, "r") as f_list:
            self.file_paths = f_list.read().split()
        if transform == None:
            self.transform = default_transform
        else:
            self.transform = transform
        print("SentenceIterable constructed.")
        
    def __iter__(self):
        return self.sentence_generator()
    
    def sentence_generator(self):
        file_count = len(self.file_paths)
        while self.file_id < file_count:
            file_path = self.file_paths[self.file_id]
            with open(file_path) as fs:
                sentences = fs.readlines()
                sentence_count = len(sentences)
                while self.stc_id < sentence_count:
                    sentence = sentences[self.stc_id]
                    sentence = self.transform(sentence)
                    if sentence == None:
                        print("sentence discarded.")
                    else:
                        yield (sentence, self.file_id, self.stc_id)
                    self.stc_id += 1
            self.stc_id = 0
            self.file_id += 1

class QuestionPairIterable(Dataset):
    def __init__(self, 
        sentence,
        mask_placeholder="[MASK]",
        miss_placeholder="[MASK]"):
        super(QuestionPairIterable).__init__()
        self.sentence = sentence["tokens"]
        self.miss_ph = miss_placeholder
        self.mask_ph = mask_placeholder
        self.miss_id = global_tokenizer.convert_tokens_to_ids(miss_placeholder)
        self.mask_id = global_tokenizer.convert_tokens_to_ids(mask_placeholder)
        length = len(self.sentence)
        self.index_pairs = [
            ([miss_index], [mask_index])
            for miss_index in range(1, length-1)
                for mask_index in range(1, length-1)
                    if miss_index != mask_index
        ]

        self.start = 0
        self.end = len(self.index_pairs)
        # print("QuestionPairIterable constructed.")

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, index):
        missing_indices, masked_indices = self.index_pairs[index]
        # unmasked_question = list(self.sentence)
        unmasked_question = self.sentence.clone()
        for missing_index in missing_indices:
            # unmasked_question[missing_index] = self.miss_ph
            unmasked_question[missing_index] = self.miss_id
        # masked_question = list(unmasked_question)
        masked_question = unmasked_question.clone()
        for masked_index in masked_indices:
            # masked_question[masked_index] = self.mask_ph
            masked_question[masked_index] = self.mask_id

        # this ia the format of a question pair.
        return {
            "label": self.sentence,
            "unmasked": unmasked_question, 
            "masked": masked_question, 
            "miss_id": torch.tensor(missing_indices), 
            "mask_id": torch.tensor(masked_indices)
        }

def get_batch_size(batch):
    for value in batch.values():
        return value.shape[0]

class QuestionPairConsumer:
    def __init__(self,
        tokenizer=global_tokenizer,
        model=global_model,
        measure=kl_divergence_dist):
        self.tokenizer = tokenizer
        self.model = model
        self.measure = measure
    
    # question_pair is batched question pairs
    def consume_question_pair(self, question_pair):
        # [B(atch), L(ength of sentence)]
        context = question_pair["label"]
        unmasked = question_pair["unmasked"]
        masked = question_pair["masked"]
        # [B(atch), n(umber of missing tokens)]
        missing_indices = question_pair["miss_id"]
        masked_indices = question_pair["mask_id"]
        # u_pred = consume_question(unmasked, context)
        # m_pred = consume_question(masked, unmasked)
        # [B(atch), L(ength of sentence), V(ocabulary size)]
        u_logits = self.model(input_ids=unmasked).logits
        m_logits = self.model(input_ids=masked).logits

        missing_label_ids = torch.gather(context, 1, missing_indices) # [B, n]
        answer_shape = list(missing_indices.shape)
        answer_shape.append(u_logits.shape[2])
        missing_indices = missing_indices.unsqueeze(2).expand(answer_shape) # [B, n, V]
        missing_label_ids = missing_label_ids.unsqueeze(2).expand(answer_shape) # [B, n, V]
        
        ones_template = torch.tensor([[[1.]]]).expand(answer_shape) # [B, n, V]
        # golden logits ,g_logits[b][n][index[b][n]] = 1
        g_logits = torch.scatter(torch.zeros(answer_shape), 2, missing_label_ids, ones_template)
        # unmasked logits
        u_logits = torch.gather(u_logits, 1, missing_indices)
        # masked logits
        m_logits = torch.gather(m_logits, 1, missing_indices)

        return self.measure(m_logits, u_logits, g_logits)     

class SaveManager:
    def __init__(self,
        dump_dir=DUMP_DIR,
        counter=0,
        log_interval=100,
        save_interval=500):
        self.sentence_dict = {}
        self.relation_list = []
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.counter = counter - counter % save_interval
        self.dump_dir = dump_dir
        self.progress_path = os.path.join(self.dump_dir, "progress.log")
        self.rel_template = os.path.join(dump_dir, "relation_list_cnt_{}.dump")
        self.stc_template = os.path.join(dump_dir, "sentence_dict_cnt_{}.dump")

    def load_progress(self):
        return (0, 0)

        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r") as p_log:
                progress = json.load(p_log)
                file_id = progress["file_id"]
                stc_id = progress["stc_id"]
                self.save_interval = progress["save_interval"]
                self.counter = progress["counter"]
                return (file_id, stc_id)
        return (0, 0)

    def dump_progress(self, file_id, stc_id):
        with open(self.progress_path, "w") as p_log:
            progress = {
                "file_id": file_id,
                "stc_id": stc_id,
                "counter": self.counter,
                "save_interval": self.save_interval
            }
            p_log.write(json.dumps(progress))
        
    def save_sentence_list(self):
        sentence_list = []
        for context_id, raw_tokens in self.sentence_dict.items():
            sentence_list.append({
            "id": context_id,
            "context": raw_tokens
        })
        sentence_list.sort(key=lambda x:x["id"])
        save_path = self.stc_template.format(self.counter)
        with open(save_path, "w") as f:
            for sentence in sentence_list:
                f.write(json.dumps(sentence)+"\n")

    def update_sentence(self, sentence, context_id):
        self.sentence_dict[context_id] = sentence["raw"]
    
    def save_relation_list(self):
        save_path = self.rel_template.format(self.counter)
        with open(save_path, "w") as f:
            for relation in self.relation_list:
                f.write(json.dumps(relation)+"\n")
    
    def update_relation(self, sample, distance, context_id):
        self.relation_list.append({
            "context": context_id,
            "missing_index": sample["miss_id"].tolist(),
            "masked_index": sample["mask_id"].tolist(),
            "distance": float(distance)
        })
        self.counter += 1
        if self.counter % self.log_interval == 0:
            print("Got example count: ", self.counter)
        if self.counter % self.save_interval == 0:
            print("Save examples.")
            self.save_relation_list()
            self.save_sentence_list()
            self.relation_list = []
            self.sentence_dict = {}

    def update_relation_batched(self, batch, distance, context_id):
        batch_size = get_batch_size(batch)
        for index in range(0, batch_size):
            relation = {}
            for key, batched_tensor in batch.items():
                relation[key] = batched_tensor[index]
            self.update_relation(relation, distance[index], context_id)

        
def main():
    sentence_list = []
    relation_list = []
    log_interval = 100
    save_interval = 500

    save_manager = SaveManager(log_interval=log_interval, save_interval=save_interval)
    last_file_id, last_stc_id = save_manager.load_progress()
   
    sentence_dataset = SentenceIterable(file_id=last_file_id,stc_id=last_stc_id)
    consumer = QuestionPairConsumer() 

    for sentence, file_id, stc_id in sentence_dataset:
        context_id = file_id * 50000 + stc_id
        question_pair_dataset = QuestionPairIterable(sentence)
        dataloader = DataLoader(question_pair_dataset, batch_size=32, num_workers=0)
        for sample_batched in dataloader:
            distance = consumer.consume_question_pair(sample_batched)
            save_manager.update_sentence(sentence, context_id)
            save_manager.update_relation_batched(sample_batched, distance, context_id)
            save_manager.dump_progress(file_id, stc_id)

if __name__ == "__main__":
    main()
    