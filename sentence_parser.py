
import os
import sys
import argparse
import time
import collections
import json

import nltk
from nltk.tree import Tree
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import spacy

# from transformers import BertTokenizer

print("Global initialization started.")
work_dir_dict = {
    "234-2": "/data/disk5/private/yuc/coref/bert-tagger",
    "cluster": "/home/shiyukai/project/yuc/coref/bert-tagger"
}
server_list = ["234-2", "cluster"]
parser = argparse.ArgumentParser(description='Arguments for data processing.')
parser.add_argument('location', choices=server_list,
                    help='Indicate the server this script is running on.')
parser.add_argument('--log', dest='log_interval', type=int, default=1000000,
                    help='Set log interval for the process.')
parser.add_argument('--save', dest='save_interval', type=int, default=5000,
                    help='Set save interval for the process.')
parser.add_argument('--batch', dest='batch_size', type=int, default=2000,
                    help='Batch size used by SpaCy models.')
args = parser.parse_args()
loc = args.location
if loc in work_dir_dict.keys():
    WORK_DIR = work_dir_dict[loc]
else:
    print("Input: ", loc, "  Valid keys: ", " ".join(work_dir_dict.keys()))
    exit()
default_log_interval = args.log_interval
default_save_interval = args.save_interval
default_batch_size = args.batch_size
print("log interval: ", default_log_interval, "  save interval: ", default_save_interval)

FILE_LIST = os.path.join(WORK_DIR, "playground/filelist.txt")
WIKI_DIR = os.path.join(WORK_DIR, "../wikipedia/text")
DUMP_DIR = os.path.join(WORK_DIR, "../wikipedia/parsed-text")
# global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
nlp = spacy.load("en_core_web_sm")
print("Global initialization completed.")

def default_transform(sentence):
    sentence = sentence.strip()
    if len(sentence) == 0: # empty
        return None
    if sentence[0] == "<":
        return None
    sent = sentence.split()
    # sent = word_tokenize(sentence)
    if len(sent) < 8 or 256 <= len(sent):
        return None
    # sent = pos_tag(sent)
    return sentence

class SentenceIterable:
    def __init__(self,
        path_to_file_list=FILE_LIST,
        file_id=0,
        sent_id=0,
        transform=default_transform):
        self.file_id = file_id
        self.sent_id = sent_id
        self.path_to_file_list = path_to_file_list
        if transform == None:
            self.transform = default_transform
        else:
            self.transform = transform
        print("SentenceIterable constructed.")
        
    def __iter__(self):
        return self.sentence_generator()

    def sentence_generator(self):
        # yield from self._sentence_generator()

        for doc, context in nlp.pipe(self.spacy_style_generator(), 
                batch_size=default_batch_size, as_tuples=True):
            sentence = { "sentence": doc, }
            sentence.update(context)
            yield sentence

    def spacy_style_generator(self):
        for sentence in self._sentence_generator():
            yield (
                sentence["sentence"],
                {
                    "file_id": sentence["file_id"],
                    "sent_id": sentence["sent_id"]
                }
            )

    def _sentence_generator(self):
        with open(self.path_to_file_list, "r") as f_list:
            for file_id, file_path in enumerate(f_list):
                if file_id < self.file_id:
                    continue
                file_path = os.path.join(WIKI_DIR, file_path)
                with open(file_path.strip()) as fs:
                    for sent_id, sentence in enumerate(fs):
                        if file_id == self.file_id and sent_id <= self.sent_id:
                            continue
                        sentence = self.transform(sentence)
                        if sentence != None:
                            yield {
                                "sentence": sentence, 
                                "file_id": file_id, 
                                "sent_id": sent_id
                            }

class TreeParserBase:
    def __init__(self):
        pass

    def dfs(self, tree):
        self.before_dfs()
        state_stack = []
        # node is a nltk tree        
        state_stack.append(self.create_state(tree))
        while len(state_stack) != 0:
            curr_state = state_stack[-1]
            node = curr_state["node"]
            index = curr_state["index"]
            if len(node) <= index: # end of node
                self.finish_state(state_stack.pop())
                continue
            
            if type(node[index]) == Tree:
                state_stack.append(self.create_state(node[index]))
            elif type(node[index]) == tuple:
                self.parse_tuple(node[index])
            curr_state["index"] += 1
        self.after_dfs()

    def create_state(self, node):
        state = {
            "node": node,
            "index": 0
        }
        self.init_state(state)
        return state

    def before_dfs(self):
        raise NotImplementedError

    def init_state(self, state):
        raise NotImplementedError

    def parse_tuple(self, node):
        raise NotImplementedError

    def finish_state(self, state):
        raise NotImplementedError
    
    def after_dfs(self):
        raise NotImplementedError
    
class NPTreeParser(TreeParserBase):
    def __init__(self):
        self.word_list = []
        self.np_list = []
    
    def before_dfs(self):
        self.word_list = []
        self.np_list = []

    def init_state(self, state):
        node = state["node"]
        state["is_np"] = node.label() == "NP"
        state["start"] = len(self.word_list)
        
    def parse_tuple(self, node):
        self.word_list.append({
            "token": node[0],
            "type": node[1]
        })

    def finish_state(self, state):
        state["end"] = len(self.word_list)
        if state["is_np"]:
            self.np_list.append((state["start"], state["end"]))
    
    def after_dfs(self):
        pass
    
class SentenceParser:
    def parse(self, sentence):
        raise NotImplementedError

class NLTKSentenceParser(SentenceParser):
    def __init__(self):
        self.tree_parser = NPTreeParser()
        self.grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN|NNS>}   # chunk determiner/possessive, adjectives and noun
            {<NNP>+}                # chunk sequences of proper nouns
        """
        self.chunk_parser = nltk.RegexpParser(self.grammar)

    def parse(self, sentence):
        tree = self.chunk_parser.parse(sentence["sentence"])
        self.tree_parser.dfs(tree)
        sentence.update({
            "np_list": self.tree_parser.np_list
        })

class SpacySentenceParser(SentenceParser):
    def parse(self, sentence):
        doc = sentence["sentence"]
        sent = [ token.text for token in doc ]
        nouns = [ (chunk.start, chunk.end) for chunk in doc.noun_chunks ]
        sentence.update({
            "sentence": sent,
            "np_list": nouns
        })

class SaveManager:
    def __init__(self,
        dump_dir=DUMP_DIR,
        counter=0,
        save_interval=500):
        self.save_interval = save_interval
        self.counter = counter
        self.dump_dir = dump_dir
        self.sentence_list = []
        self.progress_path = os.path.join(self.dump_dir, "progress.log")
        self.stc_template = os.path.join(dump_dir, "text_cnt_{:08d}.dump")

    def load_progress(self):
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r") as p_log:
                progress = json.load(p_log)
                file_id = progress["file_id"]
                sent_id = progress["sent_id"]
                self.counter = progress["counter"]
                self.save_interval = progress["save_interval"]
                return (file_id, sent_id)
        return (0, 0)

    def dump_progress(self, file_id, sent_id):
        with open(self.progress_path, "w") as p_log:
            progress = {
                "file_id": file_id,
                "sent_id": sent_id,
                "counter": self.counter,
                "save_interval": self.save_interval
            }
            p_log.write(json.dumps(progress))
    
    def update_sentence(self, sentence):
        self.sentence_list.append(sentence)
        self.counter += 1
        if self.counter % self.save_interval == 0:
            save_path = self.stc_template.format(self.counter)
            self.save_sentence_list(save_path)
            self.dump_progress(sentence["file_id"], sentence["sent_id"])
            self.sentence_list = []

    def save_sentence_list(self, save_path):
        with open(save_path, "w") as f:
            for sentence in self.sentence_list:
                f.write(json.dumps(sentence)+"\n")
    
    def load_sentence_list(self, load_path):
        sentence_list = []
        with open(load_path, "r") as f:
            for line in f:
                sentence_list.append(json.loads(line))
        return sentence_list    
    
class StopWatch:
    def __init__(self):
        self.ticks = []

    def start(self):
        self.ticks.append(time.time())
    
    def tick(self):
        self.ticks.append(time.time())
        return self.ticks[-1] - self.ticks[-2] 

    def total_elapsed(self):
        return self.ticks[-1] - self.ticks[0]

    def clear(self):
        self.ticks.clear()

        
def data_analysis(len_count, counter, bin_width):
    sum = 0
    max_key = 0
    for key, value in len_count.items(): 
        sum += (key * bin_width + bin_width / 2) * value
        if key > max_key:
            max_key = key
    average = sum / counter
    sum = 0
    for key in range(max_key + 1):
        sum += len_count[key]
        print("bin: ({0}, {1})  count: {2} percentage: {3:.2f}%".format(
            key*bin_width, key*bin_width+bin_width, len_count[key], sum / counter * 100
        ))
        if sum / counter > 0.997:
            break
    print("average length: ", average)
    time_stamp = (int(time.time()) // 3600) % 1000000
    with open(os.path.join(DUMP_DIR, "len_count_{:06d}.json".format(time_stamp)), "w") as f:
        json.dump(len_count, f)


def main():
    log_interval = default_log_interval
    save_interval = default_save_interval
    parser = SpacySentenceParser()
    save_manager = SaveManager(save_interval=save_interval)
    # file_id, sent_id = save_manager.load_progress()
    file_id, sent_id = 0, 0
    dataset = SentenceIterable(file_id=file_id, sent_id=sent_id)

    counter = 0
    bin_width = 8
    len_count = collections.defaultdict(int)
    watch = StopWatch()
    watch.start()
    for example in dataset:
        sentence = example["sentence"]
        len_count[len(sentence) // bin_width] += 1
        counter += 1
        if log_interval > 0 and counter % log_interval == 0:
            interval = watch.tick()
            total = watch.total_elapsed()
            print("sentence count: {0}  current speed: {1:.4f} sent/s  speed on average: {2:.4f} sent/s".format(
                    counter, log_interval / interval, counter / total
            ))
            break
        parser.parse(example)
        if save_interval > 0:
            save_manager.update_sentence(example)
        

    watch.tick()
    total = watch.total_elapsed()
    print("sentence count: {0} total time cost: {1:.4f} s,  average: {2:.4f} sent/s".format(
         counter, total, counter / total
    ))

    data_analysis(len_count, counter, bin_width)
    

if __name__ == "__main__":
    main()