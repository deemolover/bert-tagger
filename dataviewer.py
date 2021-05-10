
import re
import os
import pickle
import json
import random

import torch
import numpy as np

WORK_DIR = "/data/disk5/private/yuc/coref/bert-tagger"
FILE_LIST = "filelist.txt"
WIKI_DIR = os.path.join(WORK_DIR, "../wikipedia/text")
# DUMP_DIR =  os.path.join(WORK_DIR, "playground/dump")
DUMP_DIR = os.path.join(WORK_DIR, "playground/dump_kl_para")
LOG_DIR = os.path.join(WORK_DIR, "playground/logs")
CLEANED_DUMP_DIR = "/data/disk5/private/yuc/coref/bert-tagger/playground/dump_kl_cleaned"

def get_dump_file_list():
    counter_list = []
    file_list = []
    rel_file_list = []
    stc_file_list = []
    def get_counter(file_name):
        cnt_pattern = re.compile("cnt_([0-9]+).dump$")
        result = cnt_pattern.match(file_name)
        if result:
            return eval(result.group(1))
        else:
            return 0
    for root, dirs, files in os.walk(DUMP_DIR):
        for file_name in files:
            # rel_name_pattern = re.compile("^relation_list_cnt_([0-9]+).dump$")
            # stc_name_pattern = re.compile("^sentence_dict_cnt_([0-9]+).dump$")
            counter = get_counter(file_name)
            if counter != 0:
                counter_list.append(counter)
            file_list.append((root, file_name))
    
    counter_list.sort()
    file_list.sort(key=lambda x: get_counter(x[1]))

def load_from_pickle(counters, dump_dir):
    rel_list = []
    stc_dict = {}
    for counter in counters:
        rel_path = os.path.join(dump_dir, "relation_list_cnt_{}.dump".format(counter))
        stc_path = os.path.join(dump_dir, "sentence_dict_cnt_{}.dump".format(counter))
        with open(rel_path, "rb") as rel_file, open(stc_path, "rb") as stc_file:
            part_rel_list = pickle.load(rel_file)
            part_stc_dict = pickle.load(stc_file)
            rel_list += part_rel_list
            stc_dict.update(part_stc_dict)
    return rel_list, stc_dict

def load_from_json(counters, dump_dir):
    rel_list = []
    stc_dict = {}
    for counter in counters:
        rel_path = os.path.join(dump_dir, "relation_list_cnt_{}.dump".format(counter))
        stc_path = os.path.join(dump_dir, "sentence_dict_cnt_{}.dump".format(counter))
        with open(rel_path, "r") as rel_file, open(stc_path, "r") as stc_file:
            part_rel_list = [ json.loads(line.strip()) for line in rel_file ] 
            part_stc_list = [ json.loads(line.strip()) for line in stc_file ]
            rel_list += part_rel_list
            stc_dict.update({ stc["id"]:stc["context"] for stc in part_stc_list })
    return rel_list, stc_dict

def get_rel_and_stc(counters, dump_dir=DUMP_DIR):
    
    rel_list, stc_dict = load_from_json(counters, dump_dir)

    def sort_relations(relations):
        return relations.sort(key=lambda x:x["distance"], reverse=True)
    def human_readable_relation(relation):
        context = stc_dict[relation["context"]]
        # x-1 for the added special tokens at start
        try:
            missing_token = [ context[x-1] for x in relation["missing_index"] ]
            masked_token = [ context[x-1] for x in relation["masked_index"] ]
        except IndexError:
            print(context, relation)
            exit()
        distance = relation["distance"]
        
        if type(distance) == torch.Tensor:
            np_distance = distance.detach().numpy()
        elif type(distance) == float:
            np_distance = np.array(distance)
        elif type(distance) == np.ndarray:
            np_distance = distance
        distance = float(distance)
        relation.update({
            "missing_token": missing_token,
            "masked_token": masked_token,
            "distance": distance,
            "distance_np": np_distance
        })
        return relation
    def str_relation(h_relation):
        return "context: {0}  missing token: {1}({4})  masked token: {2}({5})  distance: {3}".format(
            h_relation["context"], h_relation["missing_token"], h_relation["masked_token"], 
            h_relation["distance_np"],h_relation["missing_index"], h_relation["masked_index"]
        )
    def pretty_print(relations):
        for relation in relations:
            print(str_relation(relation))
    def analyze_relations(relations, stat_only=False, do_sort=True):
        print("relation count: ", len(relations))
        for rel_id, relation in enumerate(relations):
            human_readable_relation(relation)
            
        if do_sort:
            sort_relations(relations)
        print("top 5:")
        pretty_print(relations[:7])
        # print("last 5:")
        # pretty_print(relations[-5:])
        if not stat_only:
            missing_order = [ x["missing_index"] for x in relations]
            # print(missing_order)
        # distances = np.stack([ x["distance_np"] for x in relations ])
        distances = [ x["distance_np"] for x in relations ]
        average_dist = np.mean(distances)
        print("Average distance: ", average_dist)

    # random.shuffle(rel_list)
    analyze_relations(rel_list, stat_only=True, do_sort=False)
    
    def analyze_context(stc_id):
        # print(list(enumerate(stc_dict[stc_id])))
        print(list(enumerate(stc_dict[stc_id])))
        context_filter = lambda x: x["context"] == stc_id
        filtered_rel_list = list(filter(context_filter, rel_list))    
        analyze_relations(filtered_rel_list)


    # analyze_context(110)
    # analyze_context(225)
    # analyze_context(226)
    analyze_context(397)
    analyze_context(421)
    analyze_context(422)
    
get_rel_and_stc(list(range(500, 10000, 500)),
                dump_dir=DUMP_DIR)

        


