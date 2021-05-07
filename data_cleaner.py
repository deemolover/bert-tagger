import re
import os
import pickle
import json
import random

import numpy as np

WIKI_DIR = "/data/disk5/private/yuc/coref/wikipedia/text"
DUMP_DIR = "/data/disk5/private/yuc/coref/bert-tagger/playground/dump_kl"
CLEANED_DUMP_DIR = "/data/disk5/private/yuc/coref/bert-tagger/playground/dump_kl_cleaned"

def get_counter_max():
    counter = -1
    for root, dirs, files in os.walk(DUMP_DIR):
        for file_name in files:
            match = re.search('([0-9]+).dump', file_name)
            if match:
                cnt = match.group(1)
                cnt = int(cnt)
                if cnt > counter:
                    counter = cnt
    return counter


def clean_rel_and_stc(counters):

    def clean_relation(relation):
        relation["distance"] = float(relation["distance"].detach())
        return relation

    def split_into_intervals(nums):
        numbers = list(nums)
        numbers.sort()
        result = []
        n_min = numbers[0]
        n_max = n_min
        for number in numbers[1:]:
            if number == n_max + 1:
                n_max = number
            else:
                result.append((n_min, n_max + 1))
                n_min = number
                n_max = n_min
        result.append((n_min, n_max + 1))
        return result

    def get_sublist_of_range(stc_dict, file_id, key_range):
        sublist = []
        for stc_id in range(key_range[0], key_range[1]):
            sublist.append({
                "id": stc_id + file_id * 50000,
                "context": stc_dict[stc_id]
            })
        return sublist

    file_id = -1
    for counter in counters:
        if counter % 10000 == 0:
            print(counter)
        rel_template = "relation_list_cnt_{}.dump"
        stc_template = "sentence_dict_cnt_{}.dump"
        get_path_fn = lambda x, y: os.path.join(x, y.format(counter))
        
        rel_path = get_path_fn(DUMP_DIR, rel_template)
        new_rel_path = get_path_fn(CLEANED_DUMP_DIR, rel_template)
        with open(rel_path, "rb") as rel_file, open(new_rel_path, "w") as new_rel_file:
            part_rel_list = pickle.load(rel_file)
            new_rel_list = [ clean_relation(x) for x in part_rel_list ]
            new_rel_list = [ json.dumps(x)+"\n" for x in new_rel_list ]
            new_rel_file.writelines(new_rel_list)

        stc_path = get_path_fn(DUMP_DIR, stc_template)
        new_stc_path = get_path_fn(CLEANED_DUMP_DIR, stc_template)
        with open(stc_path, "rb") as stc_file, open(new_stc_path, "w") as new_stc_file:
            part_stc_dict = pickle.load(stc_file)
            intervals = split_into_intervals(part_stc_dict.keys())
            new_interval = None
            old_interval = None
            if len(intervals) > 2:
                print("ERROR: found dict with too many intervals")
            elif len(intervals) == 2:
                new_interval, old_interval = intervals[0], intervals[1]
                if new_interval[0] != 0:
                    print("ERROR: found dict at bound but with nonzero start")
            else:
                if intervals[0][0] == 0:
                    new_interval = intervals[0]
                else:
                    old_interval = intervals[0]
            new_stc_list = []
            if old_interval:
                assert(file_id >= 0)
                new_stc_list += get_sublist_of_range(part_stc_dict, file_id, old_interval)
            if new_interval:
                file_id += 1
                assert(file_id >= 0)
                new_stc_list += get_sublist_of_range(part_stc_dict, file_id, new_interval)
            if len(new_stc_list) != len(part_stc_dict):
                print("ERROR: sentence dict update failed with unaligned length")
            new_stc_list = [ json.dumps(x)+"\n" for x in new_stc_list ]
            new_stc_file.writelines(new_stc_list)
            
    
def main():
    counter_max = get_counter_max()
    clean_rel_and_stc(list(range(500, counter_max+1, 500)))

main()