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

import pickle
import json

from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F


WIKI_DIR = "/data/disk5/private/yuc/coref/wikipedia/text"
DUMP_DIR = "/data/disk5/private/yuc/coref/bert-tagger/playground/dump_kl"

global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
global_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def get_indices_of_id(tokens, token_id):
    return tokens.eq(token_id).nonzero(as_tuple=True)[0]

def get_logits_of_index(outputs, index, tokenizer=global_tokenizer):
    logits = outputs.logits
    logits = torch.squeeze(logits)[index]
    return logits

def get_prediction_from_probs(probs, tokenizer=global_tokenizer):
    print(probs.shape)
    pred_id = torch.argmax(probs)
    pred_prob = probs[pred_id]
    pred_token = tokenizer.convert_ids_to_tokens([pred_id,])[0]
    return (pred_id, pred_prob, pred_token)

def get_prediction_from_logits(logits, tokenizer=global_tokenizer):
    print(logits.shape)
    probs = F.softmax(torch.squeeze(logits), dim=0)
    return get_prediction_from_probs(probs, tokenizer) 

def index_only_dist(result, target, index):
    return F.relu(target[index] - result[index])

def kl_divergence_dist(result, target, index):
    return torch.mean(torch.sum(F.softmax(target, dim=0) * ( - F.log_softmax(result, dim=0) + F.log_softmax(target, dim=0))))

def cross_entropy_dist(result, target, index):
    return torch.mean(torch.sum(- F.softmax(target, dim=0) * F.log_softmax(result, dim=0)))

def wiki_text_file_iterator():
    for root, dirs, files in os.walk(WIKI_DIR):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            yield file_path   

def sentence_iterator(file_path):
    print(file_path)
    with open(file_path) as fs:
        for line in fs.readlines():
            if line[0] == '<':
                continue
            tokens = line.strip().split()
            if len(tokens) >= 30: # ignore doc that is too long
                continue
            if len(tokens) <= 5: # ignore invalid lines and short sentences
                continue
            yield tokens

def question_pair_generator(sentence, masked_placeholder="[MASK]", missing_placeholder="[MASK]"):
    for missing_index, missing_token in enumerate(sentence):
        for masked_index, masked_token in enumerate(sentence):
            if missing_index == masked_index:
                continue
            unmasked_question = list(sentence)
            unmasked_question[missing_index] = missing_placeholder
            masked_question = list(unmasked_question)
            masked_question[masked_index] = masked_placeholder                    
            # context = " ".join(sentence)
            # unmasked_question = " ".join(unmasked_question)
            # masked_question = " ".join(masked_question)
            # answer = sentence[missing_index]
            # yield ((context, unmasked_question, masked_question), missing_index, masked_index, answer)
            context = sentence
            yield ((context, unmasked_question, masked_question), missing_index, masked_index)

def make_prob_prediction(inputs, labels, missing_index, model=global_model):
    outputs = model(**inputs, labels=labels)
    logits = get_logits_of_index(outputs, missing_index)
    probs = F.softmax(torch.squeeze(logits), dim=0)
    return probs

def question_pair_consumer(
        question_pair, 
        text_missing_index, 
        text_masked_index,
        # answer,
        missing_placeholder="[MASK]",
        masked_placeholder="[MASK]",
        tokenizer=global_tokenizer,
        model=global_model,
        measure=cross_entropy_dist):
    get_id = lambda x: tokenizer.convert_tokens_to_ids([x, ])[0]
    missing_id = get_id(missing_placeholder)
    masked_id = get_id(masked_placeholder)
    # answer_id = get_id(answer)
    answer_id = get_id(question_pair[0][text_missing_index])
    
    def consume_question(question, context, masked=False):
        inputs = tokenizer(question, return_tensors="pt")
        labels = tokenizer(context, return_tensors="pt")["input_ids"]
        if inputs["input_ids"].shape != labels.shape:
            # print("Error: inputs and labels are not aligned.")
            return None
        try: 
            outputs = model(**inputs, labels=labels)
        except ValueError as e:
            print(question, context, inputs["input_ids"].shape, labels.shape)
            exit()

        inputs = inputs["input_ids"]
        missing_index, masked_index = -1, -1
        if masked and missing_id == masked_id:
            indices = get_indices_of_id(inputs, masked_id)
            if len(indices) != 2:
                print("Error: cannot determine missing and masked tokens.")
                return None
            missing_index, masked_index = indices if text_missing_index < text_masked_index else (indices[1], indices[0])
        else:
            get_index = lambda id: get_indices_of_id(inputs, id)[0]
            missing_index = get_index(missing_id)
            if masked:
                masked_index = get_index(masked_id)

        logits = get_logits_of_index(outputs, missing_index)
        # probs = F.softmax(torch.squeeze(logits), dim=0)
        return logits

    context, u_question, m_question = list(map(lambda x: " ".join(x), question_pair))
    u_pred = consume_question(u_question, context, masked=False)
    m_pred = consume_question(m_question, context, masked=True)

    if u_pred == None or m_pred == None:
        return None

    return measure(m_pred, u_pred, answer_id)

def iterate_on_sentence(sentence, measure=kl_divergence_dist):
    sentence = sentence.strip().split()
    for question_pair_info in question_pair_generator(sentence):
        question_pair, missing_index, masked_index = question_pair_info
        context = question_pair[0]
        missing_token = context[missing_index]
        masked_token = context[masked_index]
        distance = question_pair_consumer(*question_pair_info, measure)
        if distance != None:
            relation = {
                "missing_index": missing_index,
                "masked_index": masked_index,
                "distance": distance
            }
            print(relation)

def main():
    sentence_dict = {}
    relation_list = []
    counter = 0
    log_interval = 100
    save_interval = 500
    with open("filelist.txt", "r") as f_list:
        file_paths = f_list.read().split()
    with open("progress.log", "r") as p_log:
        last_file_id, last_stc_id, last_counter = p_log.read().split()
    for file_id, file_path in enumerate(file_paths):
        for stc_id, sentence in enumerate(sentence_iterator(file_path)):
            for question_pair_info in question_pair_generator(sentence):
                sentence_dict[stc_id] = sentence
                question_pair, missing_index, masked_index = question_pair_info
                context = question_pair[0]
                missing_token = context[missing_index]
                masked_token = context[masked_index]
                distance = question_pair_consumer(*question_pair_info, measure=kl_divergence_dist)
                if distance != None:
                    counter += 1
                    relation_list.append({
                        "context": file_id * 50000 + stc_id,
                        "missing_index": missing_index,
                        "masked_index": masked_index,
                        "distance": distance
                    })
                    if counter % log_interval == 0:
                        print("Got example count: ", counter)
                    if counter % save_interval == 0:
                        print("Save examples.")
                        stc_dump_name = "sentence_dict_cnt_{0}.dump".format(counter)
                        rel_dump_name = "relation_list_cnt_{0}.dump".format(counter)
                        stc_dump_path = os.path.join(DUMP_DIR, stc_dump_name)
                        rel_dump_path = os.path.join(DUMP_DIR, rel_dump_name)
                        with open(stc_dump_path, "wb") as f_stc:
                            pickle.dump(sentence_dict, f_stc)
                        with open(rel_dump_path, "wb") as f_rel:
                            pickle.dump(relation_list, f_rel)
                        sentence_dict = {}
                        relation_list = []
                with open("progress_kl.log", "w") as progress:
                    progress.write("{0} {1} {2}".format(file_id, stc_id, counter))
    return

    list.sort(relation_list, key=lambda x:x["distance"], reverse=True)
    print("Got example count: ", len(relation_list))
    print(relation_list[0:5])
    
    

    tokenizer = global_tokenizer
    
    inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    print(inputs)
    return 
    
    outputs = model(**inputs, labels=labels)
    
    mask_id = tokenizer.convert_tokens_to_ids(["[MASK]",])[0]
    inputs = torch.squeeze(inputs["input_ids"])
    mask_index = get_indices_of_id(inputs, mask_id)[0]
    print(mask_id, mask_index, inputs[mask_index])
    mask_logits = get_logits_of_index(outputs, mask_index)
    _, _, pred_token = get_prediction_from_logits(mask_logits)
    print(pred_token)
    

if __name__ == "__main__":
    sentence = " Once the project was completed in 1995, the total cost for the Metro Subway was $1.392 billion."
    # sentence = input("sentence:")
    iterate_on_sentence(sentence)
    # main()
    