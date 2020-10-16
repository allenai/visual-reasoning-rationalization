# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# limitations under the License.
"""
Adaptation of Huggingface's code for fine-tuning the library models for language modeling.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import pickle
import h5py

import torch
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

from tqdm import tqdm, trange

from dataloaders.tokenizers import VCRGpt2Tokenizer
from utils.file_utils import read_jsonl_lines
from scripts.model import GPT2VisionAttentiveLMHead

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2_vcr': (GPT2Config, GPT2VisionAttentiveLMHead, VCRGpt2Tokenizer),
}

from config import VCR_QUESTION_FIELD, VCR_RATIONALE_FIELD, VCR_ANSWER_FIELD, SITUATION_FIELD, \
    SITU_ANNOT_FILE, VCR_IMAGES_DIR, VCR_FEATURES_DIR, VISCOMET_FIELD, COCO_JSON, VQA_TRAIN_PICKLE, \
    VQA_VAL_PICKLE, ESNLIVE_PICKLE, VISCOMET_ANNOT_DIR


def get_labels(tokenizer: VCRGpt2Tokenizer, tokenized_text):
    try:
        rationale_start_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.begin_rationale])[0]
        start_idx = tokenized_text.index(rationale_start_token_idx)
        rationale_end_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.end_rationale])[0]
        end_idx = tokenized_text.index(rationale_end_token_idx)

        labels = [-1] * len(tokenized_text)
        labels[start_idx + 1:end_idx + 1] = tokenized_text[start_idx + 1:end_idx + 1]

        assert len(tokenized_text) == len(labels)
        return labels
    except ValueError:
        import pdb; pdb.set_trace()
        raise Exception("Failed to tokenize: {}".format(tokenized_text))


def _map_numbers_to_det_numbers_objects(sent, objects, map_idx=None):
    tokens = sent.replace(',', ' , ').replace("'", " '").replace('.', ' .').replace('?', ' ?').split()
    person_ids = set()
    for t in tokens:
        if t.isdigit() and int(t) - 1 < len(objects):
            person_ids.add(t)

    person_ids = list(person_ids)
    if map_idx is None:
        map_idx = {}

    for i in range(len(person_ids)):
        tag = int(person_ids[i]) - 1
        if tag < len(objects):
            if objects[tag] == 'person':
                map_idx[person_ids[i]] = '<|det%d|>' % (int(person_ids[i]))
            else:
                map_idx[person_ids[i]] = objects[tag]

    tokens = [t if not t.isdigit() or t not in map_idx else map_idx[t] for t in tokens]
    new_sent = ' '.join(tokens).strip().replace(' .', '.')
    return new_sent


def _tokenize_append_truncate(tokenizer, text, max_length, end_token):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [end_token]
    elif len(tokens) < max_length:
        diff = max_length - len(tokens)
        tokens = tokens[:len(tokens)-1]
        tokens.extend([tokenizer.unk_token] * diff)
        tokens.append(end_token)
    return tokens


def _tokenize_truncate(tokenizer, text, max_length, end_token):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [end_token]
    return tokens


def _tokenize(tokenizer, text, max_length, end_token, do_padding):
    if do_padding:
        tokens = _tokenize_append_truncate(tokenizer, text, max_length, end_token)
    else:
        tokens = _tokenize_truncate(tokenizer, text, max_length, end_token)
    return tokens


def _truncate(tokens, max_length, end_token, unk_token, token_type_ids=None, sequential_positions=None):
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [end_token]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:max_length]
        if sequential_positions is not None:
            sequential_positions = sequential_positions[:max_length]
    else:
        diff = max_length - len(tokens)
        tokens.extend([unk_token] * diff)
        if token_type_ids is not None:
            final_segment_id = max(token_type_ids)
            token_type_ids.extend([final_segment_id] * diff)
        if sequential_positions is not None:
            sequential_positions.extend(list(range(max_length-diff, max_length)))
            assert len(tokens) == len(token_type_ids) == len(sequential_positions)
    return tokens, token_type_ids, sequential_positions


def vcr_record_to_tokens(tokenizer: VCRGpt2Tokenizer,
                         record,
                         embeddings_object_detector,
                         embeddings_situation_recognizer,
                         textual_output_object_detector,
                         textual_output_situation_recognizer,
                         textual_viscomet_inferences,
                         embeddings_viscomet,
                         max_situ_role_num,
                         max_object_num,
                         max_question_length,
                         max_answer_length,
                         max_rationale_length,
                         max_situation_length,
                         max_viscomet_inferences_length,
                         max_seq_len,
                         do_padding,
                         use_token_type_ids,
                         gsr_predictions,
                         object_boxes,
                         viscomet_embed
                         ):
    if textual_viscomet_inferences or embeddings_viscomet:
        metadata_fn = VCR_IMAGES_DIR + record["img_fn"].replace('.jpg', '.json')
        with open(metadata_fn, 'r') as fp:
            metadata = json.load(fp)
        objects = metadata['names']
    else:
        objects = record['objects']
    question = _map_numbers_to_det_numbers_objects(record[VCR_QUESTION_FIELD], objects)
    answer = _map_numbers_to_det_numbers_objects(record[VCR_ANSWER_FIELD], objects)
    rationale = _map_numbers_to_det_numbers_objects(record[VCR_RATIONALE_FIELD], objects)

    question_extended = ' '.join([tokenizer.begin_question, question, tokenizer.end_question])
    question_tokens = _tokenize(tokenizer, question_extended, max_question_length, tokenizer.end_question, do_padding)
    question_type_ids = [0]*len(question_tokens)

    answer_extended = ' '.join([tokenizer.begin_answer, answer, tokenizer.end_answer])
    answer_tokens = _tokenize(tokenizer, answer_extended, max_answer_length, tokenizer.end_answer, do_padding)
    answer_type_ids = [1]*len(answer_tokens)

    rationale_extended = ' '.join([tokenizer.begin_rationale, rationale, tokenizer.end_rationale])
    rationale_tokens = _tokenize(tokenizer, rationale_extended, max_rationale_length, tokenizer.end_rationale,
                                 do_padding)
    rationale_type_ids = [2]*len(rationale_tokens)

    textual_tokens = question_tokens + answer_tokens + rationale_tokens
    token_type_ids = question_type_ids + answer_type_ids + rationale_type_ids if use_token_type_ids else None

    if textual_output_situation_recognizer:
        situation_extended = record[SITUATION_FIELD]
        situation_tokens = _tokenize(tokenizer, situation_extended, max_situation_length, tokenizer.end_situation,
                                     do_padding)
        prompt = situation_tokens + question_tokens + answer_tokens + rationale_tokens
        if use_token_type_ids:
            token_type_ids = [0]*len(situation_tokens)+[1]*len(question_tokens)+\
                             [2]*len(answer_tokens)+[3]*len(rationale_tokens)

        prompt, token_type_ids, _ = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                           tokenizer.unk_token, token_type_ids)
        return prompt, token_type_ids, None, None, None, None

    if embeddings_situation_recognizer:
        visual_tokens = [tokenizer.begin_situation] + [tokenizer.unk_token] * max_situ_role_num + [tokenizer.end_situation]
        prompt = visual_tokens + textual_tokens
        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        # Load situation embeddings
        # The final representation is for the whole image
        situation_path = SITU_ANNOT_FILE + 'vcr/' + '_'.join(record['img_fn'].split("/")).replace('jpg','hdf5')
        situation_representations_file = h5py.File(situation_path, "r")
        situation_representations = [torch.tensor(situation_representations_file[key][()])
                                     for key in situation_representations_file.keys()]
        # Put the entire image representation to the first place
        situations_representations_reorder = [situation_representations[-1]] + situation_representations[:-1]

        # Load situation box coordinates
        img_name = '_'.join(record['img_fn'].split("/"))
        box_coordinates = gsr_predictions[img_name]['boxes']

        metadata_path = VCR_IMAGES_DIR + record['metadata_fn']
        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
        width = metadata['width']
        height = metadata['height']

        box_coordinates_extended = [[0, 0, width, height, 1.0]] # First element for the whole image
        for location in box_coordinates:
            if location:
                area_ratio = (location[3]-location[1])*(location[2]-location[0]) / (width * height)
                box_coordinates_extended.append(location + [area_ratio])
            else:
                box_coordinates_extended.append([0, 0, width, height, 1.0])
        number_of_roles = len(situations_representations_reorder)
        diff = max_situ_role_num - len(situations_representations_reorder)

        # Pad for batching
        situations_representations_reorder = torch.stack(situations_representations_reorder).squeeze(0)
        if diff > 0:
            vision_embd = situations_representations_reorder.shape[1]
            situations_representations_reorder = torch.cat((situations_representations_reorder,
                                                            torch.zeros(diff, vision_embd)), dim=0)
            box_coordinates_extended.extend([[0] * 5]*diff)
        box_coordinates_extended = torch.tensor(box_coordinates_extended)

        sequential_positions = [0]*len(visual_tokens) + list(range(1, len(prompt)-len(visual_tokens)+1))
        prompt, token_type_ids, sequential_positions = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                              tokenizer.unk_token, token_type_ids, sequential_positions)
        return prompt, token_type_ids, sequential_positions, situations_representations_reorder, \
               box_coordinates_extended, number_of_roles

    if textual_output_object_detector:
        inanimate_objects = ' '.join(list(set([r for r in record["objects"] if r != 'person'])))
        object_labels_extended = ' '.join([tokenizer.begin_img, inanimate_objects, tokenizer.end_img])
        visual_tokens = _tokenize(tokenizer, object_labels_extended, max_object_num, tokenizer.end_img, do_padding)
        prompt = visual_tokens + question_tokens + answer_tokens + rationale_tokens

        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        prompt, token_type_ids, _ = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                            tokenizer.unk_token, token_type_ids)
        return prompt, token_type_ids, None, None, None, None,

    if embeddings_object_detector:
        visual_tokens = [tokenizer.begin_img] + [tokenizer.unk_token] * max_object_num + [tokenizer.end_img]
        prompt = visual_tokens + textual_tokens

        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        # The first representation is for the entire image
        image_name = '_'.join(record['img_fn'].split('/')).replace('.jpg', '')
        representations_file = h5py.File(VCR_FEATURES_DIR + '{}.hdf5'.format(image_name), "r")
        object_representations = [torch.tensor(representations_file[key][()]) for key in representations_file.keys()]
        object_representations_tensor = torch.stack(object_representations).squeeze(0)

        box_coordinates = torch.Tensor(object_boxes[image_name]['boxes'])
        width = box_coordinates[0][2]
        height = box_coordinates[0][3]
        box_coordinates_extended = []
        for location in box_coordinates:
            area_ratio = (location[3]-location[1])*(location[2]-location[0]) / (width * height)
            if torch.isnan(torch.tensor(area_ratio)).item() or torch.isinf(torch.tensor(area_ratio)).item():
                raise ValueError("NaN or inf encountered")
            box_coordinates_extended.append(location.tolist() + [area_ratio])
        box_coordinates_extended = torch.tensor(box_coordinates_extended)
        orig_num_boxes = min(object_boxes[image_name]['orig_num_boxes'], max_object_num)

        sequential_positions = [0] * len(visual_tokens) + list(range(len(prompt) - len(visual_tokens)))
        prompt, token_type_ids, sequential_positions = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                            tokenizer.unk_token, token_type_ids, sequential_positions)

        return prompt, token_type_ids, sequential_positions, object_representations_tensor, box_coordinates_extended, \
               orig_num_boxes

    if textual_viscomet_inferences:
        viscomet_inferences_extended = record[VISCOMET_FIELD]
        viscomet_inferences = _tokenize(tokenizer, viscomet_inferences_extended, max_viscomet_inferences_length,
                                        tokenizer.end_intent, do_padding)
        prompt = viscomet_inferences + question_tokens + answer_tokens + rationale_tokens
        if use_token_type_ids:
            token_type_ids = [0]*len(viscomet_inferences)+[1]*len(question_tokens)+\
                             [2]*len(answer_tokens)+[3]*len(rationale_tokens)

        prompt, token_type_ids, _ = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                              tokenizer.unk_token, token_type_ids)
        return prompt, token_type_ids, None, None, None, None

    if embeddings_viscomet:
        visual_tokens = [tokenizer.begin_viscomet_token] + [tokenizer.unk_token]*3 + [tokenizer.end_viscomet_token]

        prompt = visual_tokens + textual_tokens

        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        idx = record['idx']
        embeddings = []
        for i in range(3):
            embeddings.append(torch.tensor(viscomet_embed[i][idx]))
        representations = torch.stack(embeddings)

        sequential_positions = [0] * len(visual_tokens) + list(range(len(prompt) - len(visual_tokens)))
        prompt, token_type_ids, sequential_positions = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                                 tokenizer.unk_token, token_type_ids,
                                                                 sequential_positions)
        return prompt, token_type_ids, sequential_positions, representations, None, None

    textual_tokens, token_type_ids, _ = _truncate(textual_tokens, max_seq_len, tokenizer.end_rationale,
                                                  tokenizer.unk_token, token_type_ids)
    return textual_tokens, token_type_ids, None, None, None, None


def record_to_tokens(tokenizer: VCRGpt2Tokenizer,
                     record,
                     embeddings_object_detector,
                     embeddings_situation_recognizer,
                     textual_output_object_detector,
                     textual_output_situation_recognizer,
                     textual_viscomet_inferences,
                     embeddings_viscomet,
                     max_situ_role_num,
                     max_object_num,
                     max_question_length,
                     max_answer_length,
                     max_rationale_length,
                     max_situation_length,
                     max_viscomet_inferences_length,
                     max_seq_len,
                     do_padding,
                     use_token_type_ids,
                     question_field,
                     answer_field,
                     rationale_field,
                     task,
                     pickle_dict,
                     gsr_predictions,
                     split,
                     viscomet_embed,
                     viscomet_text,
                     viscomet_embed_ids):

    question = record[question_field]
    answer = record[answer_field]
    if task == 'vqa':
        rationale = record[rationale_field][0]
    else:
        rationale = record[rationale_field]

    question_extended = ' '.join([tokenizer.begin_question, question, tokenizer.end_question])
    question_tokens = _tokenize(tokenizer, question_extended, max_question_length, tokenizer.end_question, do_padding)
    question_type_ids = [0]*len(question_tokens)

    answer_extended = ' '.join([tokenizer.begin_answer, answer, tokenizer.end_answer])
    answer_tokens = _tokenize(tokenizer, answer_extended, max_answer_length, tokenizer.end_answer, do_padding)
    answer_type_ids = [1]*len(answer_tokens)

    rationale_extended = ' '.join([tokenizer.begin_rationale, rationale, tokenizer.end_rationale])
    rationale_tokens = _tokenize(tokenizer, rationale_extended, max_rationale_length, tokenizer.end_rationale,
                                 do_padding)
    rationale_type_ids = [2]*len(rationale_tokens)

    textual_tokens = question_tokens + answer_tokens + rationale_tokens
    token_type_ids = question_type_ids + answer_type_ids + rationale_type_ids if use_token_type_ids else None

    if textual_output_situation_recognizer:
        situation_extended = record[SITUATION_FIELD]
        situation_tokens = _tokenize(tokenizer, situation_extended, max_situation_length, tokenizer.end_situation,
                                     do_padding)
        prompt = situation_tokens + question_tokens + answer_tokens + rationale_tokens
        if use_token_type_ids:
            token_type_ids = [0]*len(situation_tokens)+[1]*len(question_tokens)+\
                             [2]*len(answer_tokens)+[3]*len(rationale_tokens)

        prompt, token_type_ids, _ = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                           tokenizer.unk_token, token_type_ids)
        return prompt, token_type_ids, None, None, None, None

    if embeddings_situation_recognizer:
        visual_tokens = [tokenizer.begin_situation] + [tokenizer.unk_token] * max_situ_role_num + [tokenizer.end_situation]
        prompt = visual_tokens + textual_tokens
        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        if 'image_id' in record:
            img_id = record['image_id'].split('.')[0]
            situation_path = SITU_ANNOT_FILE + task + '/' + img_id + '.hdf5'
        else:
            split_coco = 'val2014' if split == 'val' else 'train2014'
            img_id = 'COCO_%s_%012d' % (split_coco, record['img_id'])
            situation_path = SITU_ANNOT_FILE + task + '/' + split + '/' + img_id + '.hdf5'

        situation_representations_file = h5py.File(situation_path, "r")
        situation_representations = [torch.tensor(situation_representations_file[key][()])
                                     for key in situation_representations_file.keys()]
        # Put the entire image representation to the first place
        situations_representations_reorder = [situation_representations[-1]] + situation_representations[:-1]

        # Load situation box coordinates
        box_coordinates = gsr_predictions[img_id+'.jpg']['boxes']

        width = pickle_dict[img_id]['width']
        height = pickle_dict[img_id]['height']

        box_coordinates_extended = [[0, 0, width, height, 1.0]] # First element for the whole image
        for location in box_coordinates:
            if location:
                area_ratio = (location[3]-location[1])*(location[2]-location[0]) / (width * height)
                box_coordinates_extended.append(location + [area_ratio])
            else:
                box_coordinates_extended.append([0, 0, width, height, 1.0])
        number_of_roles = len(situations_representations_reorder)
        diff = max_situ_role_num - len(situations_representations_reorder)

        # Pad for batching
        situations_representations_reorder = torch.stack(situations_representations_reorder).squeeze(0)
        if diff > 0:
            vision_embd = situations_representations_reorder.shape[1]
            situations_representations_reorder = torch.cat((situations_representations_reorder,
                                                            torch.zeros(diff, vision_embd)), dim=0)
            box_coordinates_extended.extend([[0] * 5]*diff)
        box_coordinates_extended = torch.tensor(box_coordinates_extended)

        sequential_positions = [0]*len(visual_tokens) + list(range(1, len(prompt)-len(visual_tokens)+1))
        prompt, token_type_ids, sequential_positions = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                              tokenizer.unk_token, token_type_ids, sequential_positions)
        return prompt, token_type_ids, sequential_positions, situations_representations_reorder, \
               box_coordinates_extended, number_of_roles

    if textual_output_object_detector:
        with open(COCO_JSON, 'r') as fp:
            coco_json = json.load(fp)
        if 'image_id' in record:
            img_id = record['image_id'].split('.')[0]
        else:
            split_coco = 'val2014' if split == 'val' else 'train2014'
            img_id = 'COCO_%s_%012d' % (split_coco, record['img_id'])

        try:
            class_indices = pickle_dict[img_id]['classes']
        except KeyError:
            import pdb; pdb.set_trace()

        objects = [coco_json[str(label)]["name"] for label in class_indices]

        inanimate_objects = ' '.join(list(set([r for r in objects if r != 'person'])))
        object_labels_extended = ' '.join([tokenizer.begin_img, inanimate_objects, tokenizer.end_img])
        visual_tokens = _tokenize(tokenizer, object_labels_extended, max_object_num, tokenizer.end_img, do_padding)
        prompt = visual_tokens + question_tokens + answer_tokens + rationale_tokens

        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        prompt, token_type_ids, _ = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                            tokenizer.unk_token, token_type_ids)
        return prompt, token_type_ids, None, None, None, None,

    if embeddings_object_detector:
        visual_tokens = [tokenizer.begin_img] + [tokenizer.unk_token] * max_object_num + [tokenizer.end_img]
        prompt = visual_tokens + textual_tokens

        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        # The first representation is for the entire image
        if 'image_id' in record:
            img_id = record['image_id'].split('.')[0]
        else:
            split_coco = 'val2014' if split=='val' else 'train2014'
            img_id = 'COCO_%s_%012d' % (split_coco, record['img_id'])
        image_features = torch.tensor(pickle_dict[img_id]['image_feature']).unsqueeze(0)

        object_features = torch.tensor(pickle_dict[img_id]['object_features'])
        object_representations_tensor = torch.cat([image_features, object_features], dim=0)

        width = pickle_dict[img_id]['width']
        height = pickle_dict[img_id]['height']
        box_coordinates = [[0, 0, width, height]] + pickle_dict[img_id]['boxes'].tolist()

        box_coordinates_extended = []
        for location in box_coordinates:
            area_ratio = (location[3]-location[1])*(location[2]-location[0]) / (width * height)
            box_coordinates_extended.append(location + [area_ratio])

        orig_num_boxes = object_representations_tensor.shape[0]
        if orig_num_boxes < max_object_num:
            diff = max_object_num - orig_num_boxes
            object_representations_tensor = torch.cat((object_representations_tensor,
                                                       torch.zeros(diff, object_representations_tensor.shape[1])), dim=0)
            box_coordinates_extended.extend([[0] * 5] * diff)
        box_coordinates_extended = torch.tensor(box_coordinates_extended)

        if orig_num_boxes > max_object_num:
            object_representations_tensor = object_representations_tensor[:max_object_num, :]
            box_coordinates_extended = box_coordinates_extended[:max_object_num, :]

        sequential_positions = [0] * len(visual_tokens) + list(range(len(prompt) - len(visual_tokens)))
        prompt, token_type_ids, sequential_positions = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                            tokenizer.unk_token, token_type_ids, sequential_positions)
        orig_num_boxes = min(orig_num_boxes, max_object_num)
        return prompt, token_type_ids, sequential_positions, object_representations_tensor, box_coordinates_extended, \
               orig_num_boxes

    if textual_viscomet_inferences:
        if task == 'vqa':
            viscomet_inferences_extended = record[VISCOMET_FIELD]
        else:
            try:
                viscomet_inferences_extended = viscomet_text[record['image_id']]
            except KeyError:
                import pdb; pdb.set_trace()
        viscomet_inferences = _tokenize(tokenizer, viscomet_inferences_extended, max_viscomet_inferences_length,
                                        tokenizer.end_intent, do_padding)
        prompt = viscomet_inferences + question_tokens + answer_tokens + rationale_tokens
        if use_token_type_ids:
            token_type_ids = [0]*len(viscomet_inferences)+[1]*len(question_tokens)+\
                             [2]*len(answer_tokens)+[3]*len(rationale_tokens)

        prompt, token_type_ids, _ = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                              tokenizer.unk_token, token_type_ids)
        return prompt, token_type_ids, None, None, None, None

    if embeddings_viscomet:
        visual_tokens = [tokenizer.begin_viscomet_token] + [tokenizer.unk_token]*3 + [tokenizer.end_viscomet_token]
        prompt = visual_tokens + textual_tokens

        if use_token_type_ids:
            token_type_ids = [0] * len(visual_tokens) + [1] * len(question_tokens) + \
                             [2] * len(answer_tokens) + [3] * len(rationale_tokens)

        embeddings = []
        for i in range(3):
            if task == 'vqa':
                idx = record['idx']
                embeddings.append(torch.tensor(viscomet_embed[i][idx]))
            else:
                embeddings.append(torch.tensor(viscomet_embed[i][viscomet_embed_ids[i].index(record['image_id'])]))
        representations = torch.stack(embeddings)

        sequential_positions = [0] * len(visual_tokens) + list(range(len(prompt) - len(visual_tokens)))
        prompt, token_type_ids, sequential_positions = _truncate(prompt, max_seq_len, tokenizer.end_rationale,
                                                                 tokenizer.unk_token, token_type_ids,
                                                                 sequential_positions)
        return prompt, token_type_ids, sequential_positions, representations, None, None

    textual_tokens, token_type_ids, _ = _truncate(textual_tokens, max_seq_len, tokenizer.end_rationale,
                                                  tokenizer.unk_token, token_type_ids)
    return textual_tokens, token_type_ids, None, None, None, None


class RationaleGenerationDataset:
    def __init__(self, args, tokenizer, is_eval):
        file_path = args.eval_data_file if is_eval else args.train_data_file
        max_seq_len = args.block_size
        split_filename = os.path.basename(file_path)
        task = args.task

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
                                         # and the others will use the cache

        self.examples = []
        self.token_type_ids = []
        self.labels = []
        self.visual_representations = []
        self.orig_num_situ_or_object_boxes = []
        self.box_coordinates = []
        self.sequential_positions = []

        if args.task == 'vcr':
            with open(os.path.join(SITU_ANNOT_FILE + 'vcr/results.json'), 'r') as json_file:
                gsr_predictions = json.load(json_file)

            with open(os.path.join(VCR_FEATURES_DIR, 'boxes.json'), 'r') as json_file:
                object_boxes = json.load(json_file)

            if args.textual_viscomet_inferences:
                with open(file_path, 'r') as fp:
                    records = json.load(fp)
                    self.records = records
            else:
                self.records = read_jsonl_lines(split_filename)

            idx = 0
            split = 'train' if 'train' in split_filename else 'val'
            viscomet_embed = [h5py.File(os.path.join(VISCOMET_ANNOT_DIR, 'vcr', split + '_' + inference_type + '.h5'), "r")['features'][()] for
                             inference_type in ['need', 'react', 'intent']]
            for record in tqdm(self.records, "Encoding Data"):
                tokens, token_type_ids, sequential_positions, visual_representations, box_coordinates, orig_num_situ_or_object_boxes = \
                                        vcr_record_to_tokens(
                                            tokenizer=tokenizer,
                                            record=record,
                                            embeddings_object_detector=args.embeddings_object_detector,
                                            embeddings_situation_recognizer=args.embeddings_situation_recognizer,
                                            textual_output_object_detector=args.textual_output_object_detector,
                                            textual_output_situation_recognizer=args.textual_output_situation_recognizer,
                                            textual_viscomet_inferences=args.textual_viscomet_inferences,
                                            embeddings_viscomet=args.embeddings_viscomet,
                                            max_situ_role_num=args.max_situ_role_num,
                                            max_object_num=args.max_object_num,
                                            max_question_length=args.max_question_length,
                                            max_answer_length=args.max_answer_length,
                                            max_rationale_length=args.max_rationale_length,
                                            max_situation_length=args.max_situation_length,
                                            max_viscomet_inferences_length=args.max_viscomet_inferences_length,
                                            max_seq_len=max_seq_len,
                                            do_padding=args.do_padding,
                                            use_token_type_ids=args.use_token_type_ids,
                                            gsr_predictions=gsr_predictions,
                                            object_boxes=object_boxes,
                                            viscomet_embed=viscomet_embed
                                        )
                text = " ".join(tokens)
                tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
                self.examples.append(tokenized_text)

                if args.use_token_type_ids:
                    self.token_type_ids.append(token_type_ids)

                labels = get_labels(tokenizer, tokenized_text)
                self.labels.append(labels)

                if args.embeddings_situation_recognizer or args.embeddings_object_detector:
                    self.visual_representations.append(visual_representations)
                    self.box_coordinates.append(box_coordinates)
                    self.orig_num_situ_or_object_boxes.append(orig_num_situ_or_object_boxes)
                    self.sequential_positions.append(sequential_positions)
                if idx < 5:
                    print("***** Example Instance *****")
                    print("Text: {}".format(text))
                    print("Tokenized Text: {}".format(tokenized_text))
                    print("Labels: {}".format(labels))
                    # print("Object = {}".format(record['objects']))
                    if args.use_token_type_ids:
                        print("Token type ids = {}".format(token_type_ids))
                    print("********\n")
                idx += 1
        if args.task in ['vqa', 'e_snli_ve']:
            file_name = args.eval_data_file if is_eval else args.train_data_file

            if args.task == 'vqa':
                question_field = 'question'
                answer_field = 'multiple_choice_answer'
                rationale_field = 'explanation'

                split = 'train' if 'train' in file_name.split('/')[-1] else 'val'
                pickle_path = VQA_VAL_PICKLE if split == 'val' else VQA_TRAIN_PICKLE
                f = open(pickle_path, 'rb')  # 'r' for reading; can be omitted
                pickle_dict = pickle.load(f)  # load file content as mydict
                f.close()

                with open(os.path.join(SITU_ANNOT_FILE + args.task + '/' + split + '/results.json'), 'r') as json_file:
                    gsr_predictions = json.load(json_file)

                viscomet_text = None

                viscomet_embed = [
                    h5py.File(os.path.join(VISCOMET_ANNOT_DIR, task, split + '_' + inference_type + '.h5'), "r")[
                        'features'][()] for
                    inference_type in ['before', 'after', 'intent']]
                viscomet_embed_ids = None
            if args.task == 'e_snli_ve':
                question_field = 'sentence2_tokens'
                answer_field = 'gold_label'
                rationale_field = 'explanation'
                f = open(ESNLIVE_PICKLE, 'rb')  # 'r' for reading; can be omitted
                pickle_dict = pickle.load(f)  # load file content as mydict
                f.close()
                with open(os.path.join(SITU_ANNOT_FILE + args.task + '/results.json'), 'r') as json_file:
                    gsr_predictions = json.load(json_file)
                split = 'train' if 'train' in file_name.split('/')[-1] else 'val'

                with open(os.path.join(VISCOMET_ANNOT_DIR, 'e_snli_ve/viscomet_inferences.json')) as fp:
                    viscomet_text = json.load(fp)

                viscomet_embed_ids = []
                viscomet_embed = []
                for inference_type in ['before', 'after', 'intent']:
                    viscomet_path = os.path.join(VISCOMET_ANNOT_DIR, task, split + '_' + inference_type + '.h5')
                    representations_file = h5py.File(viscomet_path, "r")
                    viscomet_embed.append(representations_file['features'][()])
                    viscomet_embed_ids.append([item.decode("utf-8") for item in representations_file["img_ids"][:].tolist()])

            if not args.textual_output_situation_recognizer:
                with open(file_name, 'r') as fp:
                    records = json.load(fp)
                    self.records = records

            if args.textual_output_situation_recognizer:
                self.records = read_jsonl_lines(file_name)
            idx = 0
            for record in tqdm(self.records, "Encoding Data"):
                tokens, token_type_ids, sequential_positions, visual_representations, box_coordinates, orig_num_situ_or_object_boxes = \
                    record_to_tokens(
                        tokenizer=tokenizer,
                        record=record,
                        embeddings_object_detector=args.embeddings_object_detector,
                        embeddings_situation_recognizer=args.embeddings_situation_recognizer,
                        textual_output_object_detector=args.textual_output_object_detector,
                        textual_output_situation_recognizer=args.textual_output_situation_recognizer,
                        textual_viscomet_inferences=args.textual_viscomet_inferences,
                        embeddings_viscomet=args.embeddings_viscomet,
                        max_situ_role_num=args.max_situ_role_num,
                        max_object_num=args.max_object_num,
                        max_question_length=args.max_question_length,
                        max_answer_length=args.max_answer_length,
                        max_rationale_length=args.max_rationale_length,
                        max_situation_length=args.max_situation_length,
                        max_viscomet_inferences_length=args.max_viscomet_inferences_length,
                        max_seq_len=max_seq_len,
                        do_padding=args.do_padding,
                        use_token_type_ids=args.use_token_type_ids,
                        question_field=question_field,
                        answer_field=answer_field,
                        rationale_field=rationale_field,
                        task=args.task,
                        pickle_dict=pickle_dict,
                        gsr_predictions=gsr_predictions,
                        split=split,
                        viscomet_embed=viscomet_embed,
                        viscomet_text=viscomet_text,
                        viscomet_embed_ids=viscomet_embed_ids
                    )

                text = " ".join(tokens)
                tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
                self.examples.append(tokenized_text)

                if args.use_token_type_ids:
                    self.token_type_ids.append(token_type_ids)

                labels = get_labels(tokenizer, tokenized_text)
                self.labels.append(labels)

                if args.embeddings_situation_recognizer or args.embeddings_object_detector:
                    self.visual_representations.append(visual_representations)
                    self.box_coordinates.append(box_coordinates)
                    self.orig_num_situ_or_object_boxes.append(orig_num_situ_or_object_boxes)
                    self.sequential_positions.append(sequential_positions)

                if args.embeddings_viscomet:
                    self.visual_representations.append(visual_representations)
                    self.sequential_positions.append(sequential_positions)

                if idx < 5:
                    print("***** Example Instance *****")
                    print("Text: {}".format(text))
                    print("Tokenized Text: {}".format(tokenized_text))
                    print("Labels: {}".format(labels))
                    if args.use_token_type_ids:
                        print("Token type ids = {}".format(token_type_ids))
                    print("********\n")
                idx += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokens = torch.tensor(self.examples[index])
        labels = torch.tensor(self.labels[index])

        token_type_ids = torch.tensor(self.token_type_ids[index]) if self.token_type_ids else []

        visual_representations = self.visual_representations[index] if self.visual_representations else []
        box_coordinates = self.box_coordinates[index] if self.box_coordinates else []
        orig_num_situ_or_object_boxes = torch.tensor([self.orig_num_situ_or_object_boxes[index]]) if self.orig_num_situ_or_object_boxes else []
        sequential_positions = torch.tensor(self.sequential_positions[index]) if self.sequential_positions else []
        return tokens, token_type_ids, sequential_positions, labels, visual_representations, box_coordinates, orig_num_situ_or_object_boxes
