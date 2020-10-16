#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
"""
Adaptation of Huggingface's code for fine-tuning the library models for language modeling.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import random
import os

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from scripts.model import GPT2VisionAttentiveLMHead
from scripts.run_finetuning import RationaleGenerationDataset
from dataloaders.tokenizers import VCRGpt2Tokenizer

from tqdm import tqdm
from utils.file_utils import write_items, read_jsonl_lines


MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                  (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2_vcr': (GPT2VisionAttentiveLMHead, VCRGpt2Tokenizer),
}


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        #indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, prompt_token_idx, batch, args):
    context, token_type_ids, sequential_positions, labels, visual_representations, box_coordinates, orig_num_situ_or_object_boxes = batch

    context = context.to(args.device)
    idx_of_prompt_token = (context == prompt_token_idx).nonzero()[0][1].item()
    context = context[:, :idx_of_prompt_token + 1]
    context = context.repeat(args.num_samples, 1)
    generated = context

    if args.use_token_type_ids:
        token_type_ids = token_type_ids.to(args.device)
        token_type_ids = token_type_ids[:, :idx_of_prompt_token + 1]
        token_type_ids = token_type_ids.repeat(args.num_samples, 1)
        rationale_type_idx = torch.max(token_type_ids, dim=1)[0].unsqueeze(0)

    if args.embeddings_object_detector or args.embeddings_situation_recognizer:
        visual_representations = visual_representations.to(args.device)
        box_coordinates = box_coordinates.to(args.device)
        orig_num_situ_or_object_boxes = orig_num_situ_or_object_boxes.to(args.device)
        sequential_positions = sequential_positions[:, :idx_of_prompt_token + 1].to(args.device)

    with torch.no_grad():
        for tok_idx in range(args.max_rationale_length):
            inputs = {'input_ids': generated}
            if args.use_token_type_ids:
                inputs.update({'token_type_ids': token_type_ids})

            if args.embeddings_object_detector or args.embeddings_situation_recognizer:
                inputs.update({'visual_representations': visual_representations,
                               'box_coordinates': box_coordinates,
                               'orig_num_situ_or_object_boxes': orig_num_situ_or_object_boxes,
                               'position_ids': sequential_positions})
            outputs = model(**inputs)

            next_token_logits = outputs[0][0, -1, :] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=args.num_samples)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if args.use_token_type_ids:
                token_type_ids = torch.cat((token_type_ids, rationale_type_idx), dim=1)
            if args.embeddings_object_detector or args.embeddings_situation_recognizer:
                next_position = torch.max(sequential_positions, dim=1)[0] + 1
                sequential_positions = torch.cat((sequential_positions, next_position.unsqueeze(-1)), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('-e', '--hyperparameters', type=str, required=True,
                        help='hyperparameter environment, check environments/')
    parser.add_argument("--task", type=str, required=True,
                        help="Choose one of the following: vcr, vqa, esnlive")

    # Other
    parser.add_argument("--model_type", default='gpt2_vcr', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save genarations to")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help="Max seq length")
    parser.add_argument("--num_samples", default=1, type=int, help="No. of samples to obtain.")
    parser.add_argument("--gen_batch_size", default=1, type=int, help="No. of instances per batch.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--text_only', action='store_true',
                        help="input are only question and answer, image is ignored")
    parser.add_argument('--embeddings_object_detector', action='store_true',
                        help="append to input embeddings of bounding boxes produced by object detector,"
                             "condider only objects (that is, boxes) mentioned in question and answer")
    parser.add_argument('--embeddings_situation_recognizer', action='store_true',
                        help="append to input embeddings of bounding boxes produced by situation recognizer,"
                             "condider only situation (that is, boxes) whose roles appear in question and answer")
    parser.add_argument('--textual_output_object_detector', action='store_true',
                        help="append to input tokens of object labels,"
                             "condider only objects mentioned in question and answer")
    parser.add_argument('--textual_output_situation_recognizer', action='store_true',
                        help="append to input tokens of situation output,"
                             "condider only situations whose roles are mentioned in question and answer")
    parser.add_argument('--embeddings_viscomet', action='store_true',
                        help='append VisualComet embeddings that signal the beginning of the before, after, intent '
                             'inferences')
    args = parser.parse_args()

    if args.gen_batch_size != 1 or args.num_samples != 1:
        raise ValueError(
            "Currently only the batch size of 1 is supported and only one sample per instance are supported.")

    if args.task not in ['vcr', 'vqa', 'e_snli_ve']:
        raise ValueError("Currently only 3 tasks are supported: vcr, vqa, esnlive.")

    if args.task == 'vcr':
        from environments.hyperparameters_vcr import HYPERPARAMETERS
    if args.task == 'vqa':
        from environments.hyperparameters_vqa import HYPERPARAMETERS
    if args.task == 'e_snli_ve':
        from environments.hyperparameters_esnlive import HYPERPARAMETERS

    environment = HYPERPARAMETERS[args.hyperparameters]
    args.eval_data_file = environment["eval_data_file"]
    args.model_type = environment["model_type"]
    args.output_file = os.path.join(args.model_name_or_path, args.eval_data_file.split('/')[-1])
    args.text_only = environment["text_only"]
    args.embeddings_object_detector = environment["embeddings_object_detector"]
    args.embeddings_situation_recognizer = environment["embeddings_situation_recognizer"]
    args.textual_output_object_detector = environment["textual_output_object_detector"]
    args.textual_output_situation_recognizer = environment["textual_output_situation_recognizer"]
    args.textual_viscomet_inferences = environment["textual_viscomet_inferences"]
    args.embeddings_viscomet = environment['embeddings_viscomet']
    args.max_situ_role_num = environment["max_situ_role_num"]
    args.max_object_num = environment["max_object_num"]
    args.top_p = environment["top_p"]
    args.top_k = environment["top_k"]
    args.temperature = environment["temperature"]
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.do_padding = environment["do_padding"]
    args.max_answer_length = environment["max_answer_length"]
    args.max_question_length = environment["max_question_length"]
    args.max_rationale_length = environment["max_rationale_length"]
    args.max_situation_length = environment["max_situation_length"]
    args.max_viscomet_inferences_length = environment["max_viscomet_inferences_length"]
    args.block_size = environment["block_size"]
    args.length = environment["length"]
    args.use_token_type_ids = environment["use_token_type_ids"]
    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.max_rationale_length < 0 < model.config.max_position_embeddings:
        args.max_rationale_length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.max_rationale_length:
        args.max_rationale_length = model.config.max_position_embeddings  # No generation bigger than model size
    elif args.max_rationale_length < 0:
        args.max_rationale_length = MAX_LENGTH  # avoid infinite loop

    print(args)

    if args.task == 'vcr':
        if not args.textual_viscomet_inferences:
            records = read_jsonl_lines(args.eval_data_file)
        else:
            with open(args.eval_data_file, 'r') as fp:
                records = json.load(fp)
        question_field = 'question_orig'
        answer_field = 'answer_orig'
        rationale_field = 'rationale_orig'
    else:
        if not args.textual_output_situation_recognizer:
            with open(args.eval_data_file, 'r') as fp:
                records = json.load(fp)
        if args.textual_output_situation_recognizer:
            records = read_jsonl_lines(args.eval_data_file)
        if args.task == 'vqa':
            question_field = 'question'
            answer_field = 'multiple_choice_answer'
            rationale_field = 'explanation'
        if args.task == 'e_snli_ve':
            question_field = 'sentence2_tokens'
            answer_field = 'gold_label'
            rationale_field = 'explanation'

    dataset = RationaleGenerationDataset(args, tokenizer=tokenizer, is_eval=True)

    results = []
    idx = 0
    eval_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.gen_batch_size)

    for batch in tqdm(eval_dataloader):
        prompt_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.begin_rationale])[0]

        out = sample_sequence(model=model, prompt_token_idx=prompt_token_idx, batch=batch, args=args)
        idx_of_prompt_token = (batch[0] == prompt_token_idx).nonzero()[0][1].item()
        instance_generations_indices = out[0, idx_of_prompt_token + 1:].tolist()
        sample_text = tokenizer.decode(instance_generations_indices, clean_up_tokenization_spaces=True,
                                       skip_special_tokens=True)
        input_record = records[idx]
        if "generated_rationales" in input_record:
            del input_record["generated_rationales"]

        input_record['generated_rationale'] = sample_text
        results.append(input_record)

        if idx < 20:
            print("Input context format: {}".format(input_record[question_field]))
            print("Generated rationale: {}".format(sample_text))
            print("Correct rationale: {}".format(input_record[rationale_field]))
            print("Correct answer: {}".format(input_record[answer_field]))
            print('\n')
        idx += 1

    assert results
    write_items([json.dumps(r) for r in results], args.output_file)


if __name__ == '__main__':
    main()
