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
"""
Adaptation of Huggingface's code for fine-tuning the library models for language modeling.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from dataloaders.tokenizers import VCRGpt2Tokenizer
from scripts.model import GPT2VisionAttentiveLMHead
from dataloaders.rationale_generation import RationaleGenerationDataset

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2_vcr': (GPT2Config, GPT2VisionAttentiveLMHead, VCRGpt2Tokenizer),
}


def load_examples(args, tokenizer, is_eval):
    dataset = RationaleGenerationDataset(args=args, tokenizer=tokenizer, is_eval=is_eval)
    return dataset


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


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, "tb/"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_dataset))
    logger.info("  Num train epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        logging.info("\n\n*** Starting Epoch: {} ***\n\n".format(epoch))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs, token_type_ids, sequential_positions, labels, visual_representations, box_coordinates, orig_num_situ_or_object_boxes = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            token_type_ids = token_type_ids.to(args.device) if args.use_token_type_ids else None
            visual_representations = visual_representations.to(args.device) if (args.embeddings_object_detector or args.embeddings_situation_recognizer) else None

            box_coordinates = box_coordinates.to(args.device) if (args.embeddings_object_detector or args.embeddings_situation_recognizer) else None
            orig_num_situ_or_object_boxes = orig_num_situ_or_object_boxes.to(args.device) if (args.embeddings_object_detector or args.embeddings_situation_recognizer) else None
            sequential_positions = sequential_positions.to(args.device) if (
                                                                            args.embeddings_situation_recognizer or
                                                                            args.embeddings_object_detector) else None
            outputs = model(inputs, token_type_ids=token_type_ids, position_ids=sequential_positions, labels=labels,
                            visual_representations=visual_representations, box_coordinates=box_coordinates,
                            orig_num_situ_or_object_boxes=orig_num_situ_or_object_boxes)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps,
                                         global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    logging.info("Evaluate step {}".format(global_step))
                    results = evaluate(args, model, tokenizer, prefix=str(global_step))
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key.split("_")[0]), value, global_step)

                    output_eval_file = open(os.path.join(args.output_dir, "metrics_steps.out"), "a")
                    out_string = str(global_step) + '\t'
                    for key, value in results.items():
                        out_string += str(value) + '\t'
                    out_string += '\n'
                    output_eval_file.write(out_string)
                    output_eval_file.close()

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir

    eval_dataset = load_examples(args, tokenizer, is_eval=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, drop_last=True)

    # Eval!
    logger.info("\n***** Running evaluation at step {} *****".format(prefix))
    logger.info("  Num eval examples = %d", len(eval_dataset))
    logger.info("  Eval batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs, token_type_ids, sequential_positions, labels, visual_representations, box_coordinates, orig_num_situ_or_object_boxes = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            token_type_ids = token_type_ids.to(args.device) if args.use_token_type_ids else None
            visual_representations = visual_representations.to(args.device) if (args.embeddings_object_detector or args.embeddings_situation_recognizer) else None
            box_coordinates = box_coordinates.to(args.device) if (args.embeddings_object_detector or args.embeddings_situation_recognizer) else None
            orig_num_situ_or_object_boxes = orig_num_situ_or_object_boxes.to(args.device) if (args.embeddings_object_detector or args.embeddings_situation_recognizer) else None
            sequential_positions = sequential_positions.to(args.device) if (
                                                                            args.embeddings_situation_recognizer or
                                                                            args.embeddings_object_detector) else None
            outputs = model(inputs, position_ids=sequential_positions, labels=labels, token_type_ids=token_type_ids,
                            visual_representations=visual_representations, box_coordinates=box_coordinates,
                            orig_num_situ_or_object_boxes=orig_num_situ_or_object_boxes)

            tmp_eval_loss = outputs[0]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    output_eval_file = os.path.join(eval_output_dir, "metrics.json")

    if os.path.exists(output_eval_file):
        results = json.load(open(output_eval_file))
    else:
        results = {}

    if len(prefix) == 0:
        results.update({
            "perplexity": perplexity.item(),
            "lm_loss": eval_loss,
        })
    else:
        results.update({
            "perplexity_{}".format(prefix): perplexity.item(),
            "lm_loss_{}".format(prefix): eval_loss,
        })

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write(json.dumps(results))
        writer.close()
    logger.info("***** Finished evaluation {} *****\n".format(prefix))

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('-e', '--hyperparameters', type=str, required=True,
                        help='hyperparameter environment, check environments/hyperparameters.py')
    parser.add_argument("--task", type=str, required=True,
                        help="Choose one of the following: vcr, vqa, e_snlive_ve")

    # Other parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default='gpt2_vcr', type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='gpt2', type=str,
                        help="The model checkpoint for weights initialization. "
                             "Leave None if you want to train a model from scratch.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path. "
                             "If both are None, initialize a new config.",)
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. "
                             "If both are None, initialize a new tokenizer.")
    parser.add_argument("--block_size", default=192, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs"
                             "(take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Limit the total amount of checkpoints, "
                             "delete the older checkpoints in the output_dir, does not delete by default")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and"
                             "ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Not available in huggingace/transformers; specific for our project
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
    parser.add_argument('--textual_viscomet_inferences', action='store_true',
                        help='append to input visual comet inference tokens')
    parser.add_argument('--embeddings_viscomet', action='store_true',
                        help='append VisualComet embeddings that signal the beginning of the before, after, intent '
                             'inferences')

    parser.add_argument('--max_situ_role_num', type=int, default=-1, help='the number of situations')
    parser.add_argument('--max_object_num', type=int, default=-1, help='the number of boxes')

    parser.add_argument('--do_padding', default=False, help='pad question/answer/rationale to their max length ')
    parser.add_argument('--max_answer_length', type=int, default=59, help='maximum answer length in number of tokens')
    parser.add_argument('--max_question_length', type=int, default=71, help='maximum question length in number of tokens')
    parser.add_argument('--max_rationale_length', type=int, default=173, help='maximum rationale length in number of tokens')
    parser.add_argument('--max_situation_length', type=int, default=30, help='maximum situation length in number of tokens')
    parser.add_argument('--max_viscomet_inferences_length', type=int, default=148, help='maximum viscomet inferences length')
    parser.add_argument('--use_token_type_ids', default=False, help='use segments')


    # These are not used
    '''
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_cache", action='store_true', help="Don't cache features")
    '''
    args = parser.parse_args()

    if args.task not in ['vcr', 'vqa', 'e_snli_ve']:
        raise ValueError("Currently only 3 tasks are supported: vcr, vqa, esnlive.")

    if args.task == 'vcr':
        from environments.hyperparameters_vcr import HYPERPARAMETERS
    if args.task == 'vqa':
        from environments.hyperparameters_vqa import HYPERPARAMETERS
    if args.task == 'e_snli_ve':
        from environments.hyperparameters_esnlive import HYPERPARAMETERS

    environment = HYPERPARAMETERS[args.hyperparameters]
    args.train_data_file = environment["train_data_file"]
    args.output_dir = environment["output_dir"]
    args.task = environment["task"]
    args.model_type = environment["model_type"]
    args.model_name_or_path = environment["model_name_or_path"]
    args.per_gpu_train_batch_size = environment["per_gpu_train_batch_size"]
    args.per_gpu_eval_batch_size = environment["per_gpu_eval_batch_size"]
    args.eval_data_file = environment["eval_data_file"]
    args.save_steps = environment["save_steps"]
    args.num_train_epochs = environment["num_train_epochs"]
    args.do_train = environment["do_train"]
    args.learning_rate = environment["learning_rate"]
    args.logging_steps = environment["logging_steps"]
    args.do_eval = environment["do_eval"]
    args.text_only = environment["text_only"]
    args.embeddings_object_detector = environment["embeddings_object_detector"]
    args.embeddings_situation_recognizer = environment["embeddings_situation_recognizer"]
    args.textual_output_object_detector = environment["textual_output_object_detector"]
    args.textual_output_situation_recognizer = environment["textual_output_situation_recognizer"]
    args.textual_viscomet_inferences = environment["textual_viscomet_inferences"]
    args.embeddings_viscomet = environment['embeddings_viscomet']
    args.max_situ_role_num = environment["max_situ_role_num"]
    args.max_object_num = environment["max_object_num"]
    args.do_padding = environment["do_padding"]
    args.max_answer_length = environment["max_answer_length"]
    args.max_question_length = environment["max_question_length"]
    args.max_rationale_length = environment["max_rationale_length"]
    args.max_situation_length = environment["max_situation_length"]
    args.max_viscomet_inferences_length = environment["max_viscomet_inferences_length"]
    args.use_token_type_ids = environment["use_token_type_ids"]
    args.block_size = environment["block_size"]

    if args.task == 'vcr' and args.embeddings_object_detector and args.max_object_num != 28:
        raise ValueError("For VCR the maximum number of objects has to be 28.")

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training
                                     # download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.vision_embd = environment['vision_embd'] if 'vision_embd' in environment else None
    config.n_coordinates = environment['n_coordinates'] if 'n_coordinates' in environment else None
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training
                                     # download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process
                                         # the dataset, and the others will use the cache

        train_dataset = load_examples(
            args,
            tokenizer,
            is_eval=False
        )

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    return results


if __name__ == "__main__":
    main()
