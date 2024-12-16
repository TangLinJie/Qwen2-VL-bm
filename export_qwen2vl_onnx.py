#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import torch
import argparse
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import Qwen2VLForConditionalGeneration
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=2048, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
parser.add_argument('--lmhead_with_topk', action="store_true", default=False, help="trace the LmHeadWithTopK, otherwise trace PenaltySampleHead")

args = parser.parse_args()

model_path = args.model_path

execution_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(execution_dir, "../models/onnx/llm") # folder for LLM
if not os.path.exists(folder):
    os.makedirs(folder)

vit_folder = os.path.join(execution_dir, "../models/onnx/vit") # folder for VIT
if not os.path.exists(vit_folder):
    os.makedirs(vit_folder)

device = torch.device(args.device)
if args.device == 'cpu':
    torch.set_num_threads(args.num_threads)

origin_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float, device_map="auto"
).eval()

for param in origin_model.parameters():
    param.requires_grad = False

ViT = origin_model.visual
config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size

if SEQ_LENGTH + 1 > config.max_position_embeddings:
    for layer_id in range(NUM_LAYERS):
        layers[layer_id].self_attn.rotary_emb._set_cos_sin_cache(
            seq_len=SEQ_LENGTH + 100, device="cpu", dtype=torch.get_default_dtype()
        )

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, rotary_pos_emb, visual_attention_mask):
        out = ViT(hidden_states, rotary_pos_emb, visual_attention_mask)
        return out

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


class Qwen2VLBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, 
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask,
            position_ids)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class Qwen2VLBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, attention_mask, position_ids, past_k, past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask,
            position_ids,
            past_k,
            past_v,)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.temperature = 0.01

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        m_logits /= self.temperature
        _, token = torch.topk(m_logits.float(), 1)
        # batch, 1 
        return token

class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        return m_logits


class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token

    
# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHead(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token
    

def convert_block(layer_id):
    model = Qwen2VLBlock(layer_id)
    # batch 1
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor(
        [list(range(SEQ_LENGTH))] * 3, dtype=torch.long).to(device).reshape(3, 1, SEQ_LENGTH)
    attention_mask = torch.randn(
        (SEQ_LENGTH, SEQ_LENGTH)).float().to(device)
    torch.onnx.export(
        model, (hidden_states, attention_mask, position_ids),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'attention_mask', 'position_ids'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = Qwen2VLBlockCache(layer_id)
    # batch 1
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor([[58]] * 3, dtype=torch.long).to(device).reshape(3, 1, 1)
    attention_mask = torch.ones(
        (1, SEQ_LENGTH + 1)).float().to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).float().to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).float().to(device)

    torch.onnx.export(
        model, (hidden_states, attention_mask, position_ids, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'attention_mask', 'position_ids', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

def convert_vision_transformer():
    model = VisionTransformer()
    hidden_states = torch.randn(11616, 1176).to(dtype=torch.float32, device=device)
    rotary_pos_emb = torch.randn(11616, 40).to(dtype=torch.float32, device=device)
    visual_attention_mask = torch.randn(1, 11616, 11616).to(dtype=torch.float32, device=device)
    # grid_thw = torch.tensor([[ 2, 44, 66],[ 2, 44, 66]]).to(dtype=torch.int64, device=device)
    x = (hidden_states, rotary_pos_emb, visual_attention_mask)
    dynamic_axes = {'hidden_states': {0: 'batch'},
                    "rotary_pos_emb": {0: 'batch'},
                    "visual_attention_mask": {1: 'batch1', 2: 'batch2'},
                    "output": {0: 'batch'}}

    torch.onnx.export(
        model, x,
        f'{vit_folder}/vision_transformer.onnx',
        verbose=False,
        input_names=['hidden_states', 'rotary_pos_emb', 'visual_attention_mask'],
        output_names=['output'],
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        opset_version=15
    )

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


def convert_lm_head_with_topk():
    model = LmHeadWithTopK()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head_with_topk.pt')

def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)

def convert_penalty_sample_head():   
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)


# export models
print(f'Convert Vision Transformer')
convert_vision_transformer()

print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
if not args.lmhead_with_topk:
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()
else:
    convert_lm_head_with_topk()
print("Done")