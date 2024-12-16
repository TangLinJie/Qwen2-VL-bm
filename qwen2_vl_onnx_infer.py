import time
import argparse
from PIL import Image
from sophon import sail
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, PretrainedConfig, Qwen2VLConfig
from qwen_vl_utils import process_vision_info
import json
import os
import torch
from typing import Optional, Tuple
import onnxruntime as ort
from typing import List
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Preprocess the images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_rope_index(
        config,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

def get_position_ids(messages, processor, config, text="Describe this image and tell a story."):
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # SEQ_LENGTH = config['max_position_embeddings']
    SEQ_LENGTH = 2048
    # SEQ_LENGTH = self.SEQLEN
    if SEQ_LENGTH <= inputs.input_ids.shape[-1]:
        raise ValueError(
                f"The input_length must be shorter than model's seq_length (got `input_length`: {inputs.input_ids.shape[-1]}"
                f" and `seq_length`: {SEQ_LENGTH})."
            )
    breakpoint()
    input_ids = inputs.input_ids
    if "pixel_values" in inputs:
        pixel_values = inputs.pixel_values
    else:
        pixel_values_videos = inputs.pixel_values_videos
    if "image_grid_thw" in inputs:
        image_grid_thw = inputs.image_grid_thw
        video_grid_thw = None
    else:
        image_grid_thw = None
        video_grid_thw = inputs.video_grid_thw
    input_ids_prefill = torch.zeros(input_ids.shape[0], SEQ_LENGTH).to(torch.int32)
    input_ids_prefill[:, :input_ids.shape[-1]] = input_ids
    attention_mask_prefill = torch.zeros(inputs.attention_mask.shape[0], SEQ_LENGTH)
    attention_mask_prefill[:, :input_ids.shape[-1]] = inputs.attention_mask
    pretrained_config = PretrainedConfig(**config)
    with open('config.json', 'r') as json_file:
        config_dict = json.load(json_file)
        loaded_config = Qwen2VLConfig(**config_dict)
        # print(loaded_config)
    image_mask = (input_ids_prefill == loaded_config.image_token_id)
    true_indices = torch.nonzero(image_mask, as_tuple=True)[1]

    if true_indices.numel() > 0:
        first_true_index = true_indices[0].item()
    else:
        first_true_index = None
    

    # config = Qwen2VLConfig(
    #     # vocab_size=151936,
    #     # hidden_size=1536,
    #     # num_hidden_layers=28,
    #     # num_attention_heads=12,
    #     # intermediate_size=8960,
    #     # max_position_embeddings=32768,
    #     # 添加其他必要的参数
    #     **config
    # )

    # 创建模型实例
    # model = Qwen2VLForConditionalGeneration(loaded_config)
    # position_ids, _ = Qwen2VLForConditionalGeneration(loaded_config).get_rope_index(
    #     input_ids_prefill, image_grid_thw, None, attention_mask_prefill
    # )
    breakpoint()
    position_ids, _ = get_rope_index(loaded_config,
        input_ids_prefill, image_grid_thw, video_grid_thw, attention_mask_prefill
    )

    return position_ids, inputs, first_true_index

class VisionRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2VL():

    def __init__(self, **kwargs):
        # devid
        """
        self.dev_id = kwargs.get("dev_id", 0)
        self.handle = sail.Handle(self.dev_id)
        self.net = sail.EngineLLM(kwargs["bmodel_path"], [self.dev_id])

        # graph
        self.graph_names = self.net.get_graph_names()

        # initialize qwen parameters
        self.NUM_LAYERS = 0
        for graph_name in self.graph_names:
            if "block_cache_" in graph_name:
                self.NUM_LAYERS += 1
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", 0)
        self.vit_input_shape = self.net.get_input_shape("qwen_vit", 0)
        _, self.SEQLEN, self.HIDDEN_SIZE = self.first_hidden_input_shape

        # initialize net name
        self.is_greedy_sample = True
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_sample = "greedy_head" if self.is_greedy_sample else ""
        self.name_penalty = "penalty_sample_head" if self.is_greedy_sample else ""
        self.name_vit = "vit"

        # initialize tensors (inputs & outputs)
        # forward_first: embedding_tensor

        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1,  self.HIDDEN_SIZE], False)

        # forward_first: hidden_state, position_id_tensor and attention_mask_tensor
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state, position_id_tensor and attention_mask_tensor
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor and value_tensor
        self.past_key_input = []
        self.past_value_input = []

        for _ in range(self.NUM_LAYERS): 
            self.past_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 1))
            self.past_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 2))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        # sample tensor
        self.sample_input = self.init_sail_tensor(self.name_sample, 0)
        self.sample_output = self.init_sail_tensor(self.name_sample, 0, None, False)
        
        # vit tensor
        self.vit_hidden_states = self.init_sail_tensor(self.name_vit, 0)
        self.vit_rotary_pos_emb = self.init_sail_tensor(self.name_vit, 1)
        self.vit_attention_mask = self.init_sail_tensor(self.name_vit, 2)
        self.vit_output = self.init_sail_tensor(self.name_vit, 0, None, False)
        """
        self.visual_onnx = ort.InferenceSession('models/onnx/vit/vision_transformer.onnx')
        onnx_input_names = self.visual_onnx.get_inputs()
        self.vit_input_names = []
        for onnx_input_name in onnx_input_names:
            self.vit_input_names.append(onnx_input_name.name)
        self.vit_output_names = []
        onnx_output_names = self.visual_onnx.get_outputs()
        for onnx_output_name in onnx_output_names:
            self.vit_output_names.append(onnx_output_name.name)

        # Qwen2VLModel
        self.NUM_LAYERS = 28
        self.embed_tokens = torch.jit.load("models/onnx/llm/embedding.pt")

        """
        self.layers_onnx = []
        for layer_id in range(self.NUM_LAYERS):
            layer_onnx = ort.InferenceSession(f'models/onnx/llm/block_{layer_id}.onnx')
            onnx_input_names = layer_onnx.get_inputs()
            input_names = []
            for onnx_input_name in onnx_input_names:
                input_names.append(onnx_input_name.name)
            output_names = []
            onnx_output_names = layer_onnx.get_outputs()
            for onnx_output_name in onnx_output_names:
                output_names.append(onnx_output_name.name)
            self.layers_onnx.append((layer_onnx, input_names, output_names))
        """
        
        self.layers_cache_onnx = []
        for layer_id in range(self.NUM_LAYERS):
            layer_onnx = ort.InferenceSession(f'models/onnx/llm/block_cache_{layer_id}.onnx')
            onnx_input_names = layer_onnx.get_inputs()
            input_names = []
            for onnx_input_name in onnx_input_names:
                input_names.append(onnx_input_name.name)
            output_names = []
            onnx_output_names = layer_onnx.get_outputs()
            for onnx_output_name in onnx_output_names:
                output_names.append(onnx_output_name.name)
            self.layers_cache_onnx.append((layer_onnx, input_names, output_names))
        self.head_with_topk = torch.jit.load("models/onnx/llm/lm_head_with_topk.pt")
        self.SEQLEN = 2048
        self.block_head_num = 4
        self.block_hidden_size = 4 * 128
        self.spatial_merge_size = 2
        self.vit_head_dim = 80
        self.rotary_pos_emb = VisionRotaryEmbedding(self.vit_head_dim // 2)
        self.video_token_id = 151656
        self.step = 0
        self.token_pos_length = 0
        self.past_ks = None
        self.past_vs = None

        self.processor = AutoProcessor.from_pretrained(args.processor_path,
                                                       trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                       trust_remote_code=True)
        with open(args.config, 'r') as f:
            self.config = json.load(f)

        # load model
        # self.model = chat.Qwen2VL()
        # self.model.init(self.device, args.model_path)
        # self.model.generation_mode = args.generation_mode
        # self.POSITION_IDS, _, _ = get_position_ids(processor=self.processor, config=self.config)
        # self.SEQLEN = self.model.SEQLEN
        # self.ID_EOS = self.tokenizer.eos_token_id
        # self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>") 
        self.ID_END = 151643
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor_idx) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor_idx)
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor_idx) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor_idx)
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor

    def init_kv_cache(self, batch_size):
        if self.past_ks is not None:
            del self.past_ks
        if self.past_vs is not None:
            del self.past_vs
        self.past_ks = [np.zeros((batch_size, self.SEQLEN, self.block_head_num, self.block_hidden_size//self.block_head_num), dtype=np.float32) for _ in range(self.NUM_LAYERS)]
        self.past_vs = [np.zeros((batch_size, self.SEQLEN, self.block_head_num, self.block_hidden_size//self.block_head_num), dtype=np.float32) for _ in range(self.NUM_LAYERS)]

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i][0], grid_thw[i][1], grid_thw[i][2]
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_vision_mask(self, grid_thw):
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seq_length = cu_seqlens.max()

        attention_mask = torch.zeros([1, seq_length, seq_length], device="cpu", dtype=torch.bool)
        for i in range(1, cu_seqlens.shape[0]):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        return attention_mask

    # inference for the first token
    def forward_first(
            self, 
            tokens: np.ndarray, 
            position_ids: np.ndarray, 
            pixel_values,
            pixel_values_videos: np.ndarray, 
            image_grid_thw,
            video_grid_thw):
        self.init_kv_cache(tokens.shape[0])
        input_ids = np.zeros((tokens.shape[0], self.SEQLEN), dtype=np.int32)
        input_ids[:, :min(self.SEQLEN, tokens.shape[1])] = tokens
        token_length = tokens.shape[1]
        
        # embedding
        inputs_embeds = self.embed_tokens(torch.from_numpy(input_ids)).numpy()

        # ViT Inference
        pixel_values_videos = pixel_values_videos.astype(np.float32)
        rotary_pos_emb = self.rot_pos_emb(torch.from_numpy(video_grid_thw)).numpy()
        visual_attention_mask = self.get_vision_mask(torch.from_numpy(video_grid_thw))
        visual_attention_mask_fp = torch.zeros((visual_attention_mask.shape), dtype=torch.float32)
        visual_attention_mask = visual_attention_mask_fp.masked_fill(visual_attention_mask.logical_not(), -1000000).numpy()
        video_embeds = self.visual_onnx.run(self.vit_output_names, input_feed={self.vit_input_names[0]:pixel_values_videos, self.vit_input_names[1]:rotary_pos_emb, self.vit_input_names[2]:visual_attention_mask})[0]
        """
        real_len = pixel_values_videos.shape[0]
        self.vit_input["hidden_states"].update_data(F.pad(pixel_values_videos, (0, 0, 0, self.vit_input["hidden_states"].shape()[0] - pixel_values_videos.shape[0]), value=0).detach().numpy())
        self.vit_input["rotary_pos_emb"].update_data(F.pad(rotary_pos_emb, (0, 0, 0, self.vit_input["hidden_states"].shape()[0] - pixel_values_videos.shape[0]), value=0).detach().numpy())
        visual_attention_mask_fp = torch.zeros((1, self.vit_input["hidden_states"].shape()[0], self.vit_input["hidden_states"].shape()[0]), dtype=torch.float32)
        visual_attention_mask_fp = visual_attention_mask_fp.masked_fill(F.pad(visual_attention_mask, (0, self.vit_input["hidden_states"].shape()[0] - pixel_values_videos.shape[0], 0, self.vit_input["hidden_states"].shape()[0] - pixel_values_videos.shape[0]), value=0).logical_not(), -1000)
        self.vit_input["visual_attention_mask"].update_data(visual_attention_mask_fp.detach().numpy())
        self.visual_bmodel.process(self.visual_bmodel.get_graph_names()[0], self.vit_input, self.vit_output)
        video_embeds = torch.from_numpy(self.vit_output["output_Gemm_f32"].asnumpy())[:real_len//4]
        """
        video_mask = input_ids == self.video_token_id
        inputs_embeds[video_mask] = video_embeds

        # concatenate text embedding and image embedding
        # _, begin, end = img_pos[0]
        # img_pad_len = end-begin-1
        # self.first_embed_output["data"].sync_d2d(self.vit_output["data"], 0, int(begin*self.HIDDEN_SIZE), int(img_pad_len*self.HIDDEN_SIZE))

        # blocks
        # self.first_hidden_tensor = self.first_embed_output["data"]
        # self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        # self.first_pid["data"].update_data(position_id.reshape(self.first_pid["shape"]))
        # self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"]))) # set bf16 in the future.
        # self.first_attention["data"].update_data(attention_mask.reshape(self.first_attention["shape"]).view(np.uint16))

        # input_blocks_tensors = {0: self.first_hidden_tensor, 
        #                         1: self.first_pid["data"], 
        #                         2: self.first_attention["data"]}

        # Transformer Block Inference
        # for i in range(self.NUM_LAYERS):        
        #     output_blocks_tensors = {0: self.first_hidden_tensor,
        #                              1: self.past_key_output[i]["data"],
        #                              2: self.past_value_output[i]["data"]}
        #     self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        causal_mask = torch.zeros(self.SEQLEN, self.SEQLEN, dtype=torch.float)
        temp_mask = torch.ones(self.SEQLEN, self.SEQLEN, dtype=torch.bool).tril(diagonal=0)
        causal_mask.masked_fill_(temp_mask.logical_not(), float("-100000"))
        causal_mask[token_length:] = float("-100000")
        causal_mask[:, token_length:] = float("-100000")
        hidden_states = np.pad(inputs_embeds, ((0, 0), (0, self.SEQLEN - inputs_embeds.shape[1]), (0, 0)), 'constant', constant_values=0)
        padded_position_ids = np.zeros((3, position_ids.shape[1], self.SEQLEN), dtype=int) 
        padded_position_ids[:, :, :position_ids.shape[-1]] = position_ids
        for i in range(self.NUM_LAYERS):
            # save cpu mem
            layer_onnx = ort.InferenceSession(f'models/onnx/llm/block_{i}.onnx')
            onnx_input_names = layer_onnx.get_inputs()
            input_names = []
            for onnx_input_name in onnx_input_names:
                input_names.append(onnx_input_name.name)
            output_names = []
            onnx_output_names = layer_onnx.get_outputs()
            for onnx_output_name in onnx_output_names:
                output_names.append(onnx_output_name.name)
            layer_outputs = layer_onnx.run(output_names, \
                input_feed={input_names[0]:hidden_states, input_names[1]:causal_mask.detach().numpy(), input_names[2]:padded_position_ids})
            # layer_outputs = self.layers_onnx[i][0].run(self.layers_onnx[i][2], \
            #     input_feed={self.layers_onnx[i][1][0]:hidden_states, self.layers_onnx[i][1][1]:causal_mask.detach().numpy(), self.layers_onnx[i][1][2]:padded_position_ids})
            hidden_states, first_k, first_v = layer_outputs
            self.past_ks[i][:, :token_length] = first_k[:, :token_length]
            self.past_vs[i][:, :token_length] = first_v[:, :token_length]
            del first_k
            del first_v

        # get the last token info as Lm head input
        # copy_len = self.first_hidden_tensor.shape()[-1]
        # self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
        #                               (self.token_length-1)* copy_len,  
        #                               0, 
        #                               copy_len)
        
        # input_lm_tensors = {0: self.lm_input["data"]}
        # output_lm_tensors = {0: self.lm_output["data"]}
        
        # Lm_head Inference
        # self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        
        # sample
        # input_sample_tensor = {0: self.lm_output["data"]}
        # output_sample_tensor = {0: self.sample_output["data"]}
        # self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)

        # return int(self.sample_output["data"].asnumpy()[0][0])
        self.step += token_length
        self.token_pos_length = position_ids.max() + 1
        self.last_id = self.head_with_topk(torch.from_numpy(hidden_states)).numpy()[:, :self.step][:, -1].item()
        return self.last_id

    # The following tokens prediction
    def forward_next(self):
        causal_mask = np.zeros((1, self.SEQLEN+1), dtype=np.float32)
        causal_mask[:, self.step:-1] = float("-100000")
        position_ids = np.array([self.token_pos_length]*3, dtype=int).reshape(3, 1, 1)

        # embedding
        # self.next_embed_input["data"] = self.sample_output["data"]
        # self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        # input_embed_tensors = {0: self.next_embed_input["data"]}
        # output_embed_tensors = {0: self.next_embed_output["data"]}
        # Embedding Layer Inference
        # self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)
        hidden_states = self.embed_tokens(torch.tensor([[self.last_id]], dtype=torch.int32)).numpy()
        
        # blocks
        # self.next_pid["data"].update_data(position_id.reshape(self.next_pid["shape"]))
        # self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))
        # self.next_attention["data"].update_data(attention_mask.reshape(self.next_attention["shape"]).view(np.uint16))

        # self.next_hidden_tensor = self.next_embed_output["data"]
        # self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        # Transformer Block Inference
        # for i in range(self.NUM_LAYERS):
        #     inputs_block_cache_tensors = {0: self.next_hidden_tensor, 
        #                                   1: self.next_pid["data"], 
        #                                   2: self.next_attention["data"], 
        #                                   3: self.past_key_output[i]["data"], 
        #                                   4: self.past_value_output[i]["data"]}
        #     outputs_block_cache_tensors = {0: self.next_hidden_tensor,
        #                                    1: self.present_key["data"],
        #                                    2: self.present_value["data"]}
        #     self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        #     # update kv_cache()
        #     unit_size = self.present_key["shape"][-1]*self.present_key["shape"][-2]
        #     self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
        #     self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)
        for i in range(self.NUM_LAYERS):
            layer_outputs = self.layers_cache_onnx[i][0].run(self.layers_cache_onnx[i][2], \
                input_feed={self.layers_cache_onnx[i][1][0]:hidden_states, \
                self.layers_cache_onnx[i][1][1]:causal_mask, self.layers_cache_onnx[i][1][2]:position_ids, \
                self.layers_cache_onnx[i][1][3]:self.past_ks[i], \
                self.layers_cache_onnx[i][1][4]:self.past_vs[i]})
            hidden_states, present_k, present_v = layer_outputs
            self.past_ks[i][:, self.step:self.step+1] = present_k
            self.past_vs[i][:, self.step:self.step+1] = present_v
        
        # self.lm_input_tensor = self.next_hidden_tensor
        # self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        # input_lm_tensors = {0: self.lm_input_tensor}
        # output_lm_tensors = {0: self.lm_output["data"]}
        
        # Lm_head Inference
        # self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        # sample
        # input_sample_tensor = {0: self.lm_output["data"]}
        # output_sample_tensor = {0: self.sample_output["data"]}
        # self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)

        # return int(self.sample_output["data"].asnumpy()[0][0])
        self.step += 1
        self.token_pos_length += 1
        self.last_id = self.head_with_topk(torch.from_numpy(hidden_states)).numpy().item()
        return self.last_id

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "carvana_video.mp4",
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        messages = [messages]
        # Stop Chatting with "exit" input
        while True:
            # self.input_str = input("\nQuestion: ")
            # # Quit
            # if self.input_str in ["exit", "q", "quit"]:
            #     break
            # self.image_str = input("\nImage Path: ")
            # print("\nAnswer:")
            # if self.image_str:
            #     if not os.path.exists(self.image_str):
            #         print("Can't find image: {}".format(self.image_str))
            #         continue

            # self.encode()
            # self.POSITION_IDS, inputs, image_offset = get_position_ids(processor=self.processor, config=self.config, image_path=self.image_str, text=self.input_str)
            self.POSITION_IDS, inputs, image_offset = get_position_ids(messages=messages, processor=self.processor, config=self.config)
            # messages = [
            #     {
            #         "role": "user",cd 
            #         "content": [
            #             {
            #                 "type": "image",
            #                 "image": self.image_str,
            #             },
            #             {"type": "text", "text": self.input_str},
            #         ],
            #     }
            # ]
            # text = processor.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # image_inputs, video_inputs = process_vision_info(messages)
            # inputs = processor(
            #     text=[text],
            #     images=image_inputs,
            #     videos=video_inputs,
            #     padding=True,
            #     return_tensors="pt",
            # )
            # config = origin_model.config

            # position_ids, _ = Qwen2VLForConditionalGeneration(config).get_rope_index(
            #     input_ids_prefill, image_grid_thw, None, attention_mask_prefill
            # )
            position_ids = self.POSITION_IDS
            # Chat
            first_start = time.time()
            breakpoint()
            token = self.forward_first(inputs.input_ids.numpy(), position_ids.numpy(), None, inputs.pixel_values_videos.numpy(),
                                             None, inputs.video_grid_thw.numpy())
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.step < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens,
                                             skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode(
                            [token, token],
                            skip_special_tokens=True)[len(pre_word):]
                    text += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                token = self.forward_next()
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen2VL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--bmodel_path',
                        type=str,
                        default="",
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer_path',
                        type=str,
                        default="./token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-p',
                        '--processor_path',
                        type=str,
                        default="./processor_config",
                        help='path to the processor file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="./config.json",
                        help='path to the model config file')
    parser.add_argument('-d', '--dev_id', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-g',
                        '--generation_mode',
                        type=str,
                        choices=["greedy", "penalty_sample"],
                        default="greedy",
                        help='mode for generating next token')
    args = parser.parse_args()
    main(args)
