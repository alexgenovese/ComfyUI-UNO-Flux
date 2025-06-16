# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor, nn
import torch
import os
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)
from safetensors.torch import load_file as load_sft


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = "clip" in version.lower()
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # Check if version is a safetensors file
        is_safetensors_file = version.endswith('.safetensors') and os.path.isfile(version)
        
        if is_safetensors_file:
            # For safetensors files, we need to use a base model and load weights
            if self.is_clip:
                # Use a base CLIP model
                base_model = "openai/clip-vit-large-patch14"
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base_model, max_length=max_length, **hf_kwargs)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(base_model, **hf_kwargs)
                
                # Load safetensors weights
                state_dict = load_sft(version)
                self.hf_module.load_state_dict(state_dict, strict=False)
            else:
                # Use a base T5 model
                base_model = "t5-base"
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(base_model, max_length=max_length, **hf_kwargs)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(base_model, **hf_kwargs)
                
                # Load safetensors weights
                state_dict = load_sft(version)
                self.hf_module.load_state_dict(state_dict, strict=False)
        else:
            # Original functionality for model identifiers or directories
            if self.is_clip:
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length, **hf_kwargs)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
            else:
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length, **hf_kwargs)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
