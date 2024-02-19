# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import time

import numpy as np
import paddle

# import psutil
import pytest
from tokenizers import Tokenizer as HFTokenizer

from paddlenlp.transformers import AutoTokenizer

MODEL_NAME = "qwen/qwen-7b"


def measure_time_and_memory(func, *args, **kwargs):
    start_time = time.time()
    # process = psutil.Process()
    # start_memory = process.memory_info().rss / 1024 / 1024

    result = func(*args, **kwargs)

    end_time = time.time()
    # end_memory = process.memory_info().rss / 1024 / 1024

    execution_time = end_time - start_time
    memory_usage = None

    return result, execution_time, memory_usage


@pytest.fixture
def setup_inputs():
    # single_s = (
    #     "In the intricate tapestry of linguistic expression, the amalgamation of diverse syntactic structures, nuanced vocabulary,"
    #     "and convoluted clauses not only challenges the adeptness of tokenization algorithms but also underscores the formidable complexity inherent in natural language processing tasks."
    # )
    single_s = "自然语言处理（NLP）是一种人工智能技术，致力于使计算机能够理解、解释和生成人类语言。通过NLP，计算机可以处理和分析大量的自然语言数据，实现自动翻译、情感分析、语言生成等任务，为各种应用场景提供智能化解决方案，如智能客服、信息抽取和文本分类等。"
    return single_s


@pytest.fixture
def tokenizer_fast_hf():
    from transformers import AutoTokenizer

    fast_tokenizer_hf = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", use_fast=True, trust_remote_code=True)
    return fast_tokenizer_hf


@pytest.fixture
def tokenizer_fast():
    fast_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, from_slow=True)
    return fast_tokenizer


@pytest.fixture
def tokenizer_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def test_tokenizer_type(tokenizer_fast_hf, tokenizer_fast, tokenizer_base):
    # assert isinstance(tokenizer_fast_hf, HFTokenizer)
    assert isinstance(tokenizer_fast._tokenizer, HFTokenizer)
    assert not hasattr(tokenizer_base, "_tokenizer")
    # assert tokenizer_fast_hf.from_hub == "huggingface"
    assert tokenizer_fast.from_hub == tokenizer_base.from_hub


def test_tokenizer_cost(tokenizer_fast_hf, tokenizer_fast, tokenizer_base, setup_inputs):
    costs = []
    breakpoint()
    for tokenizer in ["tokenizer_fast_hf", "tokenizer_fast", "tokenizer_base"]:
        (
            _,
            _time,
            _memory,
        ) = measure_time_and_memory(eval(tokenizer), [setup_inputs] * 20000)
        costs.append({tokenizer: (_memory, _time)})
    print(costs)


def test_tokenizer_decode(tokenizer_fast_hf, tokenizer_fast, tokenizer_base, setup_inputs):
    token_hf = tokenizer_fast_hf(setup_inputs)
    token_fast = tokenizer_fast(setup_inputs)
    token_base = tokenizer_base(setup_inputs)
    breakpoint()
    assert token_hf["input_ids"] == token_fast["input_ids"] == token_base["input_ids"]


def test_output_type(tokenizer_fast, setup_inputs):
    isinstance(tokenizer_fast.encode(setup_inputs, return_tensors="pd")["input_ids"], paddle.Tensor)
    isinstance(tokenizer_fast.encode(setup_inputs, return_tensors="np")["input_ids"], np.ndarray)
