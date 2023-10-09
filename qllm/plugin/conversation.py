import torch
from torch import nn
from .chatcli import chat_loop, generate, generate_stream


def loop_in_chat_completion(tokenizer:str, llm:nn.Module):
    llm = llm.cuda().half()
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        except:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    chat_loop(
        llm,
        tokenizer,
        generate_func=generate,
        max_new_tokens=512,
    )