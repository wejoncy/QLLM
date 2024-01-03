from torch import nn
from .chatcli import chat_loop, generate


def loop_in_chat_completion(tokenizer, llm:nn.Module):
    llm = llm.cuda().half()

    chat_loop(
        llm,
        tokenizer,
        generate_func=generate,
        max_new_tokens=512,
    )