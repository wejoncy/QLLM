from packaging import version
import time
from typing import Dict, List, Literal, Optional, Union
import torch
from torch import nn
from pydantic import BaseModel, Field

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True

def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


def loop_in_chat_completion(model_type: str, tokenizer:str, llm:nn.Module):
    llm = llm.cuda()
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        except:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    request = ChatCompletionRequest(
        model=model_type,
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }])
    print("Enter 'exit/bye' to quit anytime\n. let's have a conversation!")
    while True:
        user_input = input("user: ")
        if user_input in ["exit", "bye"]:
            print("Bye!")
            break
        request.messages.append({
            "role": "user",
            "content": user_input
        })
        prompt = get_gen_prompt(request)
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        output = llm.generate(**inputs, max_length=100)
        response = tokenizer.batch_decode(output)[0]
        print("Response: ", response)
        print(f"Time: {time.time() - start:.2f}s")
        request.messages.append({
            "role": "assistant",
            "content": response
        })
