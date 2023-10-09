import time
import torch
try:
    import fastchat
    #from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.conversation import get_conv_template, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    from fastchat.model.model_adapter import get_generate_stream_function
    from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length

    _fastchat_available = False
except ImportError:
    _fastchat_available = False
from .conversation import get_conv
from .chatio import ChatIO, SimpleChatIO
from .generation import generate_stream




def chat_loop(
    model,
    tokenizer,
    max_new_tokens: int = 512,
    generate_stream_func = generate_stream,
    generate_func = None,
    chatio: ChatIO = None,
    debug: bool = False,
    echo: bool = False,
):
    if _fastchat_available:
        return chat_loop_v2(model, tokenizer)
    model_type = str(type(model)).lower()
    assert "llama" in model_type, 'only support llama model.'
    assert generate_stream_func is not None or generate_func is not None, 'should set generate function.'

    # Set context length
    context_len = model.config.max_position_embeddings

    # Chat
    def new_chat():
        conv = get_conv("llama2")
        return conv

    if chatio is None:
        chatio = SimpleChatIO(echo=echo)

    conv = None

    while True:
        if not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break

        if inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        chatio.prompt_for_output(conv.roles[1])
        if generate_stream_func is not None:
            output_stream = generate_stream_func(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                context_len=context_len,
            )
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.update_last_message(outputs.strip())
        else:
            outputs = generate_func(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                context_len=context_len,
            )
            outputs = chatio.output(outputs)
            t = time.time()
            duration = time.time() - t
            conv.update_last_message(outputs.strip())

        if debug:
            num_tokens = len(tokenizer.encode(outputs))
            msg = {
                "conv_template": conv.name,
                "prompt": prompt,
                "outputs": outputs,
                "speed (token/s)": round(num_tokens / duration, 2),
            }
            print(f"\n{msg}\n")


class SimpleChatIO_v2(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
            elif len(output_text[-1]) > 10:
                break
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)

def chat_loop_v2(
    model:torch.nn.Module,
    tokenizer,
    temperature: float = 0.7,
    repetition_penalty: float=1.0,
    max_new_tokens: int=512,
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,
    conv_template:str = None,
):
    device = next(model.parameters()).device
    chatio = SimpleChatIO_v2()
    generate_stream_func = get_generate_stream_function(model, "")

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)

    # Chat
    model_path = model.config.name_or_path
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        conv_system_msg = None
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    def reload_conv(conv):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    while True:
        if not history or not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if is_codet5p:  # codet5p is a code completion model.
            prompt = inp

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        try:
            chatio.prompt_for_output(conv.roles[1])
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.update_last_message(outputs.strip())

            if debug:
                num_tokens = len(tokenizer.encode(outputs))
                msg = {
                    #"conv_template": conv.name,
                    #"prompt": prompt,
                    #"outputs": outputs,
                    "speed (token/s)": round(num_tokens / duration, 2),
                }
                print(f"\n{msg}\n")

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)