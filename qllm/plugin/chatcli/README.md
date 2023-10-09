# Chat CLI
This code is inspired by FastChat, and modified for easy to use in CLI mode.

## Usage
The main function is `chat_loop`, an example of using this ChatCLI is:

```
from chatcli import chat_loop

# type !!exit to exit the chat loop
# type !!reset to reset the conversation
chat_loop(llama_baseline, tokenizer, echo=True)
```

There are two command: 
* `!!exit` means to exit chat loop
* `!!reset` means to reset the chat context and start a new conversation.

## Chat loop
Please refer to the `chat_loop` function:

```
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
```

Mostly, you only need to input `model` and `tokenizer`, leave the other args as default.

The `generete_stream_func` is used to stream output tokens, while the `generete_func` will wait all tokens are generated and then output them. These two functions are responsible of doing generation work, if you need to modify the generation process, please refer to `generation.py` to modify these functions.

The `echo` option is used for jupyter, because jupyter is not echo user input to the screen.

When set `echo=True`, it will print user input to the screen, then it's better to see your input history.

## Conversation
We use a class `Conversation` to save the chat context of one conversation.

For different model, the conversation may be different, for example using different role name.

Currently we only include `Llama2` conversation.
