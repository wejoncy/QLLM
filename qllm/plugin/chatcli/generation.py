import time
import torch


def generate_stream(model, tokenizer, prompt: str, max_new_tokens:int, context_len: int, echo: bool=False, stream_interval=2):
    stop_token_ids = [model.config.eos_token_id]
    device = model.device

    inputs = tokenizer(prompt)

    lhs_tokens = torch.tensor(inputs.input_ids, dtype=torch.int64, device=device).unsqueeze(0)

    past_kvs = None
    output_ids = list(inputs.input_ids)
    input_echo_len = len(output_ids)

    # check max_new_tokens
    remain_tokens = context_len - input_echo_len
    max_new_tokens = min(remain_tokens, max_new_tokens)

    for i in range(max_new_tokens):
        with torch.no_grad():
            lhs_results = model(lhs_tokens, past_key_values=past_kvs, use_cache=True)
        
        logits = lhs_results.logits
        past_kvs = lhs_results.past_key_values

        # greedy search
        lhs_tokens = torch.argmax(
            lhs_results.logits[:, -1, :], dim=1, keepdim=True)

        token = lhs_tokens[0].item()
        output_ids.append(token)

        if token in stop_token_ids:
            stoped = True
        else:
            stoped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stoped:
            if echo:
                tmp_output_ids = output_ids
            else:
                tmp_output_ids = output_ids[input_echo_len:]

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            yield {
              'text': output,
            }

        if stoped:
            break

    yield {
      'text': output
    }


def generate(model, tokenizer, prompt: str, max_new_tokens:int, context_len: int, echo: bool=False):
    stop_token_ids = [model.config.eos_token_id]
    device = model.device

    inputs = tokenizer(prompt)

    lhs_tokens = torch.tensor(inputs.input_ids, dtype=torch.int64, device=device).unsqueeze(0)

    past_kvs = None
    output_ids = list(inputs.input_ids)
    input_echo_len = len(output_ids)

    # check max_new_tokens
    remain_tokens = context_len - input_echo_len
    max_new_tokens = min(remain_tokens, max_new_tokens)

    for i in range(max_new_tokens):
        with torch.no_grad():
            lhs_results = model(lhs_tokens, past_key_values=past_kvs, use_cache=True)
        
        logits = lhs_results.logits
        past_kvs = lhs_results.past_key_values

        # greedy search
        lhs_tokens = torch.argmax(
            lhs_results.logits[:, -1, :], dim=1, keepdim=True)

        token = lhs_tokens[0].item()
        output_ids.append(token)

        if token in stop_token_ids:
            stoped = True
        else:
            stoped = False

        if stoped:
            break

    if echo:
        tmp_output_ids = output_ids
    else:
        tmp_output_ids = output_ids[input_echo_len:]

    output = tokenizer.decode(
        tmp_output_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return {'text': output}
