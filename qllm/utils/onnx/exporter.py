import torch
from onnxruntime.transformers import large_model_exporter
from pathlib import Path

from ..logger import get_logger
from .merge_encoder_decoder import merge_decoders
logger = get_logger()


def export_onnx(model: torch.nn.Module, onnx_path_str: str, sample_inputs: tuple, with_past: bool = False, opset=16) -> Path:
    # since onnxruntime 1.7
    logger.info("Exporting onnx model ...")
    sample_inputs_tp = list(sample_inputs)
    if sample_inputs_tp[1] is None:
        sample_inputs_tp[1] = torch.ones_like(sample_inputs_tp[0])
    #FIXME: this is a workaround for the bug in onnxruntime 1.7
    move_to_device = large_model_exporter.move_to_appropriate_device if hasattr(
        large_model_exporter, "move_to_appropriate_device") else large_model_exporter.move_to_approprate_device
    model = move_to_device(model, sample_inputs_tp)

    sample_inputs = large_model_exporter.adapt_inputs_to_device(
        sample_inputs_tp, next(model.parameters()).device)

    # input_keys would be usesful if the model has some special inputs
    input_keys, onnx_inputs, past_key_value = large_model_exporter.retrieve_onnx_inputs(model, sample_inputs, with_past)

    onnx_io_tuple = large_model_exporter.fetch_onnx_inputs_outputs_name(model, onnx_inputs, input_keys, past_key_value, with_past, False)

    onnx_model_name = "model.onnx"
    onnx_path: Path = Path(onnx_path_str).absolute()
    onnx_path_enc = onnx_path / onnx_model_name if onnx_path.suffix != ".onnx" else onnx_path
    onnx_path_enc.parent.mkdir(parents=True, exist_ok=True)

    large_model_exporter.do_export_internal(
        model, onnx_io_tuple, onnx_inputs, onnx_path_enc, opset)
    if not with_past:
        return onnx_path_enc

    onnx_io_tuple = large_model_exporter.fetch_onnx_inputs_outputs_name(model, onnx_inputs, input_keys, past_key_value, with_past, True)
    # workaround for attention_mask
    onnx_inputs[1] = onnx_inputs[1].long()

    onnx_model_name = "model_with_past.onnx"
    onnx_path_dec = onnx_path_enc.parent / onnx_model_name

    large_model_exporter.do_export_internal(
        model, onnx_io_tuple, onnx_inputs, onnx_path_dec, opset)

    onnx_path_one_for_all = onnx_path_enc.parent / "model_one_for_all.onnx"
    merge_decoders(onnx_path_enc, onnx_path_dec, save_path=onnx_path_one_for_all)
    return onnx_path_one_for_all


def verify_correcness(model: torch.nn.Module, sample_inputs: tuple, onnx_model_path:str, with_past: bool,):
    import onnxruntime
    #import onnx_ops
    import numpy as np

    ref = model(sample_inputs[0].cuda(), torch.ones(sample_inputs[0].shape, dtype=torch.int64).cuda())

    mask = np.ones(sample_inputs[0].shape, dtype=np.int64)
    num_layers = model.config.num_hidden_layers
    inputs = {'input_ids': sample_inputs[0].cpu().numpy(), 'attention_mask': mask}
    if with_past:
        inputs['use_cache_branch'] = np.array([0], dtype=np.bool_)
        for i in range(num_layers):
            inputs[f'past_key_values.{i}.key'] = np.zeros(ref.past_key_values[0][0].shape, dtype=np.float16)
            inputs[f'past_key_values.{i}.value'] = np.zeros(ref.past_key_values[0][0].shape, dtype=np.float16)
    session_options = onnxruntime.SessionOptions()
    #session_options.register_custom_ops_library(onnx_ops.__file__)
    #onnx_path_str = Path(onnx_model_path).parent.absolute()
    #session = onnxruntime.InferenceSession(f'{onnx_path_str}/model.onnx', providers=['CUDAExecutionProvider'], sess_options=session_options)
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'], sess_options=session_options)
    outputs = session.run(None, inputs)
    err_1 = ref.logits.cpu().numpy() - outputs[0]

    if with_past:
        #session = onnxruntime.InferenceSession(f'{onnx_path_str}/model_with_past.onnx', providers=['CUDAExecutionProvider'], sess_options=session_options)
        mask = np.concatenate([mask, np.array([[1]])], axis=1)
        inputs = {'input_ids': np.array([[3]]), 'attention_mask': mask}
        
        inputs['use_cache_branch'] = np.array([1], dtype=np.bool_)
        for i in range(num_layers):
            inputs[f'past_key_values.{i}.key'] = ref.past_key_values[i][0].cpu().numpy()
            inputs[f'past_key_values.{i}.value'] = ref.past_key_values[i][1].cpu().numpy()
        outputs = session.run(None, inputs)

        ref = model(torch.tensor([[3]],device="cuda"), torch.from_numpy(mask).cuda(), past_key_values=ref.past_key_values)

    err = ref.logits.cpu().numpy() - outputs[0]
    print("max abs err:", np.abs(err).max(), np.abs(err_1).max(), "correctness check ",
            "" if np.abs(err).max() < 1e-2 else "not", " passed")