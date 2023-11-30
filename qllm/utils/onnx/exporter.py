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
    model = large_model_exporter.move_to_appropriate_device(model, sample_inputs_tp)

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

    onnx_model_name = "model_with_past.onnx"
    onnx_path_dec = onnx_path_enc.parent / onnx_model_name

    large_model_exporter.do_export_internal(
        model, onnx_io_tuple, onnx_inputs, onnx_path_dec, opset)

    onnx_path_one_for_all = onnx_path_enc.parent / "model_one_for_all.onnx"
    merge_decoders(onnx_path_enc, onnx_path_dec, save_path=onnx_path_one_for_all)
    return onnx_path_one_for_all
