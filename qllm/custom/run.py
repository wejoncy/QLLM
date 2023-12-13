from texttable import Texttable
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:  # NOQA
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # NOQA

import warnings
import torch.nn as nn
import torch
from pathlib import Path

from ..utils.logger import get_logger
from ..model_quantization_base import ModelQuantizationBase

logger = get_logger()

import sys
sys.path.append(os.path.dirname(__file__))  # NOQA
sys.path.append(os.getcwd())   # NOQA

import loralib as lora

NEED_CHECK_PACK = False


class CustomModel(ModelQuantizationBase):
    def __init__(self):
        super().__init__()
        self.argv_user = None
        self.quant_layers = [torch.nn.Linear, lora.MergedLinear, lora.Linear]
        self.datsets = None

    def get_torch_model(self, args, dev):
        argv_user = self.argv_user
        if 'ckpt/mpt-' not in argv_user[argv_user.index('--model_name_or_path')+1]:
            lora_ind = argv_user.index('--use_lora')
            argv_user[lora_ind+1] = 'False'
        try:
            import examples_ads
            from examples_ads import run_mpt_prompt
        except:
            logger.error(f"Do you forget to run the command in the root directory of the project? `examples_ads` is not find in {os.getcwd()},\
please switch to the right directory and try again")
            raise
        argv_user.insert(0, run_mpt_prompt.__file__)
        argv_back = sys.argv
        sys.argv = argv_user

        os.environ['init_device'] = "cpu"
        model, data_sets = run_mpt_prompt.main(True)
        new_data = []
        for idx, indata in enumerate(data_sets):
            if idx >= args.nsamples:
                break
            input_ = (torch.tensor([indata["input_ids"]]),
                      torch.tensor([indata["attention_mask"]]))
            new_data.append(input_)
        self.datsets = new_data
        return model.half()

    def get_datasets(self, args):
        cache_dir = Path(
            f"/tmp/qllm_v1/{args.model.replace(' ','_')}_{args.dataset}_dataloader.pt")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"loading dataset from {args.dataset}")

        if self.datsets is not None:
            torch.save(self.datsets, str(cache_dir))
            return self.datsets

        if cache_dir.exists():
            logger.info(f"found cached dataloader in {cache_dir}")
            dataloader = torch.load(cache_dir)

        return dataloader

    @torch.no_grad()
    def eval_model(self, model, dev):
        logger.info('Evaluating ...')
        sys.argv = self.argv_user
        import examples_ads
        from examples_ads import run_llama_prompt
        run_llama_prompt.main(quant_model=model.to(dev))


    def process_forward_args(self, args):
        argv_user = args.forward_args
        import re
        key_with_space = re.findall(r'(".*"|\'.*\')', argv_user)
        argv_map = {}
        for idx, v in enumerate(key_with_space):
            argv_user = re.sub(v, f'____{idx}___', argv_user)
            argv_map[f'____{idx}___'] = v.strip('"')
        argv_user = argv_user.split(' ')
        argv_user = list(filter(None, argv_user))
        idx = 0
        for i in range(len(argv_user)):
            if argv_user[i] == f'____{idx}___':
                argv_user[i] = argv_map[f'____{idx}___']
                idx += 1
        self.argv_user = argv_user


    def export_onnx(self, model: torch.nn.Module, onnx_path_str: str, sample_inputs: tuple, with_past: bool = False, args=None):
        try:
            import onnxruntime
            from packaging import version
            assert version.parse(onnxruntime.__version__) >= version.parse('1.17.0')
            assert version.parse(torch.__version__) >= version.parse('2.0.0')
            return super().export_onnx(model, onnx_path_str, sample_inputs, with_past, args)
        except:
            warnings.warn('this exporter will be deprecated, please upgrade to torch 2.1.0+ and onnxruntime 1.17+',
                      DeprecationWarning, stacklevel=2)
        #model = self.pipeline_to_multiple_gpu(model, [torch.device(i)
        #                                            for i in range(torch.cuda.device_count())], sample_inputs)
        # model = model.cpu().float()
        model = model.cuda()
        os.environ["export_onnx"] = "1"
        from pathlib import Path
        import shutil
        onnx_path = Path(onnx_path_str).absolute()
        assert onnx_path.suffix == '.onnx'
        inputs = {'input_ids': sample_inputs[0].to(
            model.device), "attention_mask": sample_inputs[1].to(model.device)}
        onnx_filepath_export_multi_files_tmp = onnx_path.parent/'tmp/tmp.onnx'
        onnx_filepath_export_multi_files_tmp.parent.exists() and shutil.rmtree(
            onnx_filepath_export_multi_files_tmp.parent)
        os.makedirs(onnx_filepath_export_multi_files_tmp.parent)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        past_key_values = None
        onnx_inputs = (input_ids, past_key_values, attention_mask,
                    None, None, None, True, False, False, False)
        onnx_inp_names = ("input_ids", "attention_mask")
        onnx_out_names = ("logits",)
        onnx_dynamic_axes = {"input_ids": {0: 'batch_size', 1: "seq_len"},
                            "attention_mask": {0: 'batch_size', 1: "seq_len"}}
        torch.onnx.export(model=model, args=onnx_inputs, f=str(onnx_filepath_export_multi_files_tmp), verbose=False, opset_version=16,
                        input_names=onnx_inp_names, output_names=onnx_out_names, dynamic_axes=onnx_dynamic_axes)
        import onnx
        onnx_model = onnx.load(str(onnx_filepath_export_multi_files_tmp))

        onnx_path.exists() and onnx_path.unlink()
        (onnx_path.parent/'model_ext.data').exists() and (onnx_path.parent /
                                                          'model_ext.data').unlink()
        onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="model_ext.data", size_threshold=1024, convert_attribute=False)


def main():
    from .. import run
    parser = run.define_basic_args()
    parser.add_argument('--forward_args', type=str,default=None, help='args for run_prompts_mpt.py')
    sys.argv = sys.argv + ["--model=./a"]
    args = parser.parse_args()

    mpt_quanter = CustomModel()
    mpt_quanter.process_forward_args(args)
    if args.load:
        mpt_quanter.argv_user[mpt_quanter.argv_user.index('--model_name_or_path')+1] = os.path.abspath(args.load)

    mpt_quanter.run(args)

if __name__ == '__main__':
    main()