from texttable import Texttable
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:  # NOQA
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # NOQA

import torch.nn as nn
import torch
from pathlib import Path

import sys
sys.path.append(os.path.dirname(__file__))  # NOQA
sys.path.append(os.getcwd())   # NOQA

import loralib as lora
import model_quantization_base as model_quantization_base

NEED_CHECK_PACK = False


class MPT(model_quantization_base.ModelQuantizationBase):
    def __init__(self):
        super().__init__()
        self.argv_user = None
        self.quant_layers = [torch.nn.Linear, lora.MergedLinear, lora.Linear]

    def get_torch_model(self, args):
        argv_user = self.argv_user
        if 'ckpt/mpt-' not in argv_user[argv_user.index('--model_name_or_path')+1]:
            lora_ind = argv_user.index('--use_lora')
            argv_user[lora_ind+1] = 'False'

        import examples_ads
        from examples_ads import run_mpt_prompt
        argv_user.insert(0, run_mpt_prompt.__file__)
        argv_back = sys.argv
        sys.argv = argv_user

        os.environ['init_device'] = "cpu"
        model, data_sets = run_mpt_prompt.main(True)
        new_data = []
        for idx, indata in enumerate(data_sets):
            if idx >= args.nsamples:
                break
            input_ = (torch.tensor([indata["input_ids"]]), torch.tensor([indata["attention_mask"]]))
            new_data.append(input_)
        return model.half(), new_data

    @torch.no_grad()
    def eval_model(self, model, dev):
        print('Evaluating ...')
        sys.argv = self.argv_user
        import examples_ads
        from examples_ads import run_mpt_prompt
        run_mpt_prompt.main(quant_model=model.to(dev))

    def export_onnx(self, model, onnx_path, sample_inputs: tuple):
        model = self.pipeline_to_multiple_gpu(model, [torch.device(i)
                                              for i in range(torch.cuda.device_count())], sample_inputs)
        # model = model.cpu().float()
        # model = model.cuda()
        os.environ["export_onnx"] = "1"
        from pathlib import Path
        import shutil
        onnx_path = Path(onnx_path).absolute()
        assert onnx_path.suffix == '.onnx'
        inputs = {'input_ids': sample_inputs[0].to(model.device), "attention_mask": sample_inputs[1].to(model.device)}
        onnx_filepath_export_multi_files_tmp = onnx_path.parent/'tmp/tmp.onnx'
        onnx_filepath_export_multi_files_tmp.parent.exists() and shutil.rmtree(onnx_filepath_export_multi_files_tmp.parent)
        os.makedirs(onnx_filepath_export_multi_files_tmp.parent)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        past_key_values = None
        onnx_inputs = (input_ids, past_key_values, attention_mask, None, None, None, True, False, False, False)
        onnx_inp_names = ("input_ids", "attention_mask")
        onnx_out_names = ("logits",)
        onnx_dynamic_axes = {"input_ids": {0: 'batch_size', 1: "seq_len"},
                             "attention_mask": {0: 'batch_size', 1: "seq_len"}}
        torch.onnx.export(model=model, args=onnx_inputs, f=str(onnx_filepath_export_multi_files_tmp), verbose=False, opset_version=16,
                          input_names=onnx_inp_names, output_names=onnx_out_names, dynamic_axes=onnx_dynamic_axes)
        import onnx
        onnx_model = onnx.load(str(onnx_filepath_export_multi_files_tmp))

        onnx_path.exists() and onnx_path.unlink()
        (onnx_path.parent/'mpt_ext.data').exists() and (onnx_path.parent/'mpt_ext.data').unlink()
        onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="mpt_ext.data", size_threshold=1024, convert_attribute=False)

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


if __name__ == '__main__':
    mpt_quanter = MPT()
    parser = mpt_quanter.define_basic_args()
    parser.add_argument('--forward_args', type=str, default=None, help='args for run_prompts_mpt.py')

    args = parser.parse_args()
    mpt_quanter.process_forward_args(args)
    if args.load:
        mpt_quanter.argv_user[mpt_quanter.argv_user.index('--model_name_or_path')+1] = os.path.abspath(args.load)

    mpt_quanter.run(args)
