from pathlib import Path
import tempfile
import re
import torch
from .. import utils
import os


def cur_user():
    try:
        return os.getlogin()
    except:  # noqa: E722
        return "root"  # in docker


def get_sample_datas_for_quantization(tokenizer_path, dataset, nsamples, seed, seqlen=2048):
    logger = utils.logger.get_logger()
    normlized_tokenizer = re.sub(r"[^0-9a-zA-Z_-]", "", tokenizer_path)
    named_hash = f"{normlized_tokenizer}_{dataset}_{nsamples}_{seqlen}_{seed}"
    cache_dir = Path(f"{tempfile.gettempdir()}/qllm_v{cur_user()}/_{named_hash}_dataloader.pt")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"loading dataset from {dataset}")
    if cache_dir.exists():
        logger.info(f"found cached dataloader in {cache_dir}")
        dataloader = torch.load(cache_dir)
    else:
        dataloader, _ = utils.get_loaders(dataset, nsamples=nsamples, seed=seed, model=tokenizer_path, seqlen=seqlen)
        torch.save(dataloader, str(cache_dir))
    assert len(dataloader) > 0, f"dataset {dataset} is empty"
    return dataloader
