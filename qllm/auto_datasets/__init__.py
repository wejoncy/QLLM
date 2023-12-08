from pathlib import Path
import tempfile
import torch
from .. import utils


def get_sample_datas_for_quantization(args, seqlen=2048):
    logger = utils.logger.get_logger()
    named_hash = f"{args.model.replace(' ', '_')}_{args.dataset}"
    named_hash = f"{named_hash}_{args.nsamples}_{args.seed}_{seqlen}_{args.tokenizer}"
    cache_dir = Path(f"/{tempfile.gettempdir()}/qllm_v1/_{named_hash}_dataloader.pt")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"loading dataset from {args.dataset}")
    if cache_dir.exists():
        logger.info(f"found cached dataloader in {cache_dir}")
        dataloader = torch.load(cache_dir)
    else:
        dataloader, _ = utils.get_loaders(args.dataset, nsamples=args.nsamples,
                                          seed=args.seed, model=args.tokenizer, seqlen=seqlen)
        torch.save(dataloader, str(cache_dir))
    assert len(dataloader) > 0, f"dataset {args.dataset} is empty"
    return dataloader