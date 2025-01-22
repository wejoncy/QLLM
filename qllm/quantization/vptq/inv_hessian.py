import torch
from argparse import ArgumentParser
import os
import tqdm
from ...utils.logger import get_logger

# load Hessian from files
def load_hessian(hessian_path, pbar=None, logger=None):
    if logger is None and pbar is None:
        print(f'load Hessian from {hessian_path}')
    elif pbar is not None:
        pbar.set_postfix_str(f'load Hessian from {hessian_path[-10:]}')
    else:
        logger.info(f'load Hessian from {hessian_path}')
    H_data = torch.load(f'{hessian_path}', weights_only=True, map_location='cpu')

    # convert H to sym matrix
    def flat_to_sym(V, N):
        A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
        idxs = torch.tril_indices(N, N, device=V.device)
        A[idxs.unbind()] = V
        A[idxs[1, :], idxs[0, :]] = V
        return A

    def regularize_H(H, n, sigma_reg):
        H.div_(torch.diag(H).mean())
        idx = torch.arange(n)
        H[idx, idx] += sigma_reg
        return H

    def basic_preprocess(H, mu, n):
        H.add_(mu[None, :] * mu[:, None])
        H = regularize_H(H, n, 1e-2)
        return H, mu

    H = flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    n = H_data['n']
    H, mu = basic_preprocess(H, mu, n)

    return H, mu

def main(args):
    logger = get_logger("qllm")
    # create folder
    os.makedirs(args.store_inv_hessian_dir, exist_ok=True)

    percdamp = 0.01
    hessian_files = [f for f in os.listdir(
        args.load_hessian_dir) if f.endswith('.pt')]

    for hessian_file in (pbar := tqdm.tqdm(hessian_files, desc="Inverting Hessian")):
        hessian_path = os.path.join(args.load_hessian_dir, hessian_file)
        hessian, mu = load_hessian(hessian_path, pbar=pbar, logger=logger)
        dev = 'cuda'
        hessian = hessian.to(dev)

        zero_idx = torch.diag(hessian) == 0
        hessian[zero_idx, zero_idx] = 1

        # get permutation
        perm = torch.argsort(torch.diag(hessian), descending=True).to(dev)
        if args.enable_perm:
            hessian = hessian[perm][:, perm]

        # add damping
        damp = percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(hessian.shape[0], device=dev)
        hessian[diag, diag] += damp

        # inverse Hessian
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        inv_hessian = hessian

        # Saving the inverted Hessian to the specified directory with the same file name
        save_path = os.path.join(args.store_inv_hessian_dir, hessian_file)
        if args.enable_perm is False:
            perm = torch.arange(inv_hessian.shape[0])
            
        torch.save({'invH': inv_hessian.to('cpu'),
                        'perm': perm.to('cpu'),
                        'zero_idx': zero_idx.to('cpu')}, save_path)
        
        pbar.set_postfix_str(f'Saved inverted Hessian to {save_path}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--load_hessian_dir', type=str, default=None,
                        help='Directory containing Hessian .pt files')
    parser.add_argument('--store_inv_hessian_dir', type=str, default=None,
                        help='Directory to save inverted Hessian .pt files')
    parser.add_argument('--enable_perm', action='store_true',
                        help='Enable permutation of Hessian')
    args = parser.parse_args()

    