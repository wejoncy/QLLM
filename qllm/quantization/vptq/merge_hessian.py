import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Merge Hessian components across multiple groups')
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory path (e.g., "./Hessians-Qwen2-57B-A14B-Instruct-6144-8k-seed-")')
    parser.add_argument('--save-dir', type=str, required=True,
                       help='Directory to save merged results')
    parser.add_argument('--groups', type=int, nargs='+', required=True,
                       help='Group numbers to merge (e.g., 4 5)')
    return parser.parse_args()

def merge_and_save_hessian(base_dir, groups, save_dir, entry):
    """
    Merges Hessian components across multiple groups and saves the merged result.
    Args:
        base_dir: Base directory path
        groups: List of group numbers
        save_dir: Directory to save results
        entry: File name to process
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_flatH = None
    total_mu = None
    total_ct = 0

    for group in groups:
        full_path = os.path.join(f'{base_dir}{group}', entry)
        if full_path.endswith('.txt'):continue
        data = torch.load(full_path, weights_only=False)

        if total_flatH is None:
            total_flatH = torch.zeros_like(data['flatH'])
            total_mu = torch.zeros_like(data['mu'])

        total_flatH += data['flatH']
        total_mu += data['mu'] * data['ct']
        total_ct += data['ct']

    average_mu = total_mu / total_ct if total_ct > 0 else total_mu

    merged_data = {
        'flatH': total_flatH / len(groups),
        'mu': average_mu,
        'n': data['n'],
        'ct': total_ct
    }
    
    save_path = os.path.join(save_dir, entry)
    torch.save(merged_data, save_path)
    # print(f"Merged data saved to {save_path}")

def main(args):    
    # Use the first group to get the list of files to process
    first_group_dir = f'{args.base_dir}{args.groups[0]}'
    for entry in os.listdir(first_group_dir):
        if entry.endswith('.txt'):continue
        merge_and_save_hessian(
            args.base_dir,
            args.groups,
            args.save_dir,
            entry
        )
        # print('----')

if __name__ == "__main__":
    args = parse_args()
    main(args)