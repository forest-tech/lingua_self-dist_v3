import torch
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
import os
from argparse import ArgumentParser

def list_pth_files(input_dir: str):
    """
    input_dir 以下に存在する .pth ファイルの絶対パスをすべて取得する
    """
    pth_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".pth"):
                pth_files.append(os.path.join(root, f))
    return pth_files

def print_state_dict_keys(state_dict):
    """
    state_dict のキーをすべて表示する
    """
    for key in state_dict.keys():
        print(key)

def print_state_dict_tree(state_dict):
    """
    state_dict のキーを tree 形式で表示する
    """
    tree = {}

    # キーを分解して木構造を作る
    for key in state_dict.keys():
        parts = key.split(".")
        current = tree
        for part in parts:
            current = current.setdefault(part, {})

    # 再帰的に表示
    def _print_tree(node, indent=""):
        for i, (key, child) in enumerate(node.items()):
            is_last = i == len(node) - 1
            branch = "└── " if is_last else "├── "
            print(indent + branch + key)
            _print_tree(child, indent + ("    " if is_last else "│   "))

    _print_tree(tree)

def convert_ckpt_to_distcp(ckpt_path: str, distcp_path: str) -> None:
    """
    Convert a PyTorch checkpoint file to the Distributed Checkpoint format.

    Args:
        ckpt_path (str): Path to the input PyTorch checkpoint file.
        distcp_path (str): Path where the output Distributed Checkpoint file will be saved.
    """
    torch_save_to_dcp(ckpt_path, distcp_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to Distributed Checkpoint format.")
    parser.add_argument("--input_dir", type=str, help="Path to the input PyTorch checkpoint file. (pth)")
    parser.add_argument("--output_dir", type=str, help="Path to save the output Distributed Checkpoint file.")

    args = parser.parse_args()

    # args.input_dir 内の pth ファイルパスを抽出
    pth_files = list_pth_files(args.input_dir)
    print('Found .pth files:', pth_files)
    pth_file = pth_files[0] if len(pth_files) > 0 else None
    print('convert file:', pth_file)


    state_dict = torch.load(pth_file)
    # weight_tyingの場合、tok_embeddingsをoutput.tied_moduleにもコピー
    state_dict["model"]["output.tied_module.weight"] = state_dict["model"]["tok_embeddings.weight"]

    # 保存
    fixed_pth_file = args.input_dir + "/consolidated_fixed.pth"
    torch.save(state_dict, fixed_pth_file)

    # print('Converting .pth to Distributed Checkpoint format...')
    # model_state_dict = torch.load(pth_file)
    # print_state_dict_keys(model_state_dict)
    # print_state_dict_tree(model_state_dict)

    # dcp に変換
    convert_ckpt_to_distcp(fixed_pth_file, args.output_dir)
