#!/usr/bin/env python3
"""
analyze_adjacency_heatmap.py

给定一个 checkpoint（通常是最后一次迭代），提取 ALPINE 论文定义的 W_M 矩阵，
并将其可视化为热力图。

此版本仅绘制 W_M，不再加载或生成 A_true、差分矩阵及相关指标。
"""

import argparse
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from model import GPTConfig, GPT
except ImportError as exc:
    raise ImportError("❌ Error: Cannot import 'model.py'. Please ensure it is available.") from exc


# ==================== 配置类（保留阶段信息，去掉图结构） ====================


class ModelConfig:
    """模型配置类"""

    def __init__(self, checkpoint_dir, data_dir, model_name="Model"):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = os.path.abspath(data_dir)
        self.model_name = model_name
        self.device = torch.device("cpu")

        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 120

        self.vocab_size = None
        self.meta = {}
        self.S1 = self.S2 = self.S3 = []
        self.S1_tokens = self.S2_tokens = self.S3_tokens = []
        self.node_to_token = {}
        self.token_to_node = {}
        self.node_tokens = []
        self.num_nodes = None
        self.stoi = {}
        self.itos = {}

        self.load_meta_info()
        self.load_stage_info()

    def _preferred_match(self, matches):
        if not matches:
            return None
        matches = sorted(matches, key=lambda p: (len(p.split(os.sep)), len(p)))
        return matches[0]

    def _find_file(self, filename, friendly_name):
        search_dirs = [
            self.data_dir,
            os.path.dirname(self.data_dir),
            os.path.join(self.data_dir, "meta"),
            os.path.join(self.data_dir, "data"),
            os.path.join(self.data_dir, "graph"),
            os.path.join(self.data_dir, "graphs"),
            os.path.join(os.path.dirname(self.data_dir), "data"),
            os.path.join(os.path.dirname(self.data_dir), "graphs"),
        ]

        checked = set()
        for directory in search_dirs:
            directory = os.path.abspath(directory)
            if directory in checked or not os.path.isdir(directory):
                continue
            checked.add(directory)
            candidate = os.path.join(directory, filename)
            if os.path.isfile(candidate):
                print(f"  ✓ Located {friendly_name} at: {candidate}")
                return candidate

        recursive_matches = glob.glob(
            os.path.join(self.data_dir, "**", filename), recursive=True
        )
        match = self._preferred_match(recursive_matches)
        if match and os.path.isfile(match):
            print(f"  ✓ Located {friendly_name} via recursive search at: {match}")
            return match

        print(f"❌ ERROR: Unable to locate {friendly_name} ({filename})")
        return None

    def load_meta_info(self):
        meta_path = self._find_file("meta.pkl", "meta info")
        if meta_path is None:
            raise FileNotFoundError(
                f"Required file meta.pkl not found (search base: {self.data_dir})"
            )

        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        self.vocab_size = int(self.meta.get("vocab_size"))
        self.stoi = self.meta.get("stoi", {})
        self.itos = self.meta.get("itos", {})

        if "num_nodes" in self.meta:
            self.num_nodes = int(self.meta["num_nodes"])
        else:
            numeric_tokens = [tok for tok in self.stoi.values() if tok >= 2]
            self.num_nodes = len(numeric_tokens)

        print(
            f"  ✓ Loaded meta info: vocab_size={self.vocab_size}, "
            f"num_nodes={self.num_nodes}"
        )

    def load_stage_info(self):
        stage_info_path = self._find_file("stage_info.pkl", "stage info")
        if stage_info_path is None:
            print("⚠️ stage_info.pkl 未找到，使用 meta.pkl 的顺序节点作为 fallback。")
            if not self.stoi:
                raise RuntimeError(
                    "meta.pkl 中缺少 stoi 映射，无法推断节点 token。"
                )

            self.S1 = list(range(self.num_nodes))
            self.S2 = []
            self.S3 = []

            self.S1_tokens = [self.stoi[str(node)] for node in self.S1]
            self.S2_tokens = []
            self.S3_tokens = []

            self.node_tokens = sorted(self.S1_tokens)
            self.node_to_token = {node: token for node, token in zip(self.S1, self.node_tokens)}
            self.token_to_node = {token: node for node, token in self.node_to_token.items()}

            print(
                f"  ✓ Derived node tokens from meta: {len(self.node_tokens)} nodes "
                f"(tokens range {self.node_tokens[0]} - {self.node_tokens[-1]})"
            )
            return

        with open(stage_info_path, "rb") as f:
            stage_info = pickle.load(f)

        self.S1, self.S2, self.S3 = stage_info["stages"]

        total_nodes = len(self.S1) + len(self.S2) + len(self.S3)
        if self.num_nodes is not None and total_nodes != self.num_nodes:
            print(
                f"⚠️ Warning: stage_info nodes ({total_nodes}) != num_nodes ({self.num_nodes}). "
                "Using stage_info count."
            )
            self.num_nodes = total_nodes

        def node_list_to_tokens(node_list):
            tokens = []
            for node in node_list:
                if str(node) not in self.stoi:
                    raise KeyError(
                        f"Node {node} not present in stoi (check meta/stage info)."
                    )
                tokens.append(self.stoi[str(node)])
            return tokens

        self.S1_tokens = node_list_to_tokens(self.S1)
        self.S2_tokens = node_list_to_tokens(self.S2)
        self.S3_tokens = node_list_to_tokens(self.S3)

        self.node_tokens = sorted(
            set(self.S1_tokens + self.S2_tokens + self.S3_tokens)
        )

        self.node_to_token = {node: token for node, token in zip(self.S1, self.S1_tokens)}
        self.node_to_token.update({node: token for node, token in zip(self.S2, self.S2_tokens)})
        self.node_to_token.update({node: token for node, token in zip(self.S3, self.S3_tokens)})
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}

        print(
            f"  ✓ Loaded stage info: S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)} nodes"
        )


# ==================== 核心函数 ====================


def extract_W_M_prime(checkpoint_path, config):
    """提取 W_M 矩阵"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    model_args = checkpoint.get("model_args", {})
    if not model_args:
        model_args = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "vocab_size": config.vocab_size,
            "block_size": 512,
            "dropout": 0.0,
            "bias": False,
        }

    model_args["vocab_size"] = config.vocab_size

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    W_M_prime = []
    with torch.no_grad():
        for token_idx in range(config.vocab_size):
            token = torch.tensor([token_idx], device=config.device)
            token_emb = model.transformer.wte(token)
            ffn_out = model.transformer.h[0].mlp(token_emb)
            combined = token_emb + ffn_out
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy()[: config.vocab_size])

    return np.array(W_M_prime, dtype=np.float32)


# ==================== 辅助函数 ====================


def locate_checkpoint(checkpoint_dir, iteration=None, checkpoint_path=None):
    def parse_iteration_from_name(path):
        filename = os.path.basename(path)
        match = re.match(r"^(\d+)_ckpt", filename)
        if match:
            return int(match.group(1))
        match = re.search(r"ckpt_(\d+)", filename)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\d+", filename)
        return int(digits[0]) if digits else -1

    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path, parse_iteration_from_name(checkpoint_path)

    pattern = os.path.join(checkpoint_dir, "*ckpt*.pt")
    all_candidates = glob.glob(pattern)
    if not all_candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")

    if iteration is not None:
        matches = [p for p in all_candidates if parse_iteration_from_name(p) == iteration]
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint found whose parsed iteration equals {iteration} "
                f"(searched in {checkpoint_dir})."
            )
        matches.sort()
        chosen = matches[-1]
        return chosen, parse_iteration_from_name(chosen)

    all_candidates.sort(key=lambda p: (parse_iteration_from_name(p), p))
    chosen = all_candidates[-1]
    return chosen, parse_iteration_from_name(chosen)


def plot_W_heatmap(W_sub, node_tokens, output_path, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    vmin_w = np.percentile(W_sub, 1)
    vmax_w = np.percentile(W_sub, 99)
    if vmin_w == vmax_w:
        vmin_w -= 1
        vmax_w += 1

    im = ax.imshow(W_sub, cmap="viridis", vmin=vmin_w, vmax=vmax_w)
    ax.set_title("$W_M$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Target node token index")
    ax.set_ylabel("Source node token index")
    ax.set_xticks(range(len(node_tokens)))
    ax.set_yticks(range(len(node_tokens)))
    ax.set_xticklabels(node_tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(node_tokens, fontsize=8)
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Heatmap saved to: {output_path}")


# ==================== 主流程 ====================


def main():
    parser = argparse.ArgumentParser(description="Visualize W_M heatmap at a checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint files (e.g., 200_ckpt_20.pt).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with meta.pkl (and optionally stage_info.pkl).")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration number. If omitted, use the largest iteration parsed from filenames.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional explicit path to checkpoint file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store outputs. Default: heatmap_<name>/")
    parser.add_argument("--fig_title", type=str, default=None, help="Custom title for the heatmap figure.")
    parser.add_argument("--save_matrix", action="store_true", help="Save the cropped W_M submatrix as .npy.")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    checkpoint_path, iteration = locate_checkpoint(
        args.checkpoint_dir,
        iteration=args.iteration,
        checkpoint_path=args.checkpoint_path,
    )
    print(f"\n🎯 Using checkpoint: {checkpoint_path} (iteration={iteration})\n")

    print("=" * 60)
    print("Loading configuration / stage info...")
    print("=" * 60)
    config = ModelConfig(args.checkpoint_dir, args.data_dir, model_name=os.path.basename(args.checkpoint_dir))

    print("\n=" * 30)
    print("Extracting W_M matrix...")
    print("=" * 60)
    W_M = extract_W_M_prime(checkpoint_path, config)

    node_tokens = config.node_tokens
    if not node_tokens:
        raise ValueError("Node tokens list is empty. Check stage_info.pkl contents or meta fallback.")

    W_sub = W_M[np.ix_(node_tokens, node_tokens)]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"W_M_heatmap_{os.path.basename(os.path.abspath(args.checkpoint_dir))}"
    os.makedirs(output_dir, exist_ok=True)

    if args.fig_title:
        fig_title = args.fig_title
    else:
        fig_title = f"{config.model_name} — Iter {iteration}"

    plot_path = os.path.join(output_dir, f"W_M_heatmap_iter_{iteration}.png")
    plot_W_heatmap(
        W_sub,
        node_tokens,
        plot_path,
        title=fig_title,
    )

    if args.save_matrix:
        np.save(os.path.join(output_dir, f"W_M_iter_{iteration}.npy"), W_sub)
        print("✅ Saved W_M submatrix (.npy).")

    print("\nDone.")


if __name__ == "__main__":
    main()