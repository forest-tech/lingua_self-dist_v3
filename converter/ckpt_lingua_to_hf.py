import argparse
import json
import os
import torch
import gc
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_intermediate_size(n, ffn_dim_multiplier=1.0, multiple_of=256):
    
    import math
    raw_hidden = int(8 * n / 3 * ffn_dim_multiplier)
    return multiple_of * math.ceil(raw_hidden / multiple_of)


def permute(w, n_heads, dim1, dim2):

    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def main():
    parser = argparse.ArgumentParser(description="Convert Lingua (Llama-like) checkpoint to HF format, minimal version.")
    parser.add_argument("--input_dir", required=True, help="Directory containing params.json and consolidated.pth")
    parser.add_argument("--output_dir", required=True, help="Directory to save HF model (config.json, pytorch_model.bin)")
    parser.add_argument("--consolidated_name", default="consolidated.pth", help="Name of the single consolidated file")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"], help="Model dtype in final HF ckpt")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B", help="Tokenizer name of HF Hub")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load params.json
    params_path = os.path.join(args.input_dir, "params.json")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"{params_path} not found.")
    full_params = read_json(params_path)

    model_params = full_params["model"] if "model" in full_params else full_params

    n_layers = model_params["n_layers"]
    n_heads = model_params["n_heads"]
    dim = model_params["dim"]
    norm_eps = model_params.get("norm_eps", 1e-5)
    ffn_dim_multiplier = model_params.get("ffn_dim_multiplier", 1.0)
    multiple_of = model_params.get("multiple_of", 256)
    rope_theta = model_params.get("rope_theta", 10000.0)
    max_position_embeddings = model_params.get("max_seqlen", 2048)  
    n_kv_heads = model_params.get("n_kv_heads", n_heads)
    key_value_dim = dim // n_heads * n_kv_heads 

    # 2) load consolidated weights
    ckpt_path = os.path.join(args.input_dir, args.consolidated_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found.")

    loaded = torch.load(ckpt_path, map_location="cpu")  # 全部乗せ
    loaded = loaded["model"]

    # 3) リネーム + permute (Q/K)
    hf_state_dict = {}
    for layer_i in range(n_layers):
        # q,k,v の weight 取得
        q_w = loaded[f"layers.{layer_i}.attention.wq.weight"]
        k_w = loaded[f"layers.{layer_i}.attention.wk.weight"]
        v_w = loaded[f"layers.{layer_i}.attention.wv.weight"]
        o_w = loaded[f"layers.{layer_i}.attention.wo.weight"]

        # permute
        q_w = permute(q_w, n_heads=n_heads, dim1=dim, dim2=dim)
        k_w = permute(k_w, n_heads=n_kv_heads, dim1=key_value_dim, dim2=dim)
        # v_w, o_w は permute しない

        # 変換後の対応 (LlamaForCausalLM)
        hf_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = q_w
        hf_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = k_w
        hf_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = v_w
        hf_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = o_w

        # MLP (gate, up, down) + LayerNorm
        hf_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = loaded[f"layers.{layer_i}.feed_forward.w1.weight"]
        hf_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = loaded[f"layers.{layer_i}.feed_forward.w3.weight"]
        hf_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = loaded[f"layers.{layer_i}.feed_forward.w2.weight"]

        hf_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"] = loaded[f"layers.{layer_i}.attention_norm.weight"]
        hf_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = loaded[f"layers.{layer_i}.ffn_norm.weight"]

    # embedding, final norm, output
    hf_state_dict["model.embed_tokens.weight"] = loaded["tok_embeddings.weight"]
    hf_state_dict["model.norm.weight"] = loaded["norm.weight"]
    # hf_state_dict["lm_head.weight"] = loaded["output.weight"]
    try:
        hf_state_dict["lm_head.weight"] = loaded["output.weight"]
    except KeyError:
        hf_state_dict["model.embed_tokens.weight"] = loaded["output.tied_module.weight"]  # weight tying
        # hf_state_dict["lm_head.weight"] = loaded["output.tied_module.weight"]

    # 4) 変換後の dtype
    convert_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    for k, v in hf_state_dict.items():
        hf_state_dict[k] = v.to(dtype=convert_dtype)

    # 5) config 作成 (LlamaConfig)
    intermediate_size = compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of)
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=intermediate_size,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=norm_eps,
        num_key_value_heads=n_kv_heads,
        vocab_size=model_params.get("vocab_size", hf_state_dict["model.embed_tokens.weight"].shape[0]),
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        torch_dtype="bfloat16",
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=False,
    )
    # 保存
    config.save_pretrained(args.output_dir)
    model_path = args.output_dir + "/" +  "pytorch_model.bin"
    torch.save(hf_state_dict, model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tokenizer.save_pretrained(args.output_dir)

    # 後始末
    del hf_state_dict
    gc.collect()
    print("Conversion complete!")


if __name__ == "__main__":
    main()

