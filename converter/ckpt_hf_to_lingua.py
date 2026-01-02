import argparse
import json
import os
import torch
import gc
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_ffn_dim_multiplier(intermediate_size, dim, multiple_of=256):
    """Compute ffn_dim_multiplier from intermediate_size and dim"""
    # intermediate_size = multiple_of * ceil((8 * dim / 3 * ffn_dim_multiplier) / multiple_of)
    # So we need to reverse this
    # First approximation: ffn_dim_multiplier ≈ intermediate_size * 3 / (8 * dim)
    return intermediate_size * 3.0 / (8.0 * dim)


def unpermute(w, n_heads, dim1, dim2):
    """Reverse the permute operation - same operation since it's self-inverse"""
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def main():
    parser = argparse.ArgumentParser(description="Convert HF Llama model to Lingua (Llama-like) format.")
    parser.add_argument("--input_dir", required=True, help="Directory containing HF model files")
    parser.add_argument("--output_dir", required=True, help="Directory to save Lingua model (params.json, consolidated.pth)")
    parser.add_argument("--model_name", default="pytorch_model.bin", help="Name of the HF model file")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"], help="Model dtype in final Lingua ckpt")
    parser.add_argument("--weight-tying", default=False, help="Set to True to use tied weights for output layer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load HF config
    config_path = os.path.join(args.input_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"{config_path} not found.")
    
    config = LlamaConfig.from_pretrained(args.input_dir)
    
    # Extract parameters
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    dim = config.hidden_size
    norm_eps = config.rms_norm_eps
    intermediate_size = config.intermediate_size
    rope_theta = config.rope_theta
    max_position_embeddings = config.max_position_embeddings
    n_kv_heads = config.num_key_value_heads
    vocab_size = config.vocab_size
    
    # Compute ffn_dim_multiplier and multiple_of
    ffn_dim_multiplier = compute_ffn_dim_multiplier(intermediate_size, dim)
    multiple_of = 256  # Standard value
    
    key_value_dim = dim // n_heads * n_kv_heads

    # 2) Load HF model weights
    model_path = os.path.join(args.input_dir, args.model_name)
    if not os.path.isfile(model_path):
        # Try alternative names
        alternative_names = ["model.safetensors", "pytorch_model.safetensors"]
        for alt_name in alternative_names:
            alt_path = os.path.join(args.input_dir, alt_name)
            if os.path.isfile(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Model file not found in {args.input_dir}")

    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        hf_state_dict = load_file(model_path)
    else:
        hf_state_dict = torch.load(model_path, map_location="cpu")

    # 3) Convert HF state dict to Lingua format
    lingua_state_dict = {}
    
    for layer_i in range(n_layers):
        # Get attention weights from HF format
        q_w = hf_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
        k_w = hf_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"]
        v_w = hf_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"]
        o_w = hf_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"]

        # Unpermute Q and K (reverse the permutation)
        q_w = unpermute(q_w, n_heads=n_heads, dim1=dim, dim2=dim)
        k_w = unpermute(k_w, n_heads=n_kv_heads, dim1=key_value_dim, dim2=dim)
        # v_w, o_w don't need unpermuting

        # Convert to Lingua format
        lingua_state_dict[f"layers.{layer_i}.attention.wq.weight"] = q_w
        lingua_state_dict[f"layers.{layer_i}.attention.wk.weight"] = k_w
        lingua_state_dict[f"layers.{layer_i}.attention.wv.weight"] = v_w
        lingua_state_dict[f"layers.{layer_i}.attention.wo.weight"] = o_w

        # MLP weights (gate=w1, up=w3, down=w2)
        lingua_state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = hf_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"]
        lingua_state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = hf_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"]
        lingua_state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = hf_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"]

        # Layer norms
        lingua_state_dict[f"layers.{layer_i}.attention_norm.weight"] = hf_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"]
        lingua_state_dict[f"layers.{layer_i}.ffn_norm.weight"] = hf_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"]

    # Embedding, final norm, output
    lingua_state_dict["tok_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"]
    lingua_state_dict["norm.weight"] = hf_state_dict["model.norm.weight"]

    try:
        lingua_state_dict["output.weight"] = hf_state_dict["lm_head.weight"]
    except KeyError:
        if args.weight_tying:
            print("lm_head.weight not found, using tied embedding weights for output")
            lingua_state_dict["output.tied_module.weight"] = hf_state_dict["model.embed_tokens.weight"] # weight tying
        else:
            lingua_state_dict["output.weight"] = hf_state_dict["model.embed_tokens.weight"]
    # ['model.embed_tokens.weight', 'model.norm.weight'] # layer 意外のhf_state_dictのキー
  
    # 4) Convert to specified dtype
    convert_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    for k, v in lingua_state_dict.items():
        lingua_state_dict[k] = v.to(dtype=convert_dtype)

    
    params = {
        "model": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "dim": dim,
            "norm_eps": norm_eps,
            "ffn_dim_multiplier": ffn_dim_multiplier,
            "multiple_of": multiple_of,
            "rope_theta": rope_theta,
            "max_seqlen": max_position_embeddings,
            "vocab_size": vocab_size,
        },
        "distributed": {
                "dp_shard": 8,
                "dp_replicate": 16,
                "tp_size": 1,
                "selective_activation_checkpointing": False,
                "compile": False,
                "fsdp_type": "full_shard",
                "model_dtype": "bf16",
                "matmul_allow_tf32": False,
                "allow_bf16_reduced_precision_reduction": False,
                "detect_anomaly": False,
                "compile_cache_size_limit": 8,
                "spawn_method": "forkserver"
        }
    }

    # 6) Save files
    # Save params.json
    params_path = os.path.join(args.output_dir, "params.json")
    write_json(params, params_path)

    # Save consolidated weights
    consolidated_state_dict = {"model": lingua_state_dict}
    consolidated_path = os.path.join(args.output_dir, "consolidated.pth")
    torch.save(consolidated_state_dict, consolidated_path)

    # Clean up
    del hf_state_dict
    del lingua_state_dict
    del consolidated_state_dict
    gc.collect()
    
    print(f"Conversion complete!")
    print(f"Lingua model saved to: {args.output_dir}")
    print(f"Files created:")
    print(f"  - params.json")
    print(f"  - consolidated.pth")


if __name__ == "__main__":
    main()