# Capture real NCM coarse-to-fine generation trajectories as JSON.
#
# Loads a released NCM checkpoint (tokenizer + generator) on CPU, samples
# unconditionally the same way scripts/eval_shard.py does for TinyStories
# (level-0 codes from the checkpoint's level0_unigram, prefix = EOS, one
# forward pass per level), and records the partial reconstruction of the
# text after EVERY level. The result is a JSON "trajectory" you can replay
# in a browser to watch a story sharpen level by level.
#
# Usage:
#   python tools/capture_trajectories.py --ncm-repo <path-to-HazyResearch/ncm-clone> \
#       --checkpoint-dir <dir-with-generator.pt+tokenizer.pt+config.yaml> \
#       --n 8 --seed-start 42 --out trajectories/tinystories.json
#
# The checkpoint dir needs: generator.pt, tokenizer.pt, config.yaml and the
# BPE vocab/merges files referenced by config.yaml (all shipped in
# hazyresearch/ncm-checkpoints on the Hugging Face Hub).
import argparse, json, os, sys

import torch
from omegaconf import OmegaConf


def load_tokenizer(ck):
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    cfg = OmegaConf.load(os.path.join(ck, "config.yaml"))
    tok_cfg = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer, resolve=True))
    # resolve checkpoint-relative paths
    tok_cfg.vqvae.checkpoint_path = os.path.join(ck, tok_cfg.vqvae.checkpoint_path)
    pt = tok_cfg.pre_tokenizer
    pt.vocab_file_path = os.path.join(ck, pt.vocab_file_path)
    pt.merges_file_path = os.path.join(ck, pt.merges_file_path)
    tk = VQVAETokenizer.from_config(tok_cfg)
    tk.vqvae.eval()
    return tk, cfg


def clean_ids(ids, eos, pad):
    ids = list(ids)
    if eos in ids:
        ids = ids[: ids.index(eos)]
    return [t for t in ids if t not in (eos, pad)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm-repo", default="../ncm",
                    help="path to a clone of HazyResearch/ncm (provides models/ and tokenization/)")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--out", default="trajectories/tinystories.json")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-chars", type=int, default=700, help="truncate each level text in the JSON")
    args = ap.parse_args()

    sys.path.insert(0, args.ncm_repo)
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    ck = args.checkpoint_dir
    torch.manual_seed(args.seed_start)

    tk, cfg = load_tokenizer(ck)
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model)
    sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    l0_prob = sd["inference"]["level0_unigram"].float()

    eos = tk.pre_tokenizer.eos_token_id
    pad = tk.pre_tokenizer.pad_token_id
    levels = list(tk.vqvae.vq.levels)
    n_levels = len(levels)

    def dec_level(z_level):
        ids = torch.argmax(tk.vqvae.decoder(z_level), dim=-1).squeeze(0).tolist()
        text = tk.pre_tokenizer.decode(clean_ids(ids, eos, pad), skip_special_tokens=True)
        return text[: args.max_chars]

    samples = []
    with torch.no_grad():
        for i in range(args.n):
            seed = args.seed_start + i
            torch.manual_seed(seed)
            l0 = torch.multinomial(l0_prob, 1 * levels[0], replacement=True).view(1, levels[0])
            prefix_tokens = torch.full((1, 1), eos, dtype=torch.long)
            idx = net.generate(l0, prefix_tokens, temperature=args.temperature,
                               top_k=args.top_k, top_p=args.top_p)
            # partial reconstruction after every level (per-level text)
            _, z_levels = tk.vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
            texts = [dec_level(z) for z in z_levels]
            samples.append({
                "seed": seed,
                "level0_code": int(l0.view(-1)[0]),
                "texts": texts,          # index k = text after decoding levels 0..k
            })
            print(f"sample {i+1}/{args.n} (seed {seed}): "
                  f"{len(texts[-1])} chars, l0 code {samples[-1]['level0_code']}", flush=True)

    out = {
        "meta": {
            "checkpoint": f"hazyresearch/ncm-checkpoints ({os.path.basename(os.path.normpath(ck))})",
            "paper": "https://arxiv.org/abs/2610.14162",
            "repo": "https://github.com/HazyResearch/ncm",
            "levels": levels,
            "n_levels": n_levels,
            "generation": {"temperature": args.temperature, "top_k": args.top_k,
                           "top_p": args.top_p, "unconditional": True},
            "note": "One forward pass per level; texts are the tokenizer's partial "
                    "reconstruction of the generated codes after each level.",
        },
        "samples": samples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
