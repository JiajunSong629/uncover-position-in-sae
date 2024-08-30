import os

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import GPT2Tokenizer


def download():
    if os.path.exists("openwebtext.bin"):
        return

    ds = load_dataset(
        "Skylion007/openwebtext",
        streaming=True,
        split="train",
    )
    samples = []
    for sample in ds.take(100):
        samples.append(sample["text"])
    with open("openwebtext.txt", "w") as f:
        f.write("\n".join(samples))

    data = open("openwebtext.txt", "r").read()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tokenizer(data, return_tensors="pt")["input_ids"][0]
    ids = np.array(ids, dtype=np.int16)
    ids.tofile("openwebtext.bin")


def get_tokens(seq_len, n_samples):
    ids = np.fromfile("openwebtext.bin", dtype=np.int16)
    batch_tokens = [
        ids[i : i + seq_len]
        for i in np.random.randint(0, len(ids) - seq_len, n_samples)
    ]
    batch_tokens = torch.Tensor(np.array(batch_tokens)).long()

    return batch_tokens


@torch.no_grad()
def get_hook_activations(model, tokens, hook_name):
    hook_activations_store = torch.zeros(tokens.shape[0], tokens.shape[1], 768)

    def store_hook_activations(value, hook: HookPoint):
        hook_activations_store[:] = value

    model.run_with_hooks(
        tokens,
        return_type=None,
        fwd_hooks=[(hook_name, store_hook_activations)],
    )

    return hook_activations_store


def generate(layer, seed=42, seq_len=128, n_samples=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    tokens = get_tokens(seq_len, n_samples)

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device=device,
    )
    sae.eval()

    resids = torch.zeros(tokens.shape[0], tokens.shape[1], 768).to(device)
    for i in range(tokens.shape[0]):
        resids[i] = get_hook_activations(
            model, tokens[i : i + 1], f"blocks.{layer}.hook_resid_pre"
        )
    feature_acts = sae.encode(resids)

    os.makedirs(f"out/seed_{seed}", exist_ok=True)
    torch.save(resids, f"out/seed_{seed}/resids_l{layer}.pt")
    torch.save(feature_acts, f"out/seed_{seed}/feature_acts_l{layer}.pt")


def main():
    download()
    generate(layer=1)
    generate(layer=6)


if __name__ == "__main__":
    main()
