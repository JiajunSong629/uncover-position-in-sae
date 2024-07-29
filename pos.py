from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import os
import numpy as np
import torch
from tqdm import tqdm
import plotly.express as px

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

def main():

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.6.hook_resid_pre",  # won't always be a hook point
        device=device,
    )    

    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    with torch.no_grad():
        random_integers = np.random.randint(0, 1001, size=100)
        batch_tokens = np.array([np.full(128, num) for num in random_integers])
        batch_tokens = torch.tensor(batch_tokens, dtype=torch.long)

        # activation store can give us tokens.
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

        # Use the SAE
        feature_acts = sae.encode(cache[sae.cfg.hook_name])
        sae_out = sae.decode(feature_acts)

    # find in a batch, for a position p, the index of features that are activated all the time
    for p in range(0, 128, 5):
        m = max((feature_acts[:, p] > 0).sum(0))
        print(
            p,
            torch.nonzero((feature_acts[:, p] > 0).sum(0) >= m - 5).tolist(),
            m.item(),
        )
    

if __name__ == "__main__":
    main()