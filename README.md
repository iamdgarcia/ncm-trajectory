---
title: NCM Trajectory
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
short_description: Watch NCM text sharpen level by level
---

# NCM Trajectory

**▶ Try it live:** https://iamdgarcia-ncm-trajectory.static.hf.space

**Code on GitHub:** https://github.com/iamdgarcia/ncm-trajectory

Next-Concept Modeling ([arXiv 2610.14162](https://arxiv.org/abs/2610.14162), HazyResearch)
generates text coarse to fine: a multiscale residual VQ tokenizer represents a story as
12 nested sets of discrete codes (from 1 to 128), and a transformer predicts that hierarchy
with **one forward pass per level**.

This page replays real generations from the released TinyStories checkpoint. For each of the
12 sweeps you see the tokenizer's partial reconstruction of the codes generated so far: the
same story, sharpening. After half the sweeps it is word salad with the characters already in
place; it only becomes readable in the last three sweeps.

## Reproduce

Trajectories are captured with `tools/capture_trajectories.py` on CPU (no GPU needed):

```bash
git clone https://github.com/HazyResearch/ncm.git
git clone https://github.com/iamdgarcia/ncm-trajectory.git
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts
python ncm-trajectory/tools/capture_trajectories.py \
  --ncm-repo ncm --checkpoint-dir artifacts/tinystories \
  --n 8 --seed-start 42 --out ncm-trajectory/trajectories/tinystories.json
```

Point `--checkpoint-dir` at any released NCM checkpoint (xsum, wmt14, lm1b, owt, sudoku) to
replay its generation dynamics.

## Attribution

Model, checkpoints and generation procedure: [HazyResearch/ncm](https://github.com/HazyResearch/ncm)
(Hermann Kumbong et al., MIT license). This repo only captures and replays the coarse-to-fine
trajectory; the capture tool lives in `tools/`.
