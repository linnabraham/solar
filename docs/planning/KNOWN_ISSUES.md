# Known Issues

## vit_pytorch version mismatch with trained checkpoint

**Status:** Workaround — downgrade vit_pytorch to 1.10

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict for ViT:
    size mismatch for cls_token: copying a param with shape torch.Size([1, 1, 1024])
        from checkpoint, the shape in current model is torch.Size([0, 1024]).
    size mismatch for pos_embedding: copying a param with shape torch.Size([1, 1025, 1024])
        from checkpoint, the shape in current model is torch.Size([1024, 1024]).
```

**Root cause:**  
The trained checkpoint `outputs/glad-shape-197/trained_model.pth` was produced on the original
server with an older version of `vit_pytorch` (≤1.x) that stored:
- `cls_token` as `nn.Parameter(torch.randn(1, 1, dim))` → shape `[1, 1, 1024]`
- `pos_embedding` as `nn.Parameter(torch.randn(1, num_patches + 1, dim))` → shape `[1, 1025, 1024]`

Current `vit_pytorch` (newer) stores them without the leading batch dimension:
- `cls_token` → `[num_cls_tokens, dim]`
- `pos_embedding` → `[num_patches + num_cls_tokens, dim]`

The model in `aarp_ml/torch/model.py` has always used `pool="mean"` in this repo, meaning the
checkpoint was also trained with `pool="mean"` — confirmed by the zero cls_token in the current
model (`[0, dim]` with `pool="mean"` in new vit_pytorch).

**Fix:**  
Pin `vit_pytorch==1.10` in the conda environment (the version used during original training):
```bash
pip install vit-pytorch==1.10
```
Do NOT change `pool="mean"` in `aarp_ml/torch/model.py` — that is correct.

**TODO:**  
- Pin `vit-pytorch==1.10` explicitly in `torch-tf-312.environment.yml` to prevent regression
- Verify the exact version by checking the environment on the original training server
