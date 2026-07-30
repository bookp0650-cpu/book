# SAM3 model

See `../models/MODEL_INFO.md` and `../models/inference_best.pt.sha256`. Update a checkpoint by stopping the service, verifying its trusted SHA-256, inspecting it on CPU with `weights_only=True`, replacing the file atomically, then restarting and checking `/model-info`.

The shared source is installed from `sam3_source`; BPE is `sam3/assets/bpe_simple_vocab_16e6.txt.gz`. The owner's inference repository is pinned in `vendor/owner_repo` at commit `08553cddd6a7833fecf4e99f7f4418d34490a4da`. Runtime now uses its `core.sam3_adapter.Sam3Adapter`, which calls `core.checkpoint_export.load_inference_checkpoint`; that builds without Hugging Face weights and loads `checkpoint["model"]` using `strict=True`. The result has zero missing and zero unexpected keys.
