# ContextualLens (NAACL 2025)

Training-free implementation of **ContextualLens** from _"Beyond Logit Lens: Contextual Embeddings for Robust Hallucination Detection & Grounding in VLMs"_ (NAACL 2025).

The method probes **intermediate contextual embeddings** of a vision-language model (VLM) to judge whether a generated answer is grounded in the input image and to return a grounded region.

## What the code does
- Generates an answer with a HuggingFace VLM.
- Re-runs a forward pass with the full prompt + generated tokens and collects hidden states.
- **Hallucination detection**: averages answer-token embeddings from a mid/late text layer and measures cosine similarity against image patch embeddings from a mid image layer. The max patch score is the confidence (higher -> more likely grounded).
- **Grounding (basic)**: builds a patch heatmap by taking the max cosine similarity over all layers (as in Section 4.2.2).
- **Grounding (bounding box)**: searches all patch-aligned boxes (per Section 4.2.2 bounding-box variant) and returns the best-scoring box in pixel coordinates.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python demo.py \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --image path/to/image.jpg \
  --question "What color is the jacket?" \
  --text-layer 27 --image-layer 13
```
`detection_confidence` is the max cosine similarity between the answer embedding and any image patch (range [-1, 1]); lower scores indicate a higher chance of hallucination.
Flags:
- `--text-layer`, `--image-layer`, `--bbox-layer`: 1-based indices into the model's hidden_states tuple (index 0 is embeddings). If omitted, defaults pick a mid image layer and late text layer, mirroring the paper.
- `--no-bbox`: skip bounding-box grounding.
- `--dtype float16|bfloat16|float32`: optional inference dtype.

Recommended starting layers (mirroring Appendix C of the paper):
- InternLM-VL: image layer 13, text layer 27.
- For other VLMs, pick mid image layers and later text layers; adjust via flags or `LayerConfig`.

## Library usage
```python
from contextual_lens import ContextualLens, LayerConfig
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(model_id)

lens = ContextualLens(
    model=model,
    processor=processor,
    layers=LayerConfig(text_layer=27, image_layer=13, bbox_layer=27),
)

image = Image.open("sample.jpg").convert("RGB")
result = lens.analyze(image, "What is the man holding?", max_new_tokens=64)

print(result.answer)
print("Confidence:", result.detection_confidence)
print("Patch heatmap shape:", result.patch_heatmap.shape)
if result.bounding_box:
    print("Pixel box:", result.bounding_box.pixel_box, "score:", result.bounding_box.score)
```

## Notes and assumptions
- Designed for HuggingFace causal VLMs that expose image patch tokens (e.g., Qwen2-VL, InternLM-VL, LLaVA HF). You may need to supply `image_token_id` on the processor/model if it is not present.
- Patch grids are inferred from `image_grid_thw` when available; otherwise we factor the number of image tokens into a near-square grid.
- Bounding boxes are aligned to the patch grid and projected back to the original image size using processed image dimensions from the processor.
- Bounding-box grounding enumerates all patch-aligned boxes (O(H^2 W^2)); on very large patch grids consider restricting image resolution or skipping the bounding-box step with `--no-bbox`.
- No training or fine-tuning required; the method uses only intermediate hidden states.
- Models loaded with `device_map` are supported; inputs are sent to the first device in the map. For single-device setups pass `--device` or rely on the default.
