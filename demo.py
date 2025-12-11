"""
Minimal CLI to run the ContextualLens method on a single image/question pair.

Example:
    python demo.py \\
        --model Qwen/Qwen2-VL-2B-Instruct \\
        --image ./examples/dog.jpg \\
        --question "What color is the dog's collar?" \\
        --text-layer 27 --image-layer 13
"""

import argparse
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from contextual_lens import ContextualLens, LayerConfig


def parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    name = name.lower()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ContextualLens on an image-question pair.")
    parser.add_argument("--model", required=True, help="HuggingFace model id for a vision-language causal LM.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--question", required=True, help="Question to ask about the image.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate for the answer.")
    parser.add_argument("--text-layer", type=int, default=None, help="Layer index for answer embeddings.")
    parser.add_argument("--image-layer", type=int, default=None, help="Layer index for patch embeddings.")
    parser.add_argument("--bbox-layer", type=int, default=None, help="Layer index for bounding box grounding.")
    parser.add_argument("--device", default=None, help="Device for inference (e.g., cuda, cpu). Defaults to cuda if available.")
    parser.add_argument("--dtype", default=None, help="Optional dtype (float16, bfloat16, float32).")
    parser.add_argument("--no-bbox", action="store_true", help="Skip bounding box grounding.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)

    print(f"Loading model {args.model} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(args.model)

    lens = ContextualLens(
        model=model,
        processor=processor,
        layers=LayerConfig(text_layer=args.text_layer, image_layer=args.image_layer, bbox_layer=args.bbox_layer),
        device=device,
        dtype=dtype,
    )

    image = Image.open(args.image).convert("RGB")
    result = lens.analyze(
        image=image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        use_bounding_box=not args.no_bbox,
    )

    print(f"Answer: {result.answer}")
    print(f"Detection confidence (max cosine): {result.detection_confidence:.4f}")
    if result.bounding_box:
        x0, y0, x1, y1 = result.bounding_box.pixel_box
        print(f"Grounded box (pixels): [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}] score={result.bounding_box.score:.4f}")
    else:
        print("Grounded box: not computed")
    print(f"Patch heatmap shape: {tuple(result.patch_heatmap.shape)}")


if __name__ == "__main__":
    main()
