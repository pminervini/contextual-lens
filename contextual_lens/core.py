"""
Implementation of the ContextualLens approach from
"Beyond Logit Lens: Contextual Embeddings for Robust Hallucination
Detection & Grounding in VLMs" (NAACL 2025).

The algorithm is training-free and works by comparing contextual
embeddings of generated answer tokens with intermediate embeddings
of image patch tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

Tensor = torch.Tensor


@dataclass
class LayerConfig:
    """
    Layer indices (1-based w.r.t. hidden_states tuple where index 0 is embeddings).
    If any field is None we fall back to mid/late layers as suggested by the paper.
    """

    text_layer: Optional[int] = None
    image_layer: Optional[int] = None
    bbox_layer: Optional[int] = None


@dataclass
class BoundingBox:
    # Patch indices: (x0, y0, x1, y1) inclusive in patch grid coordinates.
    patch_box: Tuple[int, int, int, int]
    # Pixel coordinates relative to the original image size.
    pixel_box: Tuple[float, float, float, float]
    score: float


@dataclass
class LensResult:
    answer: str
    answer_tokens: List[str]
    detection_confidence: float
    patch_heatmap: Tensor  # shape (H, W)
    patch_grid: Tuple[int, int]
    bounding_box: Optional[BoundingBox]


class ContextualLens:
    """
    Lightweight wrapper that runs generation with a HuggingFace VLM and
    applies the ContextualLens scoring + grounding steps.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        layers: Optional[LayerConfig] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        system_prompt: str = "You are a helpful assistant that answers questions about images.",
    ) -> None:
        self.model = model
        self.processor = processor
        self.layers = layers or LayerConfig()
        self.system_prompt = system_prompt
        self.dtype = dtype

        self.using_device_map = getattr(self.model, "hf_device_map", None) is not None
        if self.using_device_map:
            self.device = device or self._infer_device_from_map(self.model.hf_device_map)
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            if self.dtype:
                self.model.to(dtype=self.dtype)
        self.model.eval()

        self.image_token_id = self._resolve_image_token_id()
        if self.image_token_id is None:
            raise ValueError(
                "Could not locate an image token id on the processor/model; "
                "pass it explicitly via processor.image_token_id or model.config.image_token_id."
            )

    @torch.no_grad()
    def analyze(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        use_bounding_box: bool = True,
    ) -> LensResult:
        """
        Runs generation, hallucination detection, and optional grounding.
        """

        prepared = self._prepare_inputs(image, question)
        gen_out = self._generate(prepared, max_new_tokens, temperature, do_sample, generation_kwargs)
        forward_out = self._forward_with_hidden(prepared, gen_out["sequences"])

        hidden_states: Tuple[Tensor, ...] = forward_out["hidden_states"]
        answer_tokens = gen_out["answer_tokens"]
        answer_text = gen_out["answer_text"]
        answer_token_count = gen_out["answer_token_count"]
        if answer_token_count == 0:
            raise RuntimeError("Model did not generate any answer tokens; try increasing max_new_tokens.")

        detection_confidence, patch_heatmap = self._hallucination_score(
            hidden_states, prepared, answer_token_count=answer_token_count
        )

        bbox: Optional[BoundingBox] = None
        if use_bounding_box:
            bbox = self._bounding_box_grounding(hidden_states, prepared, answer_token_count, image.size)

        return LensResult(
            answer=answer_text,
            answer_tokens=answer_tokens,
            detection_confidence=detection_confidence,
            patch_heatmap=patch_heatmap.cpu(),
            patch_grid=prepared["patch_grid"],
            bounding_box=bbox,
        )

    # -------------------------------------------------------------------------
    # Preparation and generation
    # -------------------------------------------------------------------------
    def _prepare_inputs(self, image: Image.Image, question: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_text], images=[image], return_tensors="pt")

        patch_grid = self._infer_patch_grid(inputs)

        tensor_inputs: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                target_dtype = self.dtype if (self.dtype is not None and v.is_floating_point()) else None
                tensor_inputs[k] = v.to(device=self.device, dtype=target_dtype)
            else:
                tensor_inputs[k] = v
        tensor_inputs["prompt_length"] = tensor_inputs["input_ids"].shape[1]
        tensor_inputs["image_token_mask"] = (tensor_inputs["input_ids"][0] == self.image_token_id)[: tensor_inputs["prompt_length"]]
        tensor_inputs["patch_grid"] = patch_grid
        tensor_inputs["processed_image_size"] = self._processed_image_size(inputs)
        return tensor_inputs

    def _generate(
        self,
        prepared_inputs: Dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        model_kwargs = {k: v for k, v in prepared_inputs.items() if isinstance(v, torch.Tensor) and k not in {"prompt_length", "image_token_mask"}}
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        outputs = self.model.generate(**model_kwargs, **gen_kwargs)
        sequences: Tensor = outputs.sequences  # (1, prompt_len + generated_len)
        prompt_len = prepared_inputs["prompt_length"]
        gen_token_ids = sequences[:, prompt_len:]
        answer_token_count = gen_token_ids.shape[1]
        answer_tokens = self.processor.tokenizer.convert_ids_to_tokens(gen_token_ids[0])
        answer_text = self.processor.tokenizer.decode(gen_token_ids[0], skip_special_tokens=True).strip()

        return {
            "sequences": sequences,
            "answer_tokens": answer_tokens,
            "answer_text": answer_text,
            "answer_token_count": answer_token_count,
        }

    def _forward_with_hidden(self, prepared_inputs: Dict[str, Any], full_sequences: Tensor) -> Dict[str, Any]:
        forward_inputs = {k: v for k, v in prepared_inputs.items() if isinstance(v, torch.Tensor) and k not in {"prompt_length", "image_token_mask"}}
        forward_inputs["input_ids"] = full_sequences
        forward_inputs["attention_mask"] = torch.ones_like(full_sequences)

        outputs = self.model(
            **forward_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return {"hidden_states": outputs.hidden_states}

    # -------------------------------------------------------------------------
    # Hallucination detection
    # -------------------------------------------------------------------------
    def _hallucination_score(
        self,
        hidden_states: Tuple[Tensor, ...],
        prepared_inputs: Dict[str, Any],
        answer_token_count: int,
    ) -> Tuple[float, Tensor]:
        layer_count = len(hidden_states) - 1  # exclude embedding layer
        text_layer = self._resolve_layer(self.layers.text_layer, layer_count, default=int(layer_count * 0.75))
        image_layer = self._resolve_layer(self.layers.image_layer, layer_count, default=layer_count // 2)

        prompt_len = prepared_inputs["prompt_length"]
        img_mask = prepared_inputs["image_token_mask"]
        if img_mask.sum() == 0:
            raise RuntimeError("No image patch tokens found in the prompt; cannot compute contextual similarity.")

        text_slice = slice(prompt_len, prompt_len + answer_token_count)
        answer_embeds = hidden_states[text_layer][0, text_slice, :]
        answer_vec = answer_embeds.mean(dim=0)

        patch_embeds = hidden_states[image_layer][0, :prompt_len, :][img_mask]
        patch_scores = self._cosine_batch(answer_vec, patch_embeds)

        # Grounding heatmap using max over layers (Section 4.2.2 basic technique).
        heatmaps = []
        H, W = prepared_inputs["patch_grid"]
        for layer_idx in range(1, layer_count + 1):
            ans_vec_layer = hidden_states[layer_idx][0, text_slice, :].mean(dim=0)
            patch_embeds_layer = hidden_states[layer_idx][0, :prompt_len, :][img_mask]
            if patch_embeds_layer.shape[0] != patch_embeds.shape[0]:
                continue
            heatmaps.append(self._cosine_batch(ans_vec_layer, patch_embeds_layer))

        if not heatmaps:
            heatmap = patch_scores
        else:
            stacked = torch.stack(heatmaps, dim=0)
            heatmap = stacked.max(dim=0).values

        return patch_scores.max().item(), heatmap.view(H, W)

    # -------------------------------------------------------------------------
    # Grounding via bounding boxes (Section 4.2.2)
    # -------------------------------------------------------------------------
    def _bounding_box_grounding(
        self,
        hidden_states: Tuple[Tensor, ...],
        prepared_inputs: Dict[str, Any],
        answer_token_count: int,
        original_image_size: Tuple[int, int],
    ) -> Optional[BoundingBox]:
        layer_count = len(hidden_states) - 1
        bbox_layer = self._resolve_layer(
            self.layers.bbox_layer,
            layer_count,
            default=self._resolve_layer(self.layers.text_layer, layer_count, default=int(layer_count * 0.75)),
        )

        prompt_len = prepared_inputs["prompt_length"]
        img_mask = prepared_inputs["image_token_mask"]
        if img_mask.sum() == 0:
            return None

        H, W = prepared_inputs["patch_grid"]
        text_slice = slice(prompt_len, prompt_len + answer_token_count)

        answer_vec = hidden_states[bbox_layer][0, text_slice, :].mean(dim=0)
        patch_embeds = hidden_states[bbox_layer][0, :prompt_len, :][img_mask]
        if patch_embeds.numel() == 0 or patch_embeds.shape[0] != H * W:
            return None

        patch_grid = patch_embeds.view(H, W, -1)
        # Integral image with zero padding to fetch box sums in O(1).
        integral = F.pad(patch_grid, (0, 0, 1, 0, 1, 0)).cumsum(dim=0).cumsum(dim=1)
        answer_vec = F.normalize(answer_vec, dim=-1)

        best_score = -1.0
        best_box = None

        for y0 in range(H):
            for y1 in range(y0, H):
                for x0 in range(W):
                    for x1 in range(x0, W):
                        total = (
                            integral[y1 + 1, x1 + 1]
                            - integral[y0, x1 + 1]
                            - integral[y1 + 1, x0]
                            + integral[y0, x0]
                        )
                        area = float((y1 - y0 + 1) * (x1 - x0 + 1))
                        mean_vec = total / area
                        score = torch.dot(answer_vec, F.normalize(mean_vec, dim=-1)).item()
                        if score > best_score:
                            best_score = score
                            best_box = (x0, y0, x1, y1)

        if best_box is None:
            return None

        pixel_box = self._patch_box_to_pixels(
            best_box,
            prepared_inputs["processed_image_size"],
            prepared_inputs["patch_grid"],
            original_image_size,
        )
        return BoundingBox(patch_box=best_box, pixel_box=pixel_box, score=best_score)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _resolve_image_token_id(self) -> Optional[int]:
        if hasattr(self.processor, "image_token_id"):
            return getattr(self.processor, "image_token_id")
        if hasattr(self.model.config, "image_token_id"):
            return getattr(self.model.config, "image_token_id")
        return None

    def _infer_patch_grid(self, inputs: Dict[str, Any]) -> Tuple[int, int]:
        if "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"][0, 0].tolist()  # (T, H, W)
            return int(grid[1]), int(grid[2])

        num_image_tokens = (inputs["input_ids"][0] == self.image_token_id).sum().item()
        root = int(math.isqrt(num_image_tokens))
        if root * root == num_image_tokens:
            return root, root
        # Fallback: choose a near-square factorization.
        best = (num_image_tokens, 1)
        for h in range(1, root + 2):
            if num_image_tokens % h == 0:
                w = num_image_tokens // h
                if abs(h - w) < abs(best[0] - best[1]):
                    best = (h, w)
        return best

    def _processed_image_size(self, inputs: Dict[str, Any]) -> Tuple[int, int]:
        if "pixel_values" not in inputs:
            return (1, 1)
        tensor = inputs["pixel_values"]
        if tensor.dim() == 5:
            _, _, _, h, w = tensor.shape
        elif tensor.dim() == 4:
            _, _, h, w = tensor.shape
        else:
            h = w = 1
        return (h, w)

    def _resolve_layer(self, requested: Optional[int], layer_count: int, default: int) -> int:
        layer = requested if requested is not None else default
        layer = max(1, min(layer, layer_count))
        return layer

    def _cosine_batch(self, reference: Tensor, candidates: Tensor) -> Tensor:
        ref = F.normalize(reference, dim=-1)
        cand = F.normalize(candidates, dim=-1)
        return torch.matmul(cand, ref)

    def _patch_box_to_pixels(
        self,
        patch_box: Tuple[int, int, int, int],
        processed_size: Tuple[int, int],
        patch_grid: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        proc_h, proc_w = processed_size
        grid_h, grid_w = patch_grid
        patch_h = proc_h / max(1, grid_h)
        patch_w = proc_w / max(1, grid_w)

        x0, y0, x1, y1 = patch_box
        x0_px = x0 * patch_w
        y0_px = y0 * patch_h
        x1_px = (x1 + 1) * patch_w
        y1_px = (y1 + 1) * patch_h

        orig_w, orig_h = original_size
        scale_x = orig_w / proc_w if proc_w else 1.0
        scale_y = orig_h / proc_h if proc_h else 1.0

        return (x0_px * scale_x, y0_px * scale_y, x1_px * scale_x, y1_px * scale_y)

    def _infer_device_from_map(self, device_map: Any) -> str:
        if isinstance(device_map, dict):
            first = next(iter(device_map.values()))
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return self._infer_device_from_map(first)
        return "cpu"
