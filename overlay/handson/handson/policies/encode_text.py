"""Prepare exact native FastWAM/Motus text caches from user-supplied UMT5 weights."""
import argparse
import hashlib
import json
from pathlib import Path

from .common import ROOT, read, require, sha256, vendor

TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fastwam", "motus"), required=True)
    parser.add_argument("--encoder", type=Path, required=True, help="models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local google/umt5-xxl directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", help="Repeat to restrict the cache to selected tasks")
    parser.add_argument("--device", default="cuda:0", help="Local extension: cuda:0, npu:8 or cpu")
    parser.add_argument("--allow-native-truncation", action="store_true", help="Explicit exploratory mode: retain native tokenizer truncation at its context limit and record affected prompts")
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to replace an existing text cache")
    vendor(args.model)
    import torch
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu
        torch.npu.set_device(device)
    require(args.model == "fastwam" or device.type == "cuda", "Motus requires CUDA")
    if args.model == "fastwam":
        from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
        limit, batch_size = 128, 16
        tokenizer = HuggingfaceTokenizer(str(args.tokenizer), seq_len=limit, clean="whitespace")
    else:
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.tokenizers import HuggingfaceTokenizer
        limit, batch_size = 512, 1
        tokenizer = HuggingfaceTokenizer(str(args.tokenizer), seq_len=limit, clean="whitespace")
    from ..tasks.language import bank as language_bank
    bank = language_bank()["tasks"]
    selected = args.task or list(bank)
    require(all(task in bank for task in selected), "Unknown task identifier")
    texts = sorted({text for task in selected for goal in bank[task]["goals"]
                    for text in [goal["canonical"], *goal["train"]]})
    wrapped = [TEMPLATE.format(p) if args.model == "fastwam" else p for p in texts]
    # Count cleaned tokens BEFORE native max-length padding/truncation is applied.
    lengths = [len(tokenizer.tokenizer(tokenizer._clean(t), add_special_tokens=True)["input_ids"])
               for t in wrapped]
    blocked = [{"prompt": p, "tokens": n} for p, n in zip(texts, lengths) if n > limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if blocked:
        report = args.output.with_suffix(args.output.suffix + ".blocked.json")
        report.write_text(json.dumps(dict(model=args.model, token_limit=limit, blocked=blocked), indent=2))
        if not args.allow_native_truncation:
            raise ValueError(f"{len(blocked)} instructions exceed the native token limit; see {report}. No cache written.")
        print(f"EXPLORATORY: native tokenizer truncation enabled for {len(blocked)} prompts; report: {report}", flush=True)
    if args.model == "fastwam":
        encoder = WanTextEncoder().to(dtype=torch.bfloat16)
        encoder.load_state_dict(torch.load(str(args.encoder), map_location="cpu", weights_only=True), strict=True)
        encoder = encoder.to(device).eval().requires_grad_(False)
    else:
        encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device("cuda"),
                                 checkpoint_path=str(args.encoder), tokenizer_path=str(args.tokenizer))
        encoder.model.eval().requires_grad_(False)
    result = {}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            prompts = texts[start:start + batch_size]
            if args.model == "fastwam":
                batch = [TEMPLATE.format(p) for p in prompts]
                batch += [batch[-1]] * (batch_size - len(batch))
                ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
                mask = mask.to(device).bool()
                embeddings = encoder(ids.to(device), mask)
                values = [embeddings[i, :int(mask[i].sum())].contiguous().cpu() for i in range(len(prompts))]
            else:
                values = [value.contiguous().cpu() for value in encoder(prompts, torch.device("cuda"))]
            for prompt, value in zip(prompts, values):
                require(value.ndim == 2 and value.shape[1] == 4096 and torch.isfinite(value).all(), "Invalid text features")
                key = hashlib.sha256(prompt.encode()).hexdigest() if args.model == "fastwam" else prompt
                result[key] = value
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.model == "fastwam":
        from safetensors.torch import save_file
        save_file(result, str(temporary))
    else:
        torch.save(result, temporary)
    temporary.replace(args.output)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(dict(
        model=args.model, device=str(device), prompts=len(result), tasks=selected, token_limit=limit,
        native_truncation_allowed=args.allow_native_truncation, truncated_prompts=blocked,
        encoder_sha256=sha256(args.encoder), cache_sha256=sha256(args.output)), indent=2) + "\n")


if __name__ == "__main__":
    main()
