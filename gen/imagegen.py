#!/usr/bin/env python3
"""Free, self-hosted background images for Shadow Kaizen.

Runs on GitHub's CPU runners (this repo is public, so Actions minutes are free). No account, no API key,
no daily quota. Model: an open text-to-image model run locally with diffusers (default SDXL-Turbo,
Stability AI Community License: free for commercial use under USD 1M annual revenue).

Reads jobs/*.json. Each job file is a list of
  {"out": "bg/<post id>/01.jpg", "prompt": "...", "w": 1080, "h": 1350, "look": "night"}
and makes every image whose `out` file is missing. Images are graded to the house look (navy monochrome,
grain, vignette) exactly as the studio's Cloudflare path did, so both sources match.
"""
import io, json, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/sdxl-turbo")
STEPS = int(os.environ.get("IMAGE_STEPS", "4"))
ONLY = os.environ.get("ONLY_JOB", "").strip()
STYLE = ("dark painterly digital illustration, cinematic chiaroscuro, moody night, deep navy blue and black, soft mist, "
         "epic scale, rim light, the subject sits in the lower half of the frame, the upper third is empty sky or haze")
NEGATIVE = "text, letters, words, writing, watermark, signature, logo, caption, frame, border, deformed hands, extra fingers, blurry face"
INK = ("detailed black ink pen illustration with cross-hatching on aged cream parchment, hand-drawn, symbolic, "
       "plain parchment background, the subject in the lower two thirds")


def gen_size(w, h):
    """Generate near the model's native size (512 for the turbo models), keeping the slide's shape, multiples of 64."""
    base = 512 if "turbo" in MODEL else 768
    s = (base * base / (w * h)) ** 0.5 * 1.15
    return max(64, round(w * s / 64) * 64), max(64, round(h * s / 64) * 64)


def grade(im, w, h, look="night"):
    from PIL import Image, ImageOps
    import numpy as np
    im = im.convert("RGB")
    if look == "ink":
        m = int(min(im.size) * 0.06)
        im = im.crop((m, m, im.width - m, im.height - m))
    im = ImageOps.fit(im, (w, h), Image.LANCZOS)
    g = np.asarray(ImageOps.grayscale(im), dtype=np.float32) / 255.0
    g = np.clip((g - 0.04) / 0.92, 0, 1) ** 1.15
    shadow, light = (np.array([34, 26, 20]), np.array([232, 220, 196])) if look == "ink" else (np.array([6, 10, 24]), np.array([196, 214, 240]))
    rgb = shadow + (light - shadow) * g[..., None]
    rng = np.random.default_rng(7)
    rgb += rng.normal(0, 4.0, rgb.shape)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - 0.35 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * 0.55) / (h * 0.75)) ** 2)
    rgb *= np.clip(vig, 0.82 if look == "ink" else 0.55, 1)[..., None]
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype("uint8")).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    todo = []
    for job in sorted((ROOT / "jobs").glob("*.json")):
        if ONLY and job.stem != ONLY:
            continue
        for item in json.loads(job.read_text()):
            if not (ROOT / item["out"]).exists():
                todo.append(item)
    print(f"{len(todo)} image(s) to make with {MODEL}, {STEPS} steps")
    if not todo:
        return 0
    import torch
    from diffusers import AutoPipelineForText2Image
    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()
    kw = {"variant": "fp16"} if "turbo" in MODEL else {}
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=torch.float32, **kw)
    pipe.set_progress_bar_config(disable=True)
    print(f"model loaded in {time.time() - t0:.0f}s")
    made = 0
    for n, item in enumerate(todo):
        w, h = item.get("w", 1080), item.get("h", 1350)
        gw, gh = gen_size(w, h)
        look = item.get("look", "night")
        prompt = f"{item['prompt']}, {INK if look == 'ink' else STYLE}"
        t = time.time()
        gen = torch.Generator().manual_seed(int(item.get("seed", 1000 + n)))
        args = dict(prompt=prompt, num_inference_steps=STEPS, width=gw, height=gh, generator=gen)
        if "turbo" in MODEL:
            args["guidance_scale"] = 0.0          # turbo models are trained without guidance; negative prompts do nothing here
        else:
            args.update(guidance_scale=6.0, negative_prompt=NEGATIVE)
        img = pipe(**args).images[0]
        out = ROOT / item["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(grade(img, w, h, look))
        made += 1
        print(f"  [{n + 1}/{len(todo)}] {item['out']} {gw}x{gh} in {time.time() - t:.0f}s: {item['prompt'][:60]}")
    print(f"made {made} in {time.time() - t0:.0f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
