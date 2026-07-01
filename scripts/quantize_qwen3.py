# quantize_qwen3.py -- run with: source /data/hw3/comp_venv/bin/activate && python quantize_qwen3.py

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "/data/hw3/Qwen3-8B"               # reuse the local download -- do NOT re-pull from the Hub
OUTPUT_DIR = "/data/hw3/Qwen3-8B-FP8-Dynamic"  # absolute path -- this is what Chapters 5/6/9 expect

# Load the model in BF16 first
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# FP8 dynamic quantization recipe:
# - weights: FP8 (quantized statically, per-channel -- they don't change at inference)
# - activations: FP8 dynamic, per-token (computed fresh for each token at inference time)
# - lm_head: EXCLUDED (keeps BF16 precision for logit accuracy)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],   # <--- keep lm_head in BF16
)

# Apply quantization and save -- output_dir saves the compressed model, tokenizer,
# and the quantization recipe together; do NOT overwrite the original BF16 checkpoint
oneshot(model=model, tokenizer=tokenizer, recipe=recipe, output_dir=OUTPUT_DIR)

# Verify quantization was applied -- read it back from the saved config, not the
# in-memory model object (oneshot's in-memory model.config does not reflect it)
import json
saved_config = json.load(open(f"{OUTPUT_DIR}/config.json"))
quant_config = saved_config.get("quantization_config", {})
print("Quantization config:", json.dumps(quant_config, indent=2))
print(f"\nSaved quantized model to: {OUTPUT_DIR}")
