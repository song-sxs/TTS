from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, AutoConfig

# 配置量化参数
quantize_config = BaseQuantizeConfig(bits=8, group_size=128, desc_act=False)

# 加载原始模型
model = AutoGPTQForCausalLM.from_pretrained(
    "pretrained_models/CosyVoice2-0.5B",
    quantize_config=quantize_config,
    use_triton=False,
    model_type="qwen2",
    use_safetensors=True,
    trust_remote_code=True
)

# 保存量化后的模型
model.save_quantized("pretrained_models/CosyVoice2-0.5B-AWQ", use_safetensors=True)