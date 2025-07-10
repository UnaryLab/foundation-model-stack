import torch
from transformers import pipeline, AutoTokenizer
from fms.models import get_model
from fms.models.hf import to_hf_api
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import Dataset


architecture = "hf_pretrained"
model_path = "meta-llama/Llama-3.1-8B"
variant = model_path

device = "cuda"
compile = False
torch.set_default_device(device)
torch.set_default_dtype(torch.half)


model = get_model(
    architecture,
    variant,
    device_type=device,
)
model = to_hf_api(model)
tokenizer = AutoTokenizer.from_pretrained(model_path)

batch_size = 128

class MyDataset(Dataset):
    def __len__(self):
        return batch_size*4

    def __getitem__(self, i):
        prompt = """123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123 123"""
        return prompt

if compile:
    model.decoder = torch.compile(model.decoder)

pipe = pipeline(
    task="text-generation",
    model=model,
    max_new_tokens=25,
    tokenizer=tokenizer,
    device=device,
)
pipe.tokenizer.pad_token_id = model.config.eos_token_id

dataset = MyDataset()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # with_flops=True
) as prof:
    result = pipe(dataset, batch_size=batch_size)
    for _ in result:
        pass
prof.export_chrome_trace(f"hf_{device}_{'compiled_' if compile else ''}trace.json")
