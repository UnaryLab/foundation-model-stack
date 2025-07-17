#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
from torch.profiler import profile, ProfilerActivity, ExecutionTraceObserver
from transformers import pipeline, AutoTokenizer
from fms.models import get_model
from fms.models.hf import to_hf_api
from fire import Fire


def main(
    variant: str = "meta-llama/Llama-3.1-8B",
    device: str = "cuda",
    compile: bool = False,
    batch_size: int = 32,
    n_batch: int = 1,
    token: str = '123',
    prompt_tokens: int = 32,
    max_new_tokens: int = 32,
):
    torch.set_default_device(device)
    torch.set_default_dtype(torch.half)

    architecture = "hf_pretrained"
    model = get_model(
        architecture,
        variant=variant,
        device_type=device,
    )
    model = to_hf_api(model)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    class MyDataset(Dataset):
        def __len__(self):
            return batch_size*n_batch

        def __getitem__(self, i):
            return prompt_tokens * token

    if compile:
        model.decoder = torch.compile(model.decoder)

    pipe = pipeline(
        task="text-generation",
        model=model,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
        device=device,
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id

    dataset = MyDataset()

    et = ExecutionTraceObserver()
    et.register_callback("pytorch_et.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule = tracing_schedule,
        on_trace_ready=lambda x: x.export_chrome_trace("kineto_trace.json"),
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        execution_trace_observer=et,
    ) as prof:
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # ) as prof:
        result = pipe(dataset, batch_size=batch_size)
        for r in result:
            print(r)


if __name__ == "__main__":
    Fire(main)
