#!/usr/bin/env python3
from fire import Fire
import pandas as pd
from functools import wraps, reduce
from typing import Dict, Optional


def metric_wrapper(name: str, metric_map: Optional[Dict[str, str]] = None):
    def decorator(fun):
        @wraps(fun)
        def wrapper(df):
            df[name] = fun(df)

        wrapper.name = name
        wrapper.metric_map = metric_map
        return wrapper
    return decorator


@metric_wrapper('HBM Bytes', {
    'hbm_bytes_read': ('dram__bytes_read.sum', 'sum'),
    'hbm_bytes_write': ('dram__bytes_write.sum', 'sum'),
})
def hbm_bytes(df):
    bytes_from_device_memory = df['hbm_bytes_read']*1e9
    bytes_to_device_memory = df['hbm_bytes_write']*1e9
    return bytes_from_device_memory + bytes_to_device_memory


@metric_wrapper('Tensor FLOPs', {
    'tensor_bf16_f32_flops': ('sm__ops_path_tensor_src_bf16_dst_fp32.sum', 'sum'),
})
def tensor_flops(df):
    return df['tensor_bf16_f32_flops'].str.replace(',', '', regex=False).astype(float)


@metric_wrapper('Tensor Arithmetic Intensity')
def tensor_ai(df):
    return (
        df[tensor_flops.name] /
        df[hbm_bytes.name]
    )


def print_ai(df_filename: str, n: int):
    """
    Extract arithmetic intensity from a chopper trace taken on an NVIDIA machine
    """

    metrics = (
        hbm_bytes,
        tensor_flops,
        tensor_ai,
    )

    metric_agg = reduce(
        lambda a, b: a | b,
        (metric.metric_map for metric in metrics if metric.metric_map),
        {}
    )
    metric_agg |= {'count': ('operator-name', 'count')}

    df = pd.read_pickle(df_filename).groupby(
        ['token', 'layer', 'operator-name'], dropna=False
    ).agg(
        **metric_agg
    ).reset_index()

    for metric in metrics:
        metric(df)

    agg_cols = ['median', 'min', 'max']

    nonzero = (df[tensor_ai.name] != 0) & ~df[tensor_ai.name].isna()

    print('-'*50)
    print('Largest', tensor_ai.name)
    ai = df[nonzero].groupby('operator-name')[tensor_ai.name].agg(agg_cols)
    print(ai.nlargest(n, columns=agg_cols).reset_index())

    print('Largest', tensor_flops.name)
    ai = df[nonzero].groupby('operator-name')[tensor_flops.name].agg(agg_cols)
    print(ai.nlargest(n, columns=agg_cols).reset_index())

    print('Largest', hbm_bytes.name)
    ai = df[nonzero].groupby('operator-name')[hbm_bytes.name].agg(agg_cols)
    print(ai.nlargest(n, columns=agg_cols).reset_index())

    print('Kernels')
    ai = df[nonzero].groupby('operator-name')['count'].agg(agg_cols)
    print(ai.nlargest(n, columns=agg_cols).reset_index())

    print('-'*50)

    op = 'ff_dp'
    assert op in df['operator-name'].unique()

    mask = (df['operator-name'] == op) & (df[hbm_bytes.name] > 0)
    print(df.loc[mask, ['token', 'layer', hbm_bytes.name]])

    print('-'*50)


def main(df_filename: str, n: int = 10):
    print_ai(df_filename, n)


if __name__ == '__main__':
    Fire(main)
