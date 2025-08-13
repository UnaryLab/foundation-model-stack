#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import ijson
import re
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from decimal import Decimal
from sys import stderr
from difflib import SequenceMatcher


##### WARN CHAT GPT ABOMINATION START ####

def extract_template_tokens(s):
    # Grab everything inside <...> and tokenize it
    matches = re.findall(r'<([^>]*)>', s)
    tokens = []
    for m in matches:
        tokens += re.split(r'[^\w\d]+', m)
    return tokens


def normalize_kernel_name(name):
    name = name.lower()
    name = re.sub(r'\bvoid\b', ' ', name)
    name = name.replace('::', ' ')
    name = re.sub(r'[^\w<>]+', ' ', name)  # keep angle brackets
    return name.strip()


def string_sim(s1, s2):
    # Normalize
    s1_n = normalize_kernel_name(s1)
    s2_n = normalize_kernel_name(s2)

    # Quick check
    if s1_n == s2_n:
        return 1.0

    # Extract meaningful tokens
    tokens1 = set(s1_n.split()) | set(extract_template_tokens(s1))
    tokens2 = set(s2_n.split()) | set(extract_template_tokens(s2))

    # Jaccard similarity over token sets
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    jaccard = len(intersection) / len(union) if union else 0.0

    # Fallback to SequenceMatcher for structural similarity
    seq_ratio = SequenceMatcher(None, s1_n, s2_n).ratio()

    # Combined score (you can tune this)
    return 1.0 * jaccard + 0.0 * seq_ratio

##### WARN CHAT GPT ABOMINATION END ####


def fancyprint(msg: str, dim: bool, bold: bool, type: str, color: str):
    styles = ""
    if bold:
        styles += "\033[1m"
    if dim:
        styles += "\033[2m"
    print(
        f"\033[1;{color}m{type} \033[0m{styles} {msg}\033[0m", file=stderr)


def err(msg: str, dim: bool = False, bold: bool = False):
    fancyprint(msg, dim, bold, "ERR: ", "31")


def warn(msg: str, dim: bool = False, bold: bool = False):
    fancyprint(msg, dim, bold, "WARN:", "33")


def info(msg: str, dim: bool = False, bold: bool = False):
    fancyprint(msg, dim, bold, "INFO:", "34")


def dtoi(ts):
    return np.int64(ts * Decimal('1000'))


def cpu_op_link(line, data):
    args = line['args']

    if 'Collective name' in args:
        name = f"{line['name']}:{args['Collective name']}"
    else:
        name = line['name']
    data.append({
        'ext_id': args['External id'],
        'seq_num': args['Sequence number'] if 'Sequence number' in args else None,
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'name': name,
    })


def kernel_link(line, data):
    data.append({
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
        'name': line['name'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'grid': line['args']['grid'],
        'block': line['args']['block'],
    })


def user_annotation_link(line, data):
    name = line['name']

    if re.fullmatch(r"Layer\d+", name):
        data.append({
            'type': 'layer',
            'ts': dtoi(line['ts']),
            'dur': dtoi(line['dur']),
            'value': int(name.split("Layer")[1]),
        })
    elif re.fullmatch(r"Token\d+", name):
        data.append({
            'type': 'token',
            'ts': dtoi(line['ts']),
            'dur': dtoi(line['dur']),
            'value': int(name.split("Token")[1]),
        })
    else:
        data.append({
            'type': 'operator-name',
            'ts': dtoi(line['ts']),
            'dur': dtoi(line['dur']),
            'value': name,
        })


def cuda_runtime_link(line, data):
    data.append({
        'name': line['name'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
    })


def gpu_memcpy_link(line, data):
    data.append({
        'name': line['name'],
        'device': line['args']['device'],
        # 'kind': line['args']['kind'],
        'bytes': line['args']['bytes'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
    })


def gpu_memset_link(line, data):
    data.append({
        'name': line['name'],
        'device': line['args']['device'],
        # 'kind': line['args']['kind'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'correlation': line['args']['correlation'],
        'ext_id': line['args']['External id'] if 'External id' in line['args'] else None,
    })


def json_to_pandas(rocprof_json_filename: str) -> pd.DataFrame:
    cat_func_map = {
        'cpu_op': cpu_op_link,
        'kernel': kernel_link,
        'user_annotation': user_annotation_link,
        'cuda_runtime': cuda_runtime_link,
        'gpu_memcpy': gpu_memcpy_link,
        'gpu_memset': gpu_memset_link,
    }
    cat_func_data = {
        cat: [] for cat in cat_func_map.keys()
    }

    with open(rocprof_json_filename, 'r') as fp:
        info("Reading timestamp data...")
        ignore_cat = Counter()
        for line in ijson.items(fp, 'traceEvents.item'):
            if 'cat' in line:
                cat = line['cat']
            else:
                continue

            if cat in cat_func_map:
                cat_func_map[cat](
                    line,
                    cat_func_data[cat],
                )
            else:
                ignore_cat.update((cat,))

    for cat in ignore_cat:
        warn(f"Ignoring cat: {cat} ({ignore_cat[cat]}x)")

    for cat in cat_func_data.keys():
        cat_func_data[cat] = pd.DataFrame(cat_func_data[cat])

    return cat_func_data


def merge_df(left, right, on, suffixes, combine):
    # WARN merge introduces NaN when no match
    # which converts values to floats
    if isinstance(on, tuple):
        merged = pd.merge(
            left,
            right,
            left_on=on[0],
            right_on=on[1],
            how="left",
            suffixes=suffixes,
        )
    else:
        merged = pd.merge(
            left,
            right,
            on=on,
            how="left",
            suffixes=suffixes,
        )
    for c in combine:
        col0 = merged.get(f'{c}{suffixes[0]}')
        col1 = merged.get(f'{c}{suffixes[1]}')
        mask = ((col0 == col1) | (col0.isna() & col1.isna()))
        assert mask.all(), f'combine column {c} does not match'

        merged.drop(
            columns=[f'{c}{suffixes[0]}'],
            inplace=True,
        )
        merged.rename(columns={f'{c}{suffixes[1]}': c}, inplace=True)
    return merged


def add_cuda_runtime(json_data):
    info("adding cuda runtime to gpu kernels...")
    runtime_merge = partial(
        merge_df,
        right=json_data['cuda_runtime'],
        on='correlation',
        suffixes=('', '_cuda_runtime'),
        # combine=('ext_id',)
        combine=()
    )
    for key in ('gpu_memcpy', 'gpu_memset', 'kernel'):
        json_data[key] = runtime_merge(json_data[key])


def assign_annotation(json_data):
    # TODO optimize this with bisection
    ann_df = json_data['user_annotation'].copy()
    cpu_df = json_data['cpu_op'].copy()

    ann_df['end_ts'] = ann_df['ts'] + ann_df['dur']

    for df_type in ('layer', 'token', 'operator-name'):
        info(f"assigning {df_type} to cpu ops...")
        to_merge = ann_df[ann_df['type'] == df_type][['ts', 'end_ts', 'value']].sort_values(
            ['ts', 'end_ts'],
            ascending=(True, False),
        )

        for _, row in to_merge.iterrows():
            mask = (cpu_df['ts'] >= row['ts']) & (
                cpu_df['ts'] <= row['end_ts'])
            cpu_df.loc[mask, df_type] = row['value']

    json_data['cpu_op'] = cpu_df


def add_cpu(json_data):
    cpu_merge = partial(
        merge_df,
        right=json_data['cpu_op'],
        on='ext_id',
        suffixes=('', '_cpu_op'),
        combine=()
    )

    memcpy_df = cpu_merge(json_data['gpu_memcpy'])
    memset_df = cpu_merge(json_data['gpu_memset'])
    kernel_df = cpu_merge(json_data['kernel'])

    return pd.concat((
        df for df in (memset_df, memcpy_df, kernel_df)
    )).sort_values('ts').reset_index(drop=True)


def parse_trace(trace_fn):
    json_data = json_to_pandas(trace_fn)

    add_cuda_runtime(json_data)
    assign_annotation(json_data)
    return add_cpu(json_data)


def kern_name_short(name, start=0, length=80):
    return f"{name[start:length]}{'...' if len(name) > length else ''}"


def get_pivoted(rocprof_csv_filename):
    rocprof_df = pd.read_csv(rocprof_csv_filename)

    unique_counters = list(
        c for c in rocprof_df.columns.to_list() if
        c != "Counter_Name" and c != "Counter_Value"
    )

    pivoted_df = rocprof_df.pivot_table(
        index=unique_counters,
        columns="Counter_Name",
        values="Counter_Value",
    ).sort_values("Start_Timestamp", ascending=True).reset_index()
    return pivoted_df


def get_combined_counters(
    counter_filenames,
    nvidia,
):
    kname = 'Kernel Name' if nvidia else 'Kernel_Name'
    df_combined = None
    info("Getting raw counter data...")
    for cur_csv in counter_filenames:
        if nvidia:
            df_cur = pd.read_csv(cur_csv, skiprows=[1])
        else:
            df_cur = get_pivoted(cur_csv)

        df_cur["_mi"] = df_cur.groupby(kname).cumcount()
        if df_combined is None:
            df_combined = df_cur
        else:
            df_combined = df_combined.merge(
                df_cur,
                on=[kname, "_mi"],
                how="left",
            )
            df_combined = df_combined.loc[
                :, ~df_combined.columns.str.endswith("_y")]
            df_combined.columns = df_combined.columns.str.replace("_x", "")

    assert df_combined is not None
    df_combined = df_combined.drop(columns=["_mi"])

    return df_combined


def get_merged(
    counter_filenames,
    df_ts,
    nvidia,
    token,
) -> pd.DataFrame:

    gpus = sorted(set(df_ts['gpu']))
    n_gpus = len(gpus)

    counter_filenames = tuple(
        sorted(fns)
        for fns in counter_filenames
    )
    # invert counters per GPU
    counter_filenames = tuple(list(fns) for fns in zip(*counter_filenames))
    n_counters = len(counter_filenames)

    if n_gpus != n_counters:
        warn(f"Number of counter files doesn't match {n_gpus} gpus:")
        warn(
            f"    Only the first {n_counters} counters will be used")

    df_cntr = pd.concat(
        [df.assign(gpu=i) for i, df in enumerate(
            map(partial(get_combined_counters, nvidia=nvidia), counter_filenames))],
        ignore_index=True
    )
    info("Loaded counter data")

    kname = 'Kernel Name' if nvidia else 'Kernel_Name'

    # TODO find a better way than this, it's so bad
    if nvidia:
        # preprocess Kernel names
        info("Attempting to match kernel names to timestamp names...")
        warn("Making best-effort to rename kernels, not guaranteed to be correct!")
        df_cntr[kname] = df_cntr.get(kname).str.replace(
            "<unnamed>", "(anonymous namespace)")
        df_cntr[kname] = df_cntr.get(kname).str.replace(
            r"\((int|bool|unsigned long)\)", "", regex=True)

    # sanity check GPU against agent id
    agent_id = 'Device' if nvidia else 'Agent_Id'
    aids = tuple(set(df_cntr[df_cntr['gpu'] == gpus[0]][agent_id]))
    assert len(aids) == 1, f'More than one Agent ID was present: {aids}'
    agent_id_inc = aids[0]
    for gpu in gpus[1:n_counters]:
        agent_id_inc += 1
        aids = tuple(set(df_cntr[df_cntr['gpu'] == gpu][agent_id]))
        assert len(aids) == 1, 'More than one Agent ID was present'
        assert aids[0] == agent_id_inc, 'Agent ID was unexpected'

    if nvidia:
        ts_mask = (
            (df_ts['gpu'] < n_counters) &
            (df_ts['token'] == token)
        )
    else:
        ts_mask = df_ts['gpu'] < n_counters

    ts_names = df_ts.loc[ts_mask, "name"]
    assert ts_names is not None

    cntr_names = df_cntr.get(kname)
    assert cntr_names is not None
    ts_name_set = Counter(ts_names)
    cntr_name_set = Counter(cntr_names)

    if nvidia:
        new_cntr_name = dict()
        # match cntr names to ts names by LCS count
        for cntr_name in cntr_name_set:
            best_match = None

            for ts_name in ts_name_set:
                cur_len = string_sim(cntr_name, ts_name)
                if best_match is None or best_match[1] <= cur_len:
                    best_match = (ts_name, cur_len)

            new_cntr_name[cntr_name] = best_match[0]
            # info(
            #     f"matched {kern_name_short(cntr_name, length=60)} -> {kern_name_short(best_match[0], length=60)}")
            ts_name_set.pop(best_match[0])

        df_cntr.replace({kname: new_cntr_name}, inplace=True)

        # update both name sets from renaming modifications
        ts_name_set = set(df_ts.loc[ts_mask, "name"])
        cntr_name_set = set(df_cntr.get(kname))

        cntr_diff = cntr_name_set - ts_name_set
        ts_diff = ts_name_set - cntr_name_set

        if len(cntr_diff):
            warn(
                "These counter kernels are not perent in timestamp data (and will be ignored):")
            for d in cntr_diff:
                warn(f"    {kern_name_short(d)}")

        if len(ts_diff):
            warn("These timestamp kernels are not present in counter data:")
            for d in ts_diff:
                warn(f"    {kern_name_short(d)}")

        same = ts_name_set & cntr_name_set
        ts_names = df_ts.loc[ts_mask, 'name']
        cntr_names = df_cntr.get(kname)

        # try to fix naming mismatches based on counts
        cntr_mismatch = dict()
        ts_mismatch = dict()
        for s in same:
            ts_mask = ts_names == s
            cntr_mask = cntr_names == s
            ts_count = ts_mask.sum()
            cntr_count = cntr_mask.sum()
            if ts_count != cntr_count:
                info(f"Count mismatch: timestamp({ts_count}), "
                     f"counter({cntr_count}) :"
                     f"{kern_name_short(s)}"
                     )
                cntr_entry = cntr_mismatch.setdefault(cntr_count, set())
                cntr_entry.add(s)

                ts_entry = ts_mismatch.setdefault(ts_count, set())
                ts_entry.add(s)

        new_cntr_name = dict()

        ignore_kernels = []

        for cntr_count, cntr_names_mismatched in cntr_mismatch.items():
            if cntr_count in ts_mismatch:
                ts_names_mismatched = ts_mismatch[cntr_count]

                for cntr_name in cntr_names_mismatched:
                    best_match = None
                    for ts_name in ts_names_mismatched:
                        cur_len = string_sim(ts_name, cntr_name)
                        if (best_match is None or best_match[1] <= cur_len):
                            best_match = (ts_name, cur_len)

                    if best_match is None:
                        ignore_kernels.append(cntr_name)
                    else:
                        assert cntr_name not in new_cntr_name
                        new_cntr_name[cntr_name] = best_match[0]
                        info(
                            f"new match {kern_name_short(cntr_name, start=20, length=80)} -> {kern_name_short(best_match[0], start=20, length=80)}")
                        ts_names_mismatched.remove(best_match[0])

                ignore_kernels.extend(ts_names_mismatched)

        df_cntr.replace({kname: new_cntr_name}, inplace=True)

        if len(ignore_kernels):
            warn("Failed to rename some kernels")
            for ignore_kernel in ignore_kernels:
                warn("  Ignoring kernel:")
                warn(f"    {kern_name_short(ignore_kernel)}")
        elif len(new_cntr_name):
            info("Succesfully renamed all counter kernels")
        ts_names = df_ts.get('name')

    else:
        cntr_diff = cntr_name_set - ts_name_set
        ts_diff = ts_name_set - cntr_name_set

        if len(cntr_diff):
            warn("These kernel names are missing from Pytorch Profiler data:")
            for d in cntr_diff:
                warn(f"    {kern_name_short(d)}: ({cntr_diff.get(d)})")

        if len(ts_diff):
            warn("These kernel names are missing from Counter data:")
            for d in ts_diff:
                warn(f"    {kern_name_short(d)}: ({ts_diff.get(d)})")

        ts_rename = {}
        if len(cntr_diff) and len(ts_diff):
            info("Trying to match missing names...")
            for c in cntr_diff:
                for t in ts_diff:
                    if cntr_diff.get(c) == ts_diff.get(t):
                        assert c not in ts_rename
                        ts_rename[t] = c
                        info(
                            f"renaming {kern_name_short(t)} to {kern_name_short(c)}")
                        break

        df_ts.rename(columns=ts_rename, inplace=True)

        ts_names = df_ts.loc[ts_mask, "name"]
        ts_name_set = Counter(ts_names)

        same = ts_name_set & cntr_name_set
        ignore_kernels = []
        ignore_kernels.extend(cntr_diff)
        for s in same:
            same_ts_mask = ts_names == s
            same_cntr_mask = cntr_names == s
            ts_count = same_ts_mask.sum()
            cntr_count = same_cntr_mask.sum()
            if ts_count != cntr_count:
                warn(f"Count mismatch: timestamp({ts_count}) and "
                     f"counter({cntr_count}), ignoring kernel:")
                warn(f"    {kern_name_short(s)}")
                ignore_kernels.append(s)
        ts_names = df_ts["name"]

    remove_ts_mask = ts_names.isin(ignore_kernels)
    remove_cntr_mask = cntr_names.isin(ignore_kernels)

    df_ts.drop(df_ts[remove_ts_mask].index, inplace=True)
    df_cntr.drop(df_cntr[remove_cntr_mask].index, inplace=True)

    if nvidia:
        df_cntr['token'] = token
    left_group_arrs = [
        'name', 'gpu', 'token'
    ] if nvidia else [
        'name', 'gpu',
    ]
    right_group_arrs = [
        kname, 'gpu', 'token'
    ] if nvidia else [
        kname, 'gpu',
    ]

    df_ts["_mi"] = df_ts.groupby(left_group_arrs).cumcount()
    df_cntr["_mi"] = df_cntr.groupby(right_group_arrs).cumcount()

    info("Merging timestamps and counters...")
    left_group_arrs += ["_mi"]
    right_group_arrs += ["_mi"]
    df_merged = df_ts.merge(
        df_cntr,
        left_on=left_group_arrs,
        right_on=right_group_arrs,
        how="left",
        suffixes=('', '_chopper_prev'),
    ).drop(columns=['_mi']).sort_values("ts")

    y_cols = [c for c in df_merged if c.endswith('_chopper_prev')]
    for y_col in y_cols:
        base_col = y_col[:-len('_chopper_prev')]
        df_merged[base_col] = df_merged[base_col].combine_first(
            df_merged[y_col])
    df_merged.drop(columns=y_cols, inplace=True)

    # nan_mask = ~df_merged["gpu_y"].isna()
    # test_mask = (df_merged.loc[nan_mask, "gpu"] ==
    #         df_merged.loc[nan_mask, "gpu_y"])
    # print(df_merged[nan_mask][~test_mask][['gpu', 'gpu_y']])
    # assert (df_merged.loc[nan_mask, "gpu"] ==
    #         df_merged.loc[nan_mask, "gpu_y"]).all()

    assert 'gpu_y' not in df_merged.columns
    df_merged.drop(columns=[kname], inplace=True)

    info(
        f"Got profiler token: {token}, now we have: {df_merged['token'].unique()}")

    return df_merged


def main(
        pytorch_trace,
        duration_pickle_fns,
        counters,
        nvidia,
        token,
        output_filename,
):
    # TODO double check what this does
    pd.set_option('future.no_silent_downcasting', True)

    assert not nvidia or token is not None or counters is None

    if pytorch_trace is None and duration_pickle_fns is None and counters is None:
        err("Nothing was passed to the script, what are you doing? :|")
        return -1
    if pytorch_trace is None and duration_pickle_fns is None:
        err("Please pass either raw pytorch traces or a duration pickle to merge with counters")
        return -1
    if pytorch_trace is not None and duration_pickle_fns is not None:
        err("Please only pass raw pytorch traces or a duration pickle not both")
        return -1

    if nvidia:
        info("Running for NVIDIA")
    else:
        info("Running for ROCm")
        assert token is None

    if pytorch_trace is not None:
        with ProcessPoolExecutor(max_workers=8) as ex:
            gpu_dfs = tuple(ex.map(parse_trace, pytorch_trace))
        for i, gpu_df in enumerate(gpu_dfs):
            gpu_df["gpu"] = i
        duration_pickle = pd.concat(gpu_dfs)
        if counters is None:
            duration_pickle.to_pickle(output_filename)
            return 0
    elif len(duration_pickle_fns) == 1:
        duration_pickle = pd.read_pickle(duration_pickle_fns[0])
    else:
        if not (pytorch_trace is None and counters is None):
            err("Trying to merge duration pickles, but was passed trace file or counters")
            return -1
        info(f"merging pickles: {duration_pickle_fns}")
        pickles = [pd.read_pickle(dpf) for dpf in duration_pickle_fns]
        merged = pd.concat(pickles)
        merged.to_pickle(output_filename)
        return 0

    merged = get_merged(counters, duration_pickle, nvidia, token)
    merged.to_pickle(output_filename)

    return 0


if __name__ == "__main__":
    desc = (
        "There are three ways to use this script:\n"
        "1) Pass just pytorch trace files to create a duration pickle\n"
        "2) Pass pytorch trace files with counters to create a counter pickle\n"
        "3) Pass a duration pickle with counters to create a counter pickle\n"
        "4) Pass multiple duration pickles to merge duration pickles\n"
    )
    parser = ArgumentParser(
        prog="chopper_trace",
        description=desc,
    )
    parser.add_argument(
        "--pytorch-trace",
        "-t",
        type=str,
        required=False,
        nargs="+",
        help="Filenames of pytorch trace json files"
    )
    parser.add_argument(
        "--duration_pickle",
        "-p",
        type=str,
        required=False,
        nargs='+',
        help="Pickle of pytorch traces without performance counters"
    )
    parser.add_argument(
        "--counters",
        "-c",
        action="append",
        required=False,
        nargs="+",
        type=str,
        help="List of counters csv files, pass additional lists to merge counter files together"
    )
    parser.add_argument(
        "--nvidia",
        "-nv",
        required=False,
        action="store_true",
        help="Pass for NVIDIA"
    )
    parser.add_argument(
        "--token",
        type=int,
        required=False,
        help="Pass for NVIDIA to select token to assign to timestamps (since nvidia only collects one token)"
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        type=str,
        required=True,
        help="Output pkl name",
    )
    args = parser.parse_args()
    exit(main(
        sorted(args.pytorch_trace) if args.pytorch_trace else None,
        args.duration_pickle,
        args.counters,
        args.nvidia,
        args.token,
        args.output_filename,
    ))
