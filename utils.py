import itertools
import re
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from col_names import MAF_COLUMN_NAMES


def load_maf_w_cols(maf_file: str) -> pd.DataFrame:
    return pd.read_csv(maf_file,
                       sep='\t',
                       names=MAF_COLUMN_NAMES,
                       index_col=False,
                       low_memory=False)


def df_to_pkl(df: pd.DataFrame, out_dir: str) -> None:
    df.to_pickle(out_dir)


def pkl_to_df(pkl_fp: str) -> pd.DataFrame:
    return pd.read_pickle(pkl_fp)


def mkdir_for_sample(parent_dir: str,
                     tumor_sample_barcode: str
                     ) -> None:
    sample_dir = Path(f'{parent_dir}/{tumor_sample_barcode}')
    sample_dir.mkdir(parents=True, exist_ok=False)


def mv_to_sampledir(filepath: str) -> None:
    og_fp = Path(filepath)
    root = og_fp.parents[0]

    if filepath.endswith(".html"):
        sub_folder = og_fp.stem.split(".")[0][:-3]
    elif filepath.endswith(".txt"):
        sub_folder = (og_fp.stem)[:-3]
    elif filepath.endswith(".fa"):
        sub_folder = (og_fp.stem)[:-4]
    else:
        sub_folder = og_fp.stem
    og_fp.rename(root / sub_folder / og_fp.name)


def find_out_of_len(pkl_fp: Union[str, Path],
                    length: int) -> None:
    df = pd.read_pickle(pkl_fp)
    for idx in range(len(df)):
        hla_pep = df.iloc[idx]['Mut_Frags']
        id = df.iloc[idx]['HGVSp']
        if len(hla_pep) != length:
            print(f'{pkl_fp.stem} / {len(hla_pep)} / {id}')


def get_greatest_len(pkl_fp: Union[str, Path]) -> int:
    df = pd.read_pickle(pkl_fp)
    maxLen = 0
    for idx in range(len(df)):
        hla_pep = df.iloc[idx]['Mut_Frags']
        pep_len = len(hla_pep)
        if pep_len > maxLen:
            maxLen = pep_len
    return maxLen


def path2str(path: Union[str, Path]) -> Union[str, Path]:
    if isinstance(path, Path):
        return str(path)
    else:
        return path


def overlapping_n_grams(mutant_fragment: str,
                        n_gram_size: int) -> list:
    grams = [
        mutant_fragment[i:(i+n_gram_size)]
        for i in range(len(mutant_fragment) - n_gram_size + 1)
    ]
    return grams


def ngrams_iter(input_pep: str,
                ngram_size: int,
                token_regex=r"[^\s]+") -> Iterable[str]:
    input_iters = [
        map(lambda m: m.group(0), re.finditer(token_regex, input_pep))
        for n in range(ngram_size)
    ]
    for n in range(1, ngram_size):
        list(map(next, input_iters[n:]))

    output_iter = itertools.starmap(
        lambda *args: " ".join(args),
        zip(*input_iters)
    )
    return output_iter
