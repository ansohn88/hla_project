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
