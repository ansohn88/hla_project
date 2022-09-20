import copy
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_from_dir(input_dir: Union[str, Path],
                            output_path: str,
                            ) -> None:
    filelist = list(Path(input_dir).rglob("*.pkl"))

    fnames_w_lbls = [
        (
            file.stem[:-16],
            pd.read_pickle(file).iloc[0]['OS.time'],
            pd.read_pickle(file).iloc[0]['vital_status']
        )
        for file in filelist
    ]
    df = pd.DataFrame(
        fnames_w_lbls, columns=[
            'Tumor_Sample_Barcode', 'OS_time', 'vital_status']
    )

    train, test = train_test_split(df, test_size=0.2)

    train.to_csv(
        f'{output_path}/train_list.tsv', sep='\t', index=False
    )
    test.to_csv(
        f'{output_path}/test_list.tsv', sep='\t', index=False
    )


# def get_train_test_from_list(tcga_samples_list: str) -> None:
#     if tcga_samples_list.endswith(".pkl"):
#         df = pd.read_pickle(tcga_samples_list)
#     elif tcga_samples_list.endswith(".csv"):
#         df = pd.read_csv(tcga_samples_list)
#     elif tcga_samples_list.endswith(".tsv"):
#         df = pd.read_csv(tcga_samples_list, sep="\t")


if __name__ == '__main__':
    fd = '/home/asohn3/baraslab/hla/Data/regression_subset'
    op = '/home/asohn3/baraslab/hla/Data/splits'

    dset_gen = get_train_test_from_dir(fd, output_path=op)
