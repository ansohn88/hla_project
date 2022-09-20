from pathlib import Path
from typing import Union

import pandas as pd

from utils import path2str

NCI_T_LABELS = [
    'Lung Adenocarcinoma',
    'Lung Squamous Cell Carcinoma',
    'Gastric Adenocarcinoma',
    'Muscle-Invasive Bladder Carcinoma',
    'Colorectal Adenocarcinoma',
    'Head and Neck Squamous Cell Carcinoma',
    'Clear Cell Renal Cell Carcinoma',
    'Endometrial Endometrioid Adenocarcinoma',
    'Uterine Carcinosarcoma',
    'Endometrial Serous Adenocarcinoma',
    'Endometrial Mixed Cell Adenocarcinoma'
    'Ovarian Serous Adenocarcinoma'
]

tcga_tsv = '/home/asohn3/baraslab/hla/Data/tcga_wsi_annotations.tsv'
annotation_df = pd.read_csv(tcga_tsv, sep='\t')
# subset_df = df[df['NCI.T.Label'].isin(NCI_T_LABELS)]

cancer_subset = '/home/asohn3/baraslab/hla/Data/best_os_hla_inter.pkl'
frameshift_path = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/frameshift'
no_frameshift_path = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/no_frameshift'

output_path = '/home/asohn3/baraslab/hla/Data/regression_subset'


def get_survival_info(annotation_df: pd.DataFrame,
                      tumor_sample_barcode: str) -> float:
    row_idx = annotation_df[annotation_df['Tumor_Sample_Barcode']
                            == tumor_sample_barcode].index[0]
    os_time = annotation_df.loc[row_idx, 'OS.time']
    vital_status = annotation_df.loc[row_idx, 'vital_status']
    return float(os_time), str(vital_status)


def merge_dfs_n_save(no_fs_path: Union[str, Path],
                     fs_dir: Union[str, Path],
                     output_path: str,
                     pep_len: int) -> None:
    if pep_len == 9:
        barcode = no_fs_path.stem[:-13]
    elif pep_len == 10:
        barcode = no_fs_path.stem[:-14]
    fs_path = Path(f'{fs_dir}/{barcode}/{no_fs_path.stem}.pkl')

    os_label, status = get_survival_info(annotation_df, barcode)

    no_fs_df = pd.read_pickle(path2str(no_fs_path))
    no_fs_df['Mutation_Type'] = 'Non_Frameshift'

    if not fs_path.exists():
        no_fs_df['OS.time'] = os_label
        no_fs_df['vital_status'] = status
        no_fs_df.to_pickle(
            f'{output_path}/{barcode}_final_nonsyn_p{pep_len}.pkl'
        )
    else:
        fs_df = pd.read_pickle(path2str(fs_path))
        fs_df['Mutation_Type'] = 'Frameshift'
        # concat vertically
        combined_df = pd.concat([fs_df, no_fs_df])
        combined_df['OS.time'] = os_label
        combined_df['vital_status'] = status
        combined_df.to_pickle(
            f'{output_path}/{barcode}_final_nonsyn_p{pep_len}.pkl'
        )


df = pd.read_pickle(cancer_subset)

tumor_sample_barcodes = df.loc[:, "Tumor_Sample_Barcode"].tolist()

PEP_LEN = 9

nofs_file_list = [
    Path(f'{no_frameshift_path}/{barcode}/{barcode}_pep_{PEP_LEN}_nonsyn.pkl')
    for barcode in set(tumor_sample_barcodes)
    if Path(f'{no_frameshift_path}/{barcode}/{barcode}_pep_{PEP_LEN}_nonsyn.pkl').exists()
]


def main():
    from joblib import Parallel, delayed

    Parallel(n_jobs=4)(delayed(merge_dfs_n_save)(nofs, frameshift_path, output_path, PEP_LEN)
                       for nofs in nofs_file_list)


if __name__ == '__main__':
    main()
