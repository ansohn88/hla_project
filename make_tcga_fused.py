# `fuse_adj_snvs` adaptived from:
# https://github.com/OmnesRes/DeepTMB/blob/master/files/public_maf.py
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

VEP_COLS = [
    'Chromosome',
    'Start_Position',
    'End_Position',
    'Reference_Allele',
    'Tumor_Seq_Allele2',
    'Strand',
    'Variant_Classification',
    'Variant_Type',
    'Tumor_Sample_Barcode',
    'FILTER'
]

CHROMOSOMES = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))

OUTPUT_DIR = '/home/asohn3/baraslab/hla/Data/vep'

# maf_file = '/home/asohn3/baraslab/hla/Data/mc3.v0.2.8.PUBLIC.maf'
maf_file = '/home/asohn3/baraslab/hla/Data/maf_for_vep.pkl'


def return_df(filepath: str) -> pd.DataFrame:
    if filepath.endswith('.maf'):
        maf_df: pd.DataFrame = pd.read_csv(filepath,
                                           sep='\t',
                                           index_col=False,
                                           low_memory=False)
        # maf_df = maf_df.loc[:, VEP_COLS]
    elif filepath.endswith('.pkl'):
        maf_df: pd.DataFrame = pd.read_pickle(filepath)
    else:
        raise ValueError('Use .maf or .pkl file')

    return maf_df


start = time.time()
maf_df = return_df(maf_file)
# # The MAF contains nonpreferred pairs which results in some samples having duplicated variants
# maf_df = maf_df.loc[(maf_df['FILTER'] == 'PASS') | (
#     maf_df['FILTER'] == 'wga') | (maf_df['FILTER'] == 'native_wga_mix')]
# maf_df = maf_df.loc[~pd.isna(maf_df['Tumor_Seq_Allele2'])]
# maf_df = maf_df.loc[~maf_df['Tumor_Seq_Allele2'].str.contains('N')]
# maf_df = maf_df.loc[:, VEP_COLS]
# maf_df.to_pickle('/home/asohn3/baraslab/hla/Data/maf_for_vep.pkl')
print(f'Time elapsed for maf --> dataframe: {time.time() - start}s')


def fuse_adj_snvs(sample_id: str) -> pd.DataFrame:
    if not isinstance(sample_id, str):
        sample_id = str(sample_id)
    # subset tcga by given sample
    tumor_sample = maf_df[maf_df['Tumor_Sample_Barcode'] == sample_id]
    # fuse adjacent SNVs per chromosome and SNP variant type bases
    tumor_df = []
    for chr in tumor_sample['Chromosome'].unique():
        sample_chr_snp = tumor_sample.loc[
            (tumor_sample['Chromosome'] == chr) & (
                tumor_sample['Variant_Type'] == 'SNP')
        ].copy()

        if len(sample_chr_snp) > 1:
            # find the SNVs (>=2) that are adjacent to one another
            to_merge = sum(
                sample_chr_snp['Start_Position'].values -
                sample_chr_snp['Start_Position'].values[...,
                                                        np.newaxis] == 1
            )
            merged = []
            position = 0
            indices_to_remove = []
            while sum(to_merge[position:]) > 0 and position < len(to_merge) - 1:
                for index, merge in enumerate(to_merge[position:]):
                    if merge:
                        # if there are adjacent SNVs,
                        # recapitulate `Start_Position` and `End_Position`
                        # for ex., if there is a dinucleotide with positions
                        # 101-101 and 102-102, the new start and end will be
                        # 101-102
                        first = position + index - 1
                        last = position + index
                        while to_merge[last] == 1:
                            last += 1
                            if last < len(to_merge):
                                pass
                            else:
                                break
                        position = last
                        last -= 1
                        snv = sample_chr_snp.iloc[[first]].copy()
                        snv['Tumor_Sample_Barcode'] = sample_id
                        snv['End_Position'] = sample_chr_snp.iloc[last]['Start_Position']
                        snv['Variant_Classification'] = 'Missense_Mutation'
                        if last - first == 1:
                            type = 'DNP'
                        elif last - first == 2:
                            type = 'TNP'
                        else:
                            type = 'ONP'
                        snv['Variant_Type'] = type
                        ref = ''
                        alt = ''
                        for row in sample_chr_snp.iloc[first:last+1, :].itertuples():
                            ref += row.Reference_Allele
                            alt += row.Tumor_Seq_Allele2
                        snv['Reference_Allele'] = ref
                        snv['Tumor_Seq_Allele2'] = alt
                        indices_to_remove += list(range(first, last + 1))
                        merged.append(snv)
                        break

            if len(merged) != 0:
                not_fused = sample_chr_snp[
                    ~np.array(
                        [i in indices_to_remove for i in range(len(sample_chr_snp))])
                ]
                merged = pd.concat(merged, ignore_index=True)
                sample_chr_snp = pd.concat(
                    [not_fused, merged],
                    ignore_index=True
                )
        tumor_df.append(sample_chr_snp)

    if len(tumor_df) > 0:
        return pd.concat(
            [
                pd.concat(tumor_df, ignore_index=True),
                tumor_sample[tumor_sample['Variant_Type']
                             != 'SNP'].copy()
            ],
            ignore_index=True)
    else:
        return tumor_sample


# if __name__ == '__main__':
#     tcga_sample_ids = list(set(maf_df['Tumor_Sample_Barcode']))

#     start = time.time()
#     # result = []
#     # for id in tqdm(tcga_sample_ids):
#     #     result.append(fuse_adj_snvs(maf_df, id))

#     result = []
#     with ProcessPoolExecutor(max_workers=20) as pool:
#         for r in pool.map(fuse_adj_snvs, tcga_sample_ids):
#             result.append(r)

#     tcga_fused = pd.concat(result, ignore_index=True)
#     print(f'Time elapsed to fuse the dinucleotides: {time.time() - start}s')
#     tcga_fused.to_pickle(f'{OUTPUT_DIR}/tcga_fused_df.pkl')
