from concurrent.futures import ProcessPoolExecutor

import pandas as pd

OUTPUT_DIR = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples'


def make_vep_df(vep_df: pd.DataFrame) -> pd.DataFrame:
    ref = vep_df['Reference_Allele'].astype(str)
    alt = vep_df['Tumor_Seq_Allele2'].astype(str)

    vep_df['vep_allele'] = ref + '/' + alt
    vep_df['Start_Position'].astype(int)
    vep_df['End_Position'].astype(int)

    for idx in range(len(vep_df)):
        # start = vep_df.iloc[idx, 1].copy()
        end = vep_df.iloc[idx, 2].copy()

        if (ref[idx] == '-') and (alt[idx] != '-'):
            vep_df.iloc[idx, 1] = end + 1
            vep_df.iloc[idx, 2] = end

    vep_df = vep_df.sort_values(
        ['Chromosome', 'Start_Position'], ascending=True)

    return vep_df


def make_vep_format(vep_df: pd.DataFrame,
                    outpath: str) -> None:
    vep_input_df = vep_df.loc[:, ['Chromosome',
                                  'Start_Position',
                                  'End_Position',
                                  'vep_allele',
                                  'Strand']
                              ]
    vep_input_df = vep_input_df.sort_values(
        ['Chromosome', 'Start_Position'],
        ascending=True)

    vep_input_df.columns = [
        'chromosome',
        'start',
        'end',
        'allele',
        'strand'
    ]
    vep_input_df.to_csv(outpath,
                        sep='\t',
                        header=None,
                        index=False)


fused_no_frameshift = pd.read_pickle(
    '/home/asohn3/baraslab/hla/Data/vep/VEP_fused_noFS_df.pkl')
fused_frameshift = pd.read_pickle(
    '/home/asohn3/baraslab/hla/Data/vep/VEP_fused_FSonly_df.pkl')


def df_to_vep_FS(sample_barcode: str) -> None:
    df = fused_frameshift.copy()
    sample_outdir = f'{OUTPUT_DIR}/frameshift'

    df_by_sample = df[df['Tumor_Sample_Barcode'] == sample_barcode]

    make_vep_format(
        vep_df=df_by_sample,
        outpath=f'{sample_outdir}/{sample_barcode}.vep'
    )


def df_to_vep_noFS(sample_barcode: str) -> None:
    df = fused_no_frameshift.copy()
    sample_outdir = f'{OUTPUT_DIR}/no_frameshift'

    df_by_sample = df[df['Tumor_Sample_Barcode'] == sample_barcode]

    make_vep_format(
        vep_df=df_by_sample,
        outpath=f'{sample_outdir}/{sample_barcode}.vep'
    )


# if __name__ == '__main__':
#     fused_vep = pd.read_pickle(
#         '/home/asohn3/baraslab/hla/Data/vep/VEP_fused_df.pkl')

#     fused_vep_noFS = fused_vep.loc[
#         ~fused_vep['Variant_Classification'].str.contains('Frame_Shift')
#     ].copy()
#     fused_vep_FSonly = fused_vep.loc[
#         fused_vep['Variant_Classification'].str.contains('Frame_Shift')
#     ].copy()

#     outdir = '/home/asohn3/baraslab/hla/Data/vep'

#     fused_vep_noFS.to_pickle(f'{outdir}/VEP_fused_noFS_df.pkl')
#     fused_vep_FSonly.to_pickle(f'{outdir}/VEP_fused_FSonly_df.pkl')


if __name__ == '__main__':
    # noFS_sample_ids = list(
    #     fused_no_frameshift['Tumor_Sample_Barcode'].unique()
    # )
    # with ProcessPoolExecutor(max_workers=12) as pool:
    #     pool.map(df_to_vep_noFS, noFS_sample_ids)

    FS_sample_ids = list(
        fused_frameshift['Tumor_Sample_Barcode'].unique()
    )
    with ProcessPoolExecutor(max_workers=12) as pool:
        pool.map(df_to_vep_FS, FS_sample_ids)
