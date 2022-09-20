import pandas as pd


def get_os_time(annotation_df: pd.DataFrame,
                portion_id: str) -> float:
    row = annotation_df[
        annotation_df['Tumor_Sample_Barcode'].str.contains(
            portion_id, na=False)
    ]
    os_time = row['OS.time']
    if len(os_time) > 1:
        os_time = os_time.iloc[0]
    else:
        os_time = os_time.item()
    return float(os_time)


def add_os(hla_df: pd.DataFrame,
           annotation_df: pd.DataFrame) -> pd.DataFrame:
    hla_df["OS.time"] = None
    for row_idx in range(len(hla_df)):
        row = hla_df.iloc[row_idx]
        pt_id = row['hla_samples']
        os_time = get_os_time(annotation_df, str(pt_id))
        hla_df.at[row_idx, "OS.time"] = os_time

    return hla_df


def main():
    hla_df = pd.read_pickle(
        "/home/asohn3/baraslab/hla/Data/splits/hla_genotype_os_intersect.pkl")
    annotation_df = pd.read_csv(
        "/home/asohn3/baraslab/hla/Data/tcga_wsi_annotations.tsv",
        sep="\t"
    )

    new_hla = add_os(hla_df, annotation_df)
    new_hla.to_pickle(
        "/home/asohn3/baraslab/hla/Data/splits/hla_genotype_os_intersect.pkl")


if __name__ == '__main__':
    main()
