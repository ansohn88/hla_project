import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser


FRAMESHIFT_DIR = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/frameshift'
PROTEINSEQS_DIR = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/no_frameshift'


def txt_to_df(input_txtfile: str) -> pd.DataFrame:
    if '_ps.txt' in input_txtfile:
        rows_to_skip = 43
    elif '_ds.txt' in input_txtfile:
        rows_to_skip = 45

    df = pd.read_csv(
        input_txtfile,
        sep='\t',
        skiprows=rows_to_skip
    )
    df['HGVSp'] = df['HGVSp'].str.replace("%3D", "=")
    return df


class ProteinSequence:

    def __init__(self,
                 ref_fasta_file: str,
                 ps_mut_fasta: Union[str, None],
                 ds_output_file: Union[str, None]
                 ) -> None:
        self.reference = self.get_fa_dict(ref_fasta_file)

        if ps_mut_fasta is not None:
            self.proteinseqs_output = self.get_fa_dict(ps_mut_fasta)
        if ds_output_file is not None:
            self.downstream_output = self.get_downstream_pred(ds_output_file)

    def get_fa_dict(self,
                    fp: str) -> dict:
        hgvsp_seq = {
            parsed[0]: parsed[1] for parsed in list(
                SimpleFastaParser(open(fp))
            )
        }
        return hgvsp_seq

    def get_downstream_pred(self,
                            fp: str) -> dict:
        ds_df = txt_to_df(fp)
        ds_output = dict(zip(ds_df.ENSP, ds_df.DownstreamProtein))
        return ds_output

    def get_HGVSp(self) -> List:
        # identifiers = [
        #     seq_rec.id for seq_rec in SeqIO.parse(self.fp, "fasta")
        # ]
        identifiers = list(self.proteinseqs_output.keys())
        return identifiers

    def get_prot_seq(self) -> List:
        prot_seqs = list(self.proteinseqs_output.values())
        return prot_seqs

    def get_mut_info(self,
                     hgvsp_id: str) -> str:
        ensembl_id, _, aa_change = hgvsp_id.split(".")
        return ensembl_id, aa_change

    def get_mut_pos(self,
                    aa_change: str) -> int:
        pos = re.findall(r'\d+', aa_change)[0]
        return int(pos)
    
    def get_pep_nonfs(self,
                      mut: str,
                      hgvsp_id: str,
                      pep_len: int) -> str:
        _, aa_change = self.get_mut_info(hgvsp_id)
        pos = self.get_mut_pos(aa_change)
        # the HGVSp position starts w/ 1-index; make into 0-index for arr slicing
        idx = pos - 1

        is_nan = pd.isna(mut)
        ref_peptide = mut[(idx - (pep_len-1)):idx]
        prefix_len = len(ref_peptide)
        len_wo_mut = pep_len - 1
        total_frag_len = (2 * pep_len) - 1

        if len(mut) < total_frag_len:
            return mut

        if is_nan or (hgvsp_id[-3:] == "Ter"):
            if pos > total_frag_len:
                return mut[(idx - total_frag_len):idx]
            elif pos <= total_frag_len:
                return mut[:idx]
        else:
            suffix_len = len(mut[(idx + 1):(idx + pep_len)])

            if suffix_len < len_wo_mut and pos < pep_len:
                to_add = pep_len - suffix_len
                return mut[:(idx + to_add)]
            elif suffix_len < len_wo_mut and pos >= pep_len:
                to_add = len_wo_mut - suffix_len
                return mut[(idx - (len_wo_mut + to_add)):]
            elif pos < pep_len and (len(mut[idx+1:]) > len_wo_mut):
                to_add = len_wo_mut - idx
                return mut[:(idx + (len_wo_mut + to_add + 1))]
            elif (hgvsp_id[-1:] == '=') and suffix_len < len_wo_mut:
                to_add = len_wo_mut - suffix_len
                return mut[(idx - (len_wo_mut + to_add)):]
            else:
                return mut[(idx - len_wo_mut):(idx + pep_len)]

    def get_pep_fs(self,
                   hgvsp_id: str,
                   pep_len: int) -> str:
        ensembl_id, aa_change = self.get_mut_info(hgvsp_id)
        pos = self.get_mut_pos(aa_change)
        idx = pos - 1

        if ensembl_id not in self.reference:
            print(self.ds_inp_fp)
        ref = self.reference[ensembl_id]
        downstream_peptide = self.downstream_output[hgvsp_id]

        is_nan = pd.isna(downstream_peptide)
        ref_peptide = ref[(idx - (pep_len - 1)):idx]
        prefix_len = len(ref_peptide)
        total_frag_len = (2 * pep_len) - 1

        if is_nan or (hgvsp_id[-3:] == "Ter"):
            if pos >= total_frag_len:
                return ref[(idx - total_frag_len):idx]
            elif pos < pep_len:
                return ref[:idx]
            else:
                return ref_peptide
        else:
            suffix_len = len(downstream_peptide)
            if suffix_len < pep_len:
                # the downstream protein prediction includes
                # aa at the start position, so suffix_len += 1
                to_add = pep_len - suffix_len
                if pos < (pep_len+to_add):
                    epitope_from_ref = ref[:idx]
                elif pos >= (pep_len+to_add):
                    epitope_from_ref = ref[(idx - (pep_len+to_add)):idx]
            else:
                epitope_from_ref = ref_peptide

            len_wo_mut = pep_len - 1
            if prefix_len < len_wo_mut:
                to_add = len_wo_mut - prefix_len
                if suffix_len >= (pep_len+to_add):
                    epitope_from_mut = downstream_peptide[:(pep_len+to_add)]
                elif suffix_len < (pep_len+to_add):
                    epitope_from_mut = downstream_peptide
            else:
                epitope_from_mut = downstream_peptide[:pep_len]

            # print(
            #     f'{ref_peptide} / {epitope_from_ref} ({len(epitope_from_ref)}) + {epitope_from_mut} ({len(epitope_from_mut)})')
            epitope = epitope_from_ref + epitope_from_mut
            return epitope

    def get_nonsyn_only(self,
                        proteinseq_txt: str) -> list:
        df = txt_to_df(proteinseq_txt)

        nonsyn_ids = df.loc[df['Consequence']
                            != 'synonymous_variant', 'HGVSp']

        nonsyn_ids = [
            id for id in nonsyn_ids if id in self.proteinseqs_output.keys()
        ]
        if len(nonsyn_ids) == 0:
            nonsyn_list = dict()
        else:
            for key in nonsyn_ids:
                if key == '-':
                    drop_idx = nonsyn_ids[nonsyn_ids == '-'].index
                    nonsyn_ids = nonsyn_ids.drop(index=drop_idx)
            nonsyn_list = {
                key: self.proteinseqs_output.copy()[key] for key in nonsyn_ids
            }
        return nonsyn_list

    def get_sample_peptides(self,
                            tumor_sample_barcode: str,
                            frameshift: bool,
                            pep_len: int,
                            nonsyn_only: bool) -> dict:
        if frameshift:
            fs_output_df = txt_to_df(
                f'{FRAMESHIFT_DIR}/{tumor_sample_barcode}/{tumor_sample_barcode}_ds.txt')

            hgvsp_ids = list(fs_output_df['HGVSp'])
            id_w_mut = dict()
            for id in hgvsp_ids:
                if id != '-':
                    mut_pep = self.get_pep_fs(
                        hgvsp_id=id,
                        pep_len=pep_len)
                    id_w_mut.update(
                        {str(id): str(mut_pep)}
                    )
                else:
                    pass
        else:
            if nonsyn_only:
                hgvsp_seq = self.get_nonsyn_only(
                    f'{PROTEINSEQS_DIR}/{tumor_sample_barcode}/{tumor_sample_barcode}_ps.txt'
                )
            else:
                hgvsp_seq = self.proteinseqs_output.items().copy()

            id_w_mut = dict()
            # for row in hgvsp_seq:
            #     if len(row) != 1:
            #         print(row)

            if len(hgvsp_seq) != 0:
                for hgvsp_id, mut_seq in hgvsp_seq.items():
                    mut_pep = self.get_pep_nonfs(
                        mut=str(mut_seq),
                        hgvsp_id=str(hgvsp_id),
                        pep_len=pep_len)
                    id_w_mut.update(
                        {str(hgvsp_id): str(mut_pep)}
                    )

        return id_w_mut


 def save_pep_frags(mut_fasta_file: Union[str, os.PathLike[str]],
                   ds_output_file: Union[str, os.PathLike[str]],
                   ref_fasta_file: str,
                   frameshift: bool,
                   peptide_len: int,
                   nonsyn_only: bool) -> None:

    ps_pep_frag = ProteinSequence(
        ref_fasta_file=ref_fasta_file,
        ps_mut_fasta=mut_fasta_file,
        ds_output_file=ds_output_file
    )

    if frameshift:
        sample_fp = Path(ds_output_file)
        tumor_sample_barcode = (sample_fp.stem)[:-3]
    else:
        sample_fp = Path(mut_fasta_file)
        tumor_sample_barcode = (sample_fp.stem)[:-4]

    save_dir = str(sample_fp.parents[0])

    id_w_mut = ps_pep_frag.get_sample_peptides(
        tumor_sample_barcode=tumor_sample_barcode,
        frameshift=frameshift,
        pep_len=peptide_len,
        nonsyn_only=nonsyn_only
    )
    if len(id_w_mut) != 0:
        df = pd.DataFrame.from_dict(id_w_mut.items(), orient='columns')
        df.columns = ['HGVSp', 'Mut_Frags']

        df.to_pickle(
            f'{save_dir}/{tumor_sample_barcode}_pep_frags_nonsyn.pkl')
    else:
        pass
