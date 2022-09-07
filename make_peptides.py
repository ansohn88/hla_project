import os
import re
from pathlib import Path
from typing import List, Union

import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser

"""
NON_FRAMESHIFT -- PROTEIN SEQS
# Uploaded_variation	Location	Allele	Gene	Feature	Feature_type	Consequence	cDNA_position	CDS_position	Protein_position	Amino_acids	Codons	Existing_variation	IMPACT	DISTANCE	STRAND	FLAGS	ENSP	HGVSc	HGVSp	HGVS_OFFSET
1_154148652_G/A	1:154148652	A	ENSG00000143549	ENST00000368530	Transcript	missense_variant	509	316	106	R/C	Cgc/Tgc	-	MODERATE	-	-1	-	ENSP00000357516	ENST00000368530.2:c.316C>T	ENSP00000357516.2:p.Arg106Cys	-
1_161206281_C/T	1:161206281	T	ENSG00000143257	ENST00000367980	Transcript	synonymous_variant	278	75	25	A	gcG/gcA	-	LOW	-	-1	-	ENSP00000356959	ENST00000367980.2:c.75G>A	ENSP00000356959.2:p.Ala25%3D	-
10_123810032_C/T	10:123810032	T	ENSG00000138162	ENST00000369005	Transcript	missense_variant	453	113	38	T/M	aCg/aTg	-	MODERATE	-	1	-	ENSP00000358001	ENST00000369005.1:c.113C>T	ENSP00000358001.1:p.Thr38Met	-
10_133967449_C/T	10:133967449	T	ENSG00000188385	ENST00000298622	Transcript	synonymous_variant	2307	2169	723	D	gaC/gaT	-	LOW	-	1	-	ENSP00000298622	ENST00000298622.4:c.2169C>T	ENSP00000298622.4:p.Asp723%3D	-
11_47380512_G/T	11:47380512	T	ENSG00000066336	ENST00000227163	Transcript	missense_variant	417	379	127	P/T	Cca/Aca	-	MODERATE	-	-1	-	ENSP00000227163	ENST00000227163.4:c.379C>A	ENSP00000227163.4:p.Pro127Thr	-
11_89868837_C/T	11:89868837	T	ENSG00000077616	ENST00000534061	Transcript	missense_variant,splice_region_variant	423	193	65	R/C	Cgt/Tgt	-	MODERATE	-	1	-	ENSP00000432481	ENST00000534061.1:c.193C>T	ENSP00000432481.1:p.Arg65Cys	-
"""

"""
FRAMESHIFT -- DOWNSTREAM
# Uploaded_variation	Location	Allele	Gene	Feature	Feature_type	Consequence	cDNA_position	CDS_position	Protein_position	Amino_acids	Codons	Existing_variation	IMPACT	DISTANCE	STRAND	FLAGS	ENSP	HGVSc	HGVSp	HGVS_OFFSET	DownstreamProtein	ProteinLengthChange
15_56993158_A/-	15:56993158	-	ENSG00000137871	ENST00000267807	Transcript	frameshift_variant	571	354	118	S/X	tcT/tc	-	HIGH	-	-1	-	ENSP00000267807	ENST00000267807.7:c.354del	ENSP00000267807.7:p.Val119LeufsTer12	-	SLLFSLFLNLVI	-849
17_31322643_C/-	17:31322643	-	ENSG00000141316	ENST00000269053	Transcript	frameshift_variant	321	251	84	S/X	tCc/tc	-	HIGH	-	1	-	ENSP00000269053	ENST00000269053.3:c.252del	ENSP00000269053.3:p.Ser85ValfsTer86	1	SVRPSSTVVVNWPECYMTSGWTDTGDTAWLTGSALLISQAVSTQLLWTTRLMGAPTTGSSRSTAGGGAATSPRTSPTCAGCTAQIC	-45
"""


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
                 ps_mut_fasta: Union[str, os.PathLike[str]],
                 ds_output_file: Union[str, os.PathLike[str]]
                 ) -> None:
        self.reference = self.get_fa_dict(ref_fasta_file)

        if ps_mut_fasta is not None:
            self.proteinseqs_output = self.get_fa_dict(str(ps_mut_fasta))
        if ds_output_file is not None:
            self.downstream_output = self.get_downstream_pred(
                str(ds_output_file))

        self.ds_inp_fp = ds_output_file

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
        ds_output = dict(zip(ds_df.HGVSp, ds_df.DownstreamProtein))
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
                     hgvsp_id: str) -> tuple[str, str]:
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
        if pos < pep_len:
            ref_peptide = ref[:idx]
        else:
            ref_peptide = ref[(idx - (pep_len - 1)):idx]
        prefix_len = len(ref_peptide)
        len_wo_mut = pep_len - 1
        total_frag_len = (2 * pep_len) - 1

        if is_nan or (hgvsp_id[-3:] == "Ter"):
            if pos > total_frag_len:
                return ref[(idx - total_frag_len):idx]
            elif pos <= total_frag_len:
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
                    epitope_from_ref = ref[idx - ((len_wo_mut+to_add)):idx]
            else:
                epitope_from_ref = ref_peptide

            if prefix_len < len_wo_mut:
                to_add = len_wo_mut - prefix_len
                if suffix_len >= (pep_len+to_add):
                    epitope_from_mut = downstream_peptide[:(pep_len+to_add)]
                elif suffix_len < (pep_len+to_add):
                    epitope_from_mut = downstream_peptide
            else:
                epitope_from_mut = downstream_peptide[:pep_len]

            # print(
            #     f'{ref_peptide} / {epitope_from_ref} ({len(epitope_from_ref)}) \
            #         + {epitope_from_mut} ({len(epitope_from_mut)})')

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
                if id == '-':
                    pass
                elif id[-5:] == 'Met1?':
                    pass
                else:
                    mut_pep = self.get_pep_fs(
                        hgvsp_id=id,
                        pep_len=pep_len)
                    id_w_mut.update(
                        {str(id): str(mut_pep)}
                    )
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
            f'{save_dir}/{tumor_sample_barcode}_pep_{peptide_len}_nonsyn.pkl')
    else:
        pass


if __name__ == '__main__':
    from joblib import Parallel, delayed

    ref_fp = '/home/asohn3/baraslab/hla/Data/proteinseqs_outputs/reference.fa'
    fs_root = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/frameshift'
    nofs_root = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/no_frameshift'

    # # NO FRAMESHIFT
    file_list = list(
        Path(nofs_root).rglob("*_mut.fa")
    )
    Parallel(n_jobs=20)(delayed(save_pep_frags)(mut_fa, None, ref_fp, False, 10, True)
                        for mut_fa in file_list)

    # for f in file_list:
    #     save_pep_frags(
    #         mut_fasta_file=f,
    #         ds_output_file=None,
    #         ref_fasta_file=ref_fp,
    #         frameshift=False,
    #         peptide_len=11,
    #         nonsyn_only=True
    #     )

    # # FRAMESHIFT
    file_list = list(
        Path(fs_root).rglob("*.txt")
    )
    Parallel(n_jobs=20)(delayed(save_pep_frags)(None, fs_inp, ref_fp, True, 10, True)
                        for fs_inp in file_list)

    # for f in file_list[:10]:
    #     save_pep_frags(
    #         mut_fasta_file=None,
    #         ds_output_file=f,
    #         ref_fasta_file=ref_fp,
    #         frameshift=True,
    #         peptide_len=11,
    #         nonsyn_only=True
    #     )

    # PEPER = ProteinSequence(
    #     ref_fasta_file=ref_fp,
    #     ps_mut_fasta=None,
    #     ds_output_file=f'{fs_root}/TCGA-CA-5797-01A-01D-1650-10/TCGA-CA-5797-01A-01D-1650-10_ds.txt'
    # )
    # ds_pep = PEPER.get_pep_fs(
    #     hgvsp_id='ENSP00000386149.2:p.Ser10HisfsTer16',
    #     pep_len=11
    # )
    # print(ds_pep)
