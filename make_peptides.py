import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser

"""
NON_FRAMESHIFT -- PROTEIN SEQS
#Uploaded_variation	Location	Allele	Gene	Feature	Feature_type	Consequence	cDNA_position	CDS_position	Protein_position	Amino_acids	Codons	Existing_variation	IMPACT	DISTANCE	STRAND	FLAGS	ENSP	HGVSc	HGVSp	HGVS_OFFSET
1_154148652_G/A	1:154148652	A	ENSG00000143549	ENST00000368530	Transcript	missense_variant	509	316	106	R/C	Cgc/Tgc	-	MODERATE	-	-1	-	ENSP00000357516	ENST00000368530.2:c.316C>T	ENSP00000357516.2:p.Arg106Cys	-
1_161206281_C/T	1:161206281	T	ENSG00000143257	ENST00000367980	Transcript	synonymous_variant	278	75	25	A	gcG/gcA	-	LOW	-	-1	-	ENSP00000356959	ENST00000367980.2:c.75G>A	ENSP00000356959.2:p.Ala25%3D	-
10_123810032_C/T	10:123810032	T	ENSG00000138162	ENST00000369005	Transcript	missense_variant	453	113	38	T/M	aCg/aTg	-	MODERATE	-	1	-	ENSP00000358001	ENST00000369005.1:c.113C>T	ENSP00000358001.1:p.Thr38Met	-
10_133967449_C/T	10:133967449	T	ENSG00000188385	ENST00000298622	Transcript	synonymous_variant	2307	2169	723	D	gaC/gaT	-	LOW	-	1	-	ENSP00000298622	ENST00000298622.4:c.2169C>T	ENSP00000298622.4:p.Asp723%3D	-
11_47380512_G/T	11:47380512	T	ENSG00000066336	ENST00000227163	Transcript	missense_variant	417	379	127	P/T	Cca/Aca	-	MODERATE	-	-1	-	ENSP00000227163	ENST00000227163.4:c.379C>A	ENSP00000227163.4:p.Pro127Thr	-
11_89868837_C/T	11:89868837	T	ENSG00000077616	ENST00000534061	Transcript	missense_variant,splice_region_variant	423	193	65	R/C	Cgt/Tgt	-	MODERATE	-	1	-	ENSP00000432481	ENST00000534061.1:c.193C>T	ENSP00000432481.1:p.Arg65Cys	-
"""

"""
FRAMESHIFT -- DOWNSTREAM
#Uploaded_variation	Location	Allele	Gene	Feature	Feature_type	Consequence	cDNA_position	CDS_position	Protein_position	Amino_acids	Codons	Existing_variation	IMPACT	DISTANCE	STRAND	FLAGS	ENSP	HGVSc	HGVSp	HGVS_OFFSET	DownstreamProtein	ProteinLengthChange
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
                      hgvsp_id: str) -> str:
        _, aa_change = self.get_mut_info(hgvsp_id)
        pos = self.get_mut_pos(aa_change)
        # the HGVSp position starts w/ 1-index; make into 0-index for arr slicing
        pos = pos - 1

        if 'Ter' in aa_change:
            return mut[(pos - 30):(pos + 1)]
        else:
            # print(
            #     f'{hgvsp_id}/{pos}/{len(mut[pos-15:pos])}/{len(mut[pos+1:pos+16])}')
            # return mut[(pos - 15):(pos + 16)]
            suffix_len = len(mut[(pos + 1):(pos + 16)])
            if suffix_len < 15:
                to_add = 15 - suffix_len
                return mut[(pos - (15 + to_add)):(pos + 16)]
            else:
                return mut[(pos - 15):(pos + 16)]

    def get_pep_fs(self,
                   peptide_len: int,
                   hgvsp_id: str) -> str:
        ensembl_id, aa_change = self.get_mut_info(hgvsp_id)
        pos = self.get_mut_pos(aa_change)

        ref = self.reference[ensembl_id]
        downstream_peptide = self.downstream_output[ensembl_id]

        epitope_from_ref = ref[(pos - peptide_len):pos]
        epitope_from_mut = downstream_peptide[:peptide_len]
        epitope = epitope_from_ref + epitope_from_mut

        return epitope

    def get_sample_peptides(self,
                            tumor_sample_barcode: str,
                            frameshift: bool) -> None:
        if frameshift:
            fs_output_df = txt_to_df(
                f'{FRAMESHIFT_DIR}/{tumor_sample_barcode}/{tumor_sample_barcode}_ds.txt')

            hgvsp_ids = list(fs_output_df['HGVSp'])
            id_w_mut = dict()
            for id in hgvsp_ids:
                mut_pep = self.get_pep_fs(peptide_len=16, hgvsp_id=id)
                id_w_mut.update(
                    {str(id): str(mut_pep)}
                )
        else:
            id_w_mut = dict()
            for (hgvsp_id, mut_seq) in self.proteinseqs_output.items():
                mut_pep = self.get_pep_nonfs(
                    mut=str(mut_seq), hgvsp_id=str(hgvsp_id)
                )
                id_w_mut.update(
                    {str(hgvsp_id): str(mut_pep)}
                )

        return id_w_mut


def save_non_fs_fragments(mut_fasta_file: str,
                          ref_fasta_file: str) -> None:
    ps_pep_frag = ProteinSequence(
        ref_fasta_file=ref_fasta_file,
        ps_mut_fasta=mut_fasta_file,
        ds_output_file=None)

    tumor_sample_barcode = (mut_fasta_file.stem)[:-4]
    save_dir = str(mut_fasta_file.parents[0])

    id_w_mut = ps_pep_frag.get_sample_peptides(
        tumor_sample_barcode=tumor_sample_barcode,
        frameshift=False
    )
    if len(id_w_mut) > 0:
        df = pd.DataFrame.from_dict(id_w_mut.items(), orient='columns')
        df.columns = ['HGVSp', 'Mut_Frags']

        df.to_pickle(
            f'{save_dir}/{tumor_sample_barcode}_peptide_fragments.pkl')
    else:
        pass


# def save_fs_fragments(ds_output_file: str,
#                       ref_fasta_file: str) -> None:
#     TODO


if __name__ == '__main__':
    from joblib import Parallel, delayed

    ref_fp = '/home/asohn3/baraslab/hla/Data/vep_outputs/reference.fa'
    fs_root = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/frameshift'
    nofs_root = '/home/asohn3/baraslab/hla/Data/vep/split_by_samples/no_frameshift'

    # ps_pep = ProteinSequence(
    #     ref_fasta_file=ref_fp,
    #     ps_mut_fasta=f'{nofs_root}/TCGA-02-0047-01A-01D-1490-08/TCGA-02-0047-01A-01D-1490-08_mut.fa',
    #     ds_output_file=None
    # )
    # out_dict = ps_pep.get_sample_peptides(
    #     tumor_sample_barcode='TCGA-02-0047-01A-01D-1490-08',
    #     frameshift=False
    # )

    # ProteinSeqs mutant fasta files
    mut_fasta_list = list(
        Path(nofs_root).rglob("*_mut.fa")
    )

    Parallel(n_jobs=10)(delayed(save_non_fs_fragments)(fp, ref_fp)
                        for fp in mut_fasta_list)
