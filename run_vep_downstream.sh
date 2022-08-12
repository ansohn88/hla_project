# $1 input, $2 output

export VEP=/home/abaras1/NGS/ensembl-vep/vep
export VEP_DATA=/home/abaras1/NGS/ensembl-vep/vep-data
export VEP_PATH=/home/abaras1/NGS/ensembl-vep/vep-path
export PERL5LIB=$VEP_PATH:$PERL5LIB
export PATH=$VEP_PATH/htslib:$PATH
export BUILD=GRCh37
export FASTA=/home/abaras1/NGS/ensembl-vep/vep-data/homo_sapiens/101_$BUILD/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz

export FS_VEP=/home/asohn3/baraslab/hla/Data/vep/split_by_samples/frameshift

# $VEP --input_file $1 --output_file $2 --no_stats --force_overwrite --cache --offline \
# --dir $VEP_DATA --species homo_sapiens \
# --assembly $BUILD --hgvs --coding_only --terms SO --symbol --canonical --pick \
# --fasta $FASTA --use_given_ref


EXT=.vep
cd $FS_VEP
for i in $(ls); do
    if [[ $i == *$EXT ]]; then
        FNAME=$(basename $i $EXT)
        $VEP --input_file $FS_VEP/$i --output_file $FS_VEP/${FNAME}_ds.txt \
        --tab \
        --species homo_sapiens \
        --cache \
        --offline \
        --assembly $BUILD \
        --fasta $FASTA \
        --use_given_ref \
        --dir_cache $VEP_DATA \
        --dir_plugins $VEP_DATA/Plugins \
        --force_overwrite \
        --database \
        --coding_only \
        --pick \
        --protein \
        --hgvs \
        --plugin Downstream
    fi
done