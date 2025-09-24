#!/bin/bash

# Before running this file, first activate environment in terminal:
# conda activate qiime2-amplicon-2025.7

# To run this file, run the following in terminal:
# nohup /bin/bash ~/code/data/download.sh > ~/code/data/download.log 2>&1 &

# To check log:
# tail -f ~/code/download.log


set -e

echo -e "\n\n============ Start downloading AGP dataset ============\n\n"




mkdir -p /home/jupyter/data/raw

cd /home/jupyter/data/raw



echo -e "...... Downloading contexts ......"

redbiom summarize contexts > contexts.txt

echo -e "(\u221a) Finished\n\n\n"





echo -e "...... Downloading sample ids ......"

redbiom search metadata "where qiita_study_id == 10317" | grep -vi "blank" > AGP.sampleids.txt

echo -e "(\u221a) Finished\n"

cat AGP.sampleids.txt | redbiom summarize samples --category sample_type






echo -e "\n\n\n...... Downloading abundance data ......"

export CTX=Deblur_2021.09-Illumina-16S-V4-100nt-50b3a2

redbiom fetch samples --from AGP.sampleids.txt --context $CTX --output AGP.rawASV.biom

echo -e "(\u221a) Finished\n"

biom summarize-table -i AGP.rawASV.biom | head






echo -e "\n\n\n...... Downloading metadata ......"

redbiom fetch sample-metadata --from AGP.sampleids.txt --output AGP.metadata.raw.tsv

echo -e "(\u221a) Finished\n\n"





echo "======================== Done ========================"




