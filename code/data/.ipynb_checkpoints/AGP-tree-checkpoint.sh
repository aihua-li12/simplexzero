#!/bin/bash

# Before running this file, first activate environment in terminal:
# conda activate qiime2-amplicon-2025.7
# Run this file after downloading AGP data


set -e

mkdir -p /home/jupyter/data/mapping/tree

cd /home/jupyter/data/mapping/tree




echo -e "\n\n================================== Start mapping ==================================\n\n"


echo -e "...... Download reference database ......\n"

wget https://data.qiime2.org/classifiers/sepp-ref-dbs/sepp-refs-gg-13-8.qza

echo -e "(\u221a) Finished\n\n\n"







# This never runs on any machine. Needs huge computing power and memory
echo -e "...... Creating a fragment insertion placement tree ......\n"

qiime fragment-insertion sepp \
  --i-representative-sequences ../rep_seqs.qza \
  --i-reference-database sepp-refs-gg-13-8.qza \
  --o-tree insertion-tree.qza \
  --o-placements insertion-placements.qza \
  --p-threads 14
  --verbose

echo -e "(\u221a) Finished\n\n\n"



echo -e "...... Exporting tree ......\n"

qiime tools export \
  --input-path insertion-tree.qza \
  --output-path exported-tree

echo -e "(\u221a) Finished\n\n\n"



echo "================================== Done =================================="



# To run this file, run the following in terminal:
# nohup /bin/bash ~/codes/AGP-tree.sh > ~/codes/AGP-tree.log 2>&1 &

# To check log:
# tail -f ~/codes/AGP-tree.log
