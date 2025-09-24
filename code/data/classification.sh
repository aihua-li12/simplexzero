#!/bin/bash

# Before running this file, first activate environment in terminal:
# conda activate qiime2-amplicon-2025.7
# Run this file after downloading AGP data


# To run this file, run the following in terminal:
# nohup /bin/bash ~/code/data/classification.sh > ~/code/data/classification.log 2>&1 &

# To check log:
# tail -f ~/code/data/classification.log




set -e

mkdir -p /home/jupyter/data/classification

cd /home/jupyter/data/classification

echo -e "\n\n================================== Start mapping ==================================\n\n"


echo -e "...... Converting the biom data to a QIIME 2 artifact and QIIME 2 visualization ......\n"

qiime tools import --input-path ../matched/AGP.matchedASV.biom --type 'FeatureTable[Frequency]' --input-format BIOMV210Format --output-path feature-table-AGP.qza

qiime feature-table summarize --i-table feature-table-AGP.qza --o-visualization feature-table-AGP.qzv --m-sample-metadata-file ../matched/AGP.metadata.matched.tsv

echo -e "(\u221a) Finished\n\n\n"



echo -e "...... Generating representative sequences ......\n"

biom summarize-table --observations -i ../matched/AGP.matchedASV.biom | tail -n +16 | awk -F ':' '{print ">"$1"\n"$1}' > rep_seqs.fna

echo -e "(\u221a) Finished\n\n\n"




echo -e "...... Converting the representative sequences into an artifact form ......\n"

qiime tools import --input-path rep_seqs.fna --output-path rep_seqs.qza --type 'FeatureData[Sequence]'

echo -e "(\u221a) Finished\n\n\n"




# Download a pre-trained classifer from https://library.qiime2.org/data-resources

echo -e "...... Download a pre-trained classifier ......\n"

wget https://data.qiime2.org/classifiers/sklearn-1.4.2/silva/silva-138-99-nb-classifier.qza

echo -e "(\u221a) Finished\n\n\n"





echo -e "...... Classify the sequences using the pre-trained classifier ......\n"

qiime feature-classifier classify-sklearn \
    --i-classifier silva-138-99-nb-classifier.qza \
    --i-reads rep_seqs.qza \
    --o-classification taxonomy.qza

echo -e "(\u221a) Finished\n\n\n"



echo -e "...... Visualize into .qzv tabulate ......\n"

# put on https://view.qiime2.org to visualize
qiime metadata tabulate --m-input-file taxonomy.qza --o-visualization taxonomy.qzv

echo -e "(\u221a) Finished\n\n\n"





echo -e "...... Exporting the taxonomy as TSV file ......\n"

qiime tools export --input-path taxonomy.qza --output-path exported-taxonomy

echo -e "(\u221a) Finished\n\n\n"






echo "================================== Done =================================="



