perl -lpe 's/"/""/g; s/^|$/"/g; s/\t/","/g' < 437_AD_cadd_score.txt > 437_AD_cadd_score.csv

## substitute space with underscore
sed -i -- 's/ /_/g'  adni_kegg_snps.csv

## sub comma with tab
sed -i -- 's/,/\t/g'  adni_kegg_snps.csv

## further substitutions of space and ";" with tab for further analysis

sed -i -- 's/ /\t/g'  adni_kegg_snps.csv
sed -i -- 's/;/\t/g'  adni_kegg_snps.csv


## create .txt files for each 
mkdir pathways
awk '{ print $0 > pathways/$1.txt }' adni_kegg_snps.csv
