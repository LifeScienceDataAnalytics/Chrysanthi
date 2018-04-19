for a in *
do
  python -c "import sys; print('\n'.join(' '.join(c) for c in zip(*(l.split() for l in sys.stdin.readlines() if l.strip()))))" < $a > $a.tr
  awk -f ~/scripts/snp2path/match.awk $a.tr ../437_AD_cadd_score.tr.csv > $a.cadd

done

