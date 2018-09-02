set boxwidth 0.5

set xrange [-0.5:10]
set boxwidth 0.25

set style fill solid 1.0 border rgb 'black'
set grid
set title "Accuracy results"

plot 'accuracy.dat' using ($0-.05):4:5:xtic(1) with boxerrorbars lc rgb 'coral' lw 1 title 'Real data', \
     '' using ($0+0.25):2:3 with boxerrorbars lc rgb 'grey90' lw 1 title 'Artificial data'

 set terminal png size 900,580 font "Helvetica"
 set output 'accuracy-with-error.png'

 replot
