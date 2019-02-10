set xrange [-0.5:11]
set boxwidth 0.25

set style fill solid 1.0 border rgb 'black'
set grid
set title "Memory usage in KBytes"

set xlabel "Technique"
set ylabel "Memory usage (KBytes)"

plot 'memory-usage.dat' using ($0-.05):4:5:xtic(1) with boxerrorbars lc rgb "#5F9EA0" lw 1 title 'Real data', \
      '' using ($0+0.25):2:3 with boxerrorbars lc rgb "#2F4F4F" lw 1 title 'Artificial data', \

set terminal png size 900,580 font "Helvetica"
set output 'memory-usage.png'

replot
