set yrange [-1:80]

set boxwidth 0.25

set style fill solid 1.0 border rgb 'black'
set grid
set title "Execution time (s)"

set xlabel "Number of instances in dataset for repeated cross-validation (K = 5 and 5 repetitions)"
set ylabel "Time (s)"

plot "time.dat" using 1:2 title 'Voting' with lines lw 5,\
     "time.dat" using 1:3 title 'AdaBoost' with lines lw 5,\
     "time.dat" using 1:4 title 'Bagging' with lines lw 5,\
     "time.dat" using 1:5 title 'Stacking' with lines lw 5,\
     "time.dat" using 1:6 title 'SVM' with lines lw 5,\
     "time.dat" using 1:8 title 'DT' with lines lw 5,\
     "time.dat" using 1:9 title 'MLP' with lines lw 5,\

set terminal png size 1100,1000 font "Helvetica,20"
set output 'time.png'

replot
