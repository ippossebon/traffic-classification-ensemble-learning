set yrange [0:70]

set boxwidth 0.25

set style fill solid 1.0 border rgb 'black'
set grid
set title "Execution time (ms)"

set xlabel "Number of instances to classify"
set ylabel "Time (ms)"

plot "time.dat" using 1:2 title 'Voting' with lines lw 5,\
     "time.dat" using 1:3 title 'AdaBoost' with lines lw 5,\
     "time.dat" using 1:4 title 'Bagging' with lines lw 5,\
     "time.dat" using 1:5 title 'Stacking' with lines lw 5,\
     "time.dat" using 1:6 title 'SVM' with lines lw 5,\
     "time.dat" using 1:7 title 'KNN' with lines lw 5,\
     "time.dat" using 1:8 title 'Decision Tree' with lines lw 5,\

set terminal png size 900,580 font "Helvetica"
set output 'time.png'

replot
