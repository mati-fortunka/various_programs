set datafile separator ','

set terminal pngcairo size 1366,768 enhanced font 'Calibri,14'
set output 'GFP.png'

# style linii
set style line 1 lc rgb "#000000" lw 2 pt 7
set style line 2 lc rgb "#0000EE" lw 2 pt 7
set style line 3 lc rgb "#E61C66" lw 1.5 pt 7 dt 3

# definicja fitowanej funkcji i poczatkowe parametry
f(x)=V*x/(x+Km)
V = 0.9
Km = 4.1
Vmax = 0.99

# rozne tam
set title 'GFP-ssrA degradation'
set xlabel '[GFP-ssra] (uM)'
set ylabel 'Degradation rate (uM/min)'
set xrange [0:25]
set yrange [0:1.2]

# kreski dla Km, Vmax/2
set arrow from 4.12,0 to 4.12,0.495 nohead ls 3 
set arrow from 0,0.495 to 4.12,0.495 nohead ls 3 

# dopasowanie danych
fit f(x) "dane_GFP.csv" u 1:2 via V, Km

# pozycja legendy
set key right top
#unset key -- jesli ma byc bez legendy

# rysowanie
plot "dane_GFP.csv" u 1:2:3 with yerrorbars title "Data" ls 2, f(x) title "Fit" ls 1, Vmax ls 3 notitle
pause -1
