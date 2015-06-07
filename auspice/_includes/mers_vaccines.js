var	vaccineChoice = {};
var vaccineStrains = Object.keys(vaccineChoice);

var genome_annotation = {'ORF1a':[[1,1,1], [233, 6500, 13408]],
						 'ORF1b':[[1.2,1.2,1.2], [13387, 15000,  21468]],
						 'S':[[1,1,1], [21410, 23000, 25471]],
						 'ORF3':[[1,1,1], [ 25486, 25500, 25797]],
						 'ORF4a':[[1.2,1.2,1.2], [ 25806, 25900, 26135]],
						 'ORF4b':[[1,1,1], [ 26047, 26300, 26787]],
						 'ORF5':[[1,1,1], [ 26794, 27000, 27468]],
						 'E':[[1,1,1], [ 27544, 27600, 27792]],
						 'M':[[1,1,1], [ 27807, 28000, 28466]],
						 'N':[[1,1,1], [ 28520, 29000, 29761]],
						 'ORF8b':[[1,1,1], [ 28716, 28800, 29054]],
}
var restrictTo = {"country":"all","region":"all", "lab":"all", "host":"all"};

