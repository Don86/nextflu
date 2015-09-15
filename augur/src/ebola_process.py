import time, re, os
from virus_filter import virus_filter
from virus_clean import virus_clean
from tree_refine import tree_refine
from process import process, virus_config
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
import numpy as np
from itertools import izip

# numbering starting at methionine including the signal peptide
sp = 17


virus_config.update({
	# data source and sequence parsing/cleaning/processing
	'virus':'ebola',
	'fasta_fields':{2:'strain', 3:'lab', 4:'country', 5:'region', 8:'date'},
	#>EBOV|VHFC2|G6091.1|KR105346|SLE|Tonkolili|Tani||2014-09-27
	#>EBOV|EMLab-RT|EM_COY_2015_013576||GIN|Coyah|?|MinION|2015-03-09
	'alignment_file':'data/ebola.fasta',
	'outgroup':'EM_079404',
	'aggregate_regions':[('global', None)],
	'force_include_all':False,
	'max_global':True,   # sample as evenly as possible from different geographic regions 
	'cds':[54,None], # define the HA start i n 0 numbering
	# define relevant clades in canonical HA1 numbering (+1)
	# numbering starting at methionine including the signal peptide
	'clade_designations': {},
	'min_mutation_frequency':0.499,
	'min_genotype_frequency':0.499,
	'auspice_prefix':'ebola_',
	'html_vars': {'coloring': 'lbi, dfreq, region, date',
				   'gtplaceholder': 'Genomic positions...',
					'freqdefault': ''},
	'js_vars': {'LBItau': 0.0005, 'LBItime_window': 0.5, 'dfreq_dn':2},
	})


class ebola_filter(virus_filter):
	def __init__(self,min_length = 987, **kwargs):
		'''
		parameters
		min_length  -- minimal length for a sequence to be acceptable
		'''
		virus_filter.__init__(self, **kwargs)
		self.min_length = min_length
		self.vaccine_strains =[]
		self.outgroup = {
			'strain': 'EM_079404',
			'lab': 'EMLab',
			'country': 'GUI',
			'region': 'Gueckedou',
			'date': '2014-03-28',
			'seq': '--GACACACAAAAAGAAAGAAGAATTTTTAGGATCTTTTGTGTGCGAATAACTATGAGGAAGATTAATAATTTTCCTCTCATTGAAATTTATATCGGAATTTAAATTGAAATTGTTACTGTAATCATACCTGGTTTGTTTCAGAGCCATATCACCAAGATAGAGAACAACCTAGGTCTCCGGAGGGGGCAAGGGCATCAGTGTGCTCAGTTGAAAATCCCTTGTCAACATCTAGGCCTTATCACATCACAAGTTCCGCCTTAAACTCTGCAGGGTGATCCAACAACCTTAATAGCAACATTATTGTTAAAGGACAGCATTAGTTCACAGTCAAACAAGCAAGATTGAGAATTAACTTTGATTTTGAACCTGAACACCCAGAGGACTGGAGACTCAACAACCCTAAAGCCTGGGGTAAAACATTAGAAATAGTTTAAAGACAAATTGCTCGGAATCACAAAATTCCGAGTATGGATTCTCGTCCTCAGAAAGTCTGGATGACGCCGAGTCTCACTGAATCTGACATGGATTACCACAAGATCTTGACAGCAGGTCTGTCCGTTCAACAGGGGATTGTTCGGCAAAGAGTCATCCCAGTGTATCAAGTAAACAATCTTGAGGAAATTTGCCAACTTATCATACAGGCCTTTGAAGCTGGTGTTGATTTTCAAGAGAGTGCGGACAGTTTCCTTCTCATGCTTTGTCTTCATCATGCGTACCAAGGAGATTACAAACTTTTCTTGGAAAGTGGCGCAGTCAAGTATTTGGAAGGGCACGGGTTCCGTTTTGAAGTCAAGAAGCGTGATGGAGTGAAGCGCCTTGAGGAATTGCTGCCAGCAGTATCTAGTGGGAGAAACATTAAGAGAACACTTGCTGCCATGCCGGAAGAGGAGACGACTGAAGCTAATGCCGGTCAGTTCCTCTCCTTTGCAAGTCTATTCCTTCCGAAATTGGTAGTAGGAGAAAAGGCTTGCCTTGAGAAGGTTCAAAGGCAAATTCAAGTACATGCAGAGCAAGGACTGATACAATATCCAACAGCTTGGCAATCAGTAGGACACATGATGGTGATTTTCCGTTTGATGCGAACAAATTTTTTGATCAAATTTCTTCTAATACACCAAGGGATGCACATGGTTGCCGGACATGATGCCAACGATGCTGTGATTTCAAATTCAGTGGCTCAGGCTCGTTTTTCAGGTCTATTGATTGTCAAAACAGTACTTGATCATATCCTACAAAAGACAGAACGAGGAGTTCGTCTCCATCCTCTTGCAAGGACCGCCAAGGTAAAAAATGAGGTGAACTCCTTCAAGGCTGCACTCAGCTCCCTGGCCAAGCATGGAGAGTATGCTCCTTTCGCCCGACTTTTGAACCTTTCTGGAGTAAATAATCTTGAGCATGGTCTTTTCCCTCAACTGTCGGCAATTGCACTCGGGGTCGCCACAGCCCACGGGAGCACCCTCGCAGGAGTAAATGTTGGAGAACAGTATCAACAGCTCAGAGAGGCAGCCACTGAGGCTGAGAAGCAACTCCAACAATATGCGGAGTCTCGTGAACTTGACCATCTTGGACTTGATGATCAGGAAAAGAAAATTCTTATGAACTTCCATCAGAAAAAGAACGAAATCAGCTTCCAGCAAACAAACGCGATGGTAACTCTAAGAAAAGAGCGCCTGGCCAAGCTGACAGAAGCTATCACTGCTGCATCACTGCCCAAAACAAGTGGACATTACGATGATGATGACGACATTCCCTTTCCAGGACCCATCAATGATGACGACAATCCTGGCCATCAAGATGATGATCCGACTGACTCACAGGATACGACCATTCCCGATGTGGTAGTTGATCCCGATGATGGAGGCTACGGCGAATACCAAAGTTACTCGGAAAACGGCATGAGTGCACCAGATGACTTGGTCCTATTCGATCTAGACGAGGACGACGAGGACACCAAGCCAGTGCCTAACAGATCGACCAAGGGTGGACAACAGAAAAACAGTCAAAAGGGCCAGCATACAGAGGGCAGACAGACACAATCCACGCCAACTCAAAACGTCACAGGCCCTCGCAGAACAATCCACCATGCCAGTGCTCCACTCACGGACAATGACAGAAGAAACGAACCCTCCGGCTCAACCAGCCCTCGCATGCTGACCCCAATCAACGAAGAGGCAGACCCACTGGACGATGCCGACGACGAGACGTCTAGCCTTCCGCCCTTAGAGTCAGATGATGAAGAACAGGACAGGGACGGAACTTCTAACCGCACACCCACTGTCGCCCCACCGGCTCCCGTATACAGAGATCACTCCGAAAAGAAAGAACTCCCGCAAGATGAACAACAAGATCAGGACCACATTCAAGAGGCCAGGAACCAAGACAGTGACAACACCCAGCCAGAACATTCTTTTGAGGAGATGTATCGCCACATTCTAAGATCACAGGGGCCATTTGATGCCGTTTTGTATTATCATATGATGAAGGATGAGCCTGTAGTTTTCAGTACCAGTGATGGTAAAGAGTACACGTATCCGGACTCCCTTGAAGAGGAATATCCACCATGGCTCACTGAAAAAGAGGCCATGAATGATGAGAATAGATTTGTTACACTGGATGGTCAACAATTTTATTGGCCAGTAATGAATCACAGGAATAAATTCATGGCAATCCTGCAACATCATCAGTGAATGAGCATGTAATAATGGGATGATTTAATCGACAAATAGCTAACATTAAATAGTCAAGGAACGCAAACAGGAAGAATTTTTGATGTCTAAGGTGTGAATTATTATCACAATAAAAGTGATTCTTAGTTTTGAATTTAAAGCTAGCTTATTATTACTAGCCGTTTTTCAAAGTTCAATTTGAGTCTTAATGCAAATAAGCGTTAAGCCACAGTTATAGCCATAATGGTAACTCAATATCTTAGCCAGCGATTTATCTAAATTAAATTACATTATGCTTTTATAACTTACCTACTAGCCTGCCCAACATTTACACGATCGTTTTATAATTAAGAAAAAACTAATGATGAAGATTAAAACCTTCATCATCCTTACGTCAATTGAATTCTCTAGCACTAGAAGCTTATTGTCTTCAATGTAAAAGAAAAGCTGGCCTAACAAGATGACAACTAGAACAAAGGGCAGGGGCCATACTGTGGCCACGACTCAAAACGACAGAATGCCAGGCCCTGAGCTTTCGGGCTGGATCTCTGAGCAGCTAATGACCGGAAGGATTCCTGTAAACGACATCTTCTGTGATATTGAGAACAATCCAGGATTATGCTACGCATCCCAAATGCAACAAACGAAGCCAAACCCGAAGATGCGCAACAGTCAAACCCAAACGGACCCAATTTGCAATCATAGTTTTGAGGAGGTAGTACAAACATTGGCTTCATTGGCTACTGTTGTGCAACAACAAACCATCGCATCAGAATCATTAGAACAACGCATTACGAGTCTTGAGAATGGTCTAAAGCCAGTTTATGATATGGCAAAAACAATCTCCTCATTGAACAGGGTTTGTGCTGAGATGGTTGCAAAATATGATCTTCTGGTGATGACAACCGGTCGGGCAACAGCAACCGCTGCGGCAACTGAGGCTTATTGGGCTGAACATGGTCAACCACCACCTGGACCATCACTTTATGAAGAAAGTGCGATTCGGGGTAAGATTGAATCTAGAGATGAGACTGTCCCTCAAAGTGTTAGGGAGGCATTCAACAATCTAGACAGTACCACTTCACTAACTGAGGAAAATTTTGGGAAACCTGACATTTCGGCAAAGGATTTGAGAAACATTATGTATGATCACTTGCCTGGTTTTGGAACTGCTTTCCACCAATTAGTACAAGTGATTTGTAAATTGGGAAAAGATAGCAATTCATTGGACATTATTCATGCTGAGTTCCAGGCCAGCCTGGCTGAAGGAGACTCCCCTCAATGTGCCCTAATTCAAATTACAAAAAGAGTTCCAATCTTCCAAGATGCTGCTCCACCTGTCATCCACATCCGCTCTCGAGGTGACATTCCCCGAGCTTGCCAGAAGAGCTTGCGTCCAGTCCCACCATCACCCAAGATTGATCGAGGTTGGGTATGTGTTTTTCAGTTTCAAGATGGTAAAACACTTGGACTCAAAATTTGAGCCAATCTCTTTTCCCTCCGAAAGAGGCAACTAATAGCAGAGGCTTCAACTGCTGAACTATAGGGTATGTTACATTAATGATACACTTGTGAGTATCAGCCCTAGATAATATAAGTCAATTAAACAACCAAGATAAAATTGTTCATATCCCGCTAGCAGCTTTAAAGATAAATGTAATAGGAGCTATACCTCTGACAGTATTATAATTAATTGTTATTAAGTAACCCAAACCAAAAATGATGAAGATTAAGAAAAACCTACCTCGACTGAGAGAGTGTTTTTTCATTAACCTTCATCTTGTAAACGTTGAGCAAAATTGTTAAAAATATGAGGCGGGTTATATTGCCTACTGCTCCTCCTGAATATATGGAGGCCATATACCCTGCCAGGTCAAATTCAACAATTGCTAGGGGTGGCAACAGCAATACAGGCTTCCTGACACCGGAGTCAGTCAATGGAGACACTCCATCGAATCCACTCAGGCCAATTGCTGATGACACCATCGACCATGCCAGCCACACACCAGGCAGTGTGTCATCAGCATTCATCCTCGAAGCTATGGTGAATGTCATATCGGGCCCCAAAGTGCTAATGAAGCAAATTCCAATTTGGCTTCCTCTAGGTGTCGCTGATCAAAAGACCTACAGCTTTGACTCAACTACGGCCGCCATCATGCTTGCTTCATATACTATCACCCATTTCGGCAAGGCAACCAATCCGCTTGTCAGAGTCAATCGGCTGGGTCCTGGAATCCCGGATCACCCCCTCAGGCTCCTGCGAATTGGAAACCAGGCTTTCCTCCAGGAGTTCGTTCTTCCACCAGTCCAACTACCCCAGTATTTCACCTTTGATTTGACAGCACTCAAACTGATCACTCAACCACTGCCTGCTGCAACATGGACCGATGACACTCCAACTGGATCAAATGGAGCGTTGCGTCCAGGAATTTCATTTCATCCAAAACTTCGCCCCATTCTTTTACCCAACAAAAGTGGGAAGAAGGGGAACAGTGCCGATCTAACATCTCCGGAGAAAATCCAAGCAATAATGACTTCACTCCAGGACTTTAAGATCGTTCCAATTGATCCAACCAAAAATATCATGGGTATCGAAGTGCCAGAAACTCTGGTCCACAAGCTGACCGGTAAGAAGGTGACTTCCAAAAATGGACAACCAATCATCCCTGTTCTTTTGCCAAAGTACATTGGGTTGGACCCGGTGGCTCCAGGAGACCTCACCATGGTAATCACACAGGATTGTGACACGTGTCATTCTCCTGCAAGTCTTCCAGCTGTGGTTGAGAAGTAATTGCAATAATTGACTCAGATCCAGTTTTACAGAATCTTCTCAGGGATAGTGATAACATCTTTTTAATAATCCGTCTACTAGAAGAGATACTTCTAATTGATCAATATACTAAAGGTGCTTTACACCATTGTCTCTTTTCTCTCCTAAATGTAGAGCTTAACAAAAGACTCATAATATACCTGTTTTTAAAAGATTGATTGATGAAAGATCATGACTAATAACATTACAAACAATCCTACTATAATCAATACGGTGATTCAAATGTCAATCTTTCTCATTGCACATACTCTTTGTCCTTATCCTCAAATTGCCTACATGCTTACATCTGAGGACAGCCAGTGTGACTTGGATTGGAGATGTGGAGGAAAAATCGGGGCCCATTTCTAAGTTGTTCACAATCTAAGTACAGACATTGCTCTTCTAATTAAGAAAAAATCGGCGATGAAGATTAAGCCGACAGTGAGCGTAATCTTCATCTCTCTTAGATTATTTGTCTTCCAGAGTAGGGGTCATCAGGTCCTTTTCAATTGGATAACCAAAATAAGCTTCACTAGAAGGATATTGTGAGGCGACAACACAATGGGTGTTACAGGAATATTGCAGTTACCTCGTGATCGATTCAAGAGGACATCATTCTTTCTTTGGGTAATTATCCTTTTCCAAAGAACATTTTCCATCCCGCTTGGAGTTATCCACAATAGTACATTACAGGTTAGTGATGTCGACAAACTAGTTTGTCGTGACAAACTGTCATCCACAAATCAATTGAGATCAGTTGGACTGAATCTCGAGGGGAATGGAGTGGCAACTGACGTGCCATCTGCGACTAAAAGATGGGGCTTCAGGTCCGGTGTCCCACCAAAGGTGGTCAATTATGAAGCTGGTGAATGGGCTGAAAACTGCTACAATCTTGAAATCAAAAAACCTGACGGGAGTGAGTGTCTACCAGCAGCGCCAGACGGGATTCGGGGCTTCCCCCGGTGCCGGTATGTGCACAAAGTATCAGGAACGGGACCATGTGCCGGAGACTTTGCCTTCCACAAAGAGGGTGCTTTCTTCCTGTATGATCGACTTGCTTCCACAGTTATCTACCGAGGAACGACTTTCGCTGAAGGTGTCGTTGCATTTCTGATACTGCCCCAAGCTAAGAAGGACTTCTTCAGCTCACACCCCTTGAGAGAGCCGGTCAATGCAACGGAGGACCCGTCGAGTGGCTATTATTCTACCACAATTAGATATCAGGCTACCGGTTTTGGAACTAATGAGACAGAGTACTTGTTCGAGGTTGACAATTTGACCTACGTCCAACTTGAATCAAGATTCACACCACAGTTTCTGCTCCAGCTGAATGAGACAATATATGCAAGTGGGAAGAGGAGCAACACCACGGGAAAACTAATTTGGAAGGTCAACCCCGAAATTGATACAACAATCGGGGAGTGGGCCTTCTGGGAAACTAAAAAAACCTCACTAGAAAAATTCGCAGTGAAGAGTTGTCTTTCACAGCTGTATCAAACGGACCCAAAAACATCAGTGGTCAGAGTCCGGCGCGAACTTCTTCCGACCCAGAGACCAACACAACAAATGAAGACCACAAAATCATGGCTTCAGAAAATTCCTCTGCAATGGTTCAAGTGCACAGTCAAGGAAGGAAAGCTGCAGTGTCGCATCTGACAACCCTTGCCACAATCTCCACGAGTCCTCAACCTCCCACAACCAAAACAGGTCCGGACAACAGCACCCATAATACACCCGTGTATAAACTTGACATCTCTGAGGCAACTCAAGTTGGACAACATCACCGTAGAGCAGACAACGACAGCACAGCCTCCGACACTCCCCCCGCCACGACCGCAGCCGGACCCTTAAAAGCAGAGAACACCAACACGAGTAAGAGCGCTGACTCCCTGGACCTCGCCACCACGACAAGCCCCCAAAACTACAGCGAGACTGCTGGCAACAACAACACTCATCACCAAGATACCGGAGAAGAGAGTGCCAGCAGCGGGAAGCTAGGCTTAATTACCAATACTATTGCTGGAGTAGCAGGACTGATCACAGGCGGGAGAAGGACTCGAAGAGAAGTAATTGTCAATGCTCAACCCAAATGCAACCCCAATTTACATTACTGGACTACTCAGGATGAAGGTGCTGCAATCGGATTGGCCTGGATACCATATTTCGGGCCAGCAGCCGAAGGAATTTACACAGAGGGGCTAATGCACAACCAAGATGGTTTAATCTGTGGGTTGAGGCAGCTGGCCAACGAAACGACTCAAGCTCTCCAACTGTTCCTGAGAGCCACAACTGAGCTGCGAACCTTTTCAATCCTCAACCGTAAGGCAATTGACTTCCTGCTGCAGCGATGGGGTGGCACATGCCACATTTTGGGACCGGACTGCTGTATCGAACCACATGATTGGACCAAGAACATAACAGACAAAATTGATCAGATTATTCATGATTTTGTTGATAAAACCCTTCCGGACCAGGGGGACAATGACAATTGGTGGACAGGATGGAGACAATGGATACCGGCAGGTATTGGAGTTACAGGTGTTATAATTGCAGTTATCGCTTTATTCTGTATATGCAAATTTGTCTTTTAGTCTTTCTTCAGATTGTTTCACGGCAAAACTCAACCTCAAATCAATGAAACTAGGATTTAATTATATGAATCACTTGAATCTAAGATTACTTGACAAATGATAACATAATACACTGGAGCTTCAAACATAGCCAATGTGATTCTAACTCCTTTAAACTCACAGTTAATCATAAACAAGGTTTGACATCAATCTAGCTATATCTTTAAGAATGATAAACTTGATGAAGATTAAGAAAAAGGTAATCTTTCGATTATCTTTAGTCTTCATCCTTGATTCTACAATCATGACAGTTGTCTTTAATGAAAAAGGAAAAAAGCCTTTTTATTAAGTTGTAATAATCAGATCTGCAAACCGGTAGAATTTAGTTGTAACCTAACACACACAAAGCATTGGTAAAAAAGTCAATAGAAATTTAAACAGTGAGTGCAGACAACTCTTAAATGGAAGCTTCATATGAGAGAGGACGCCCCCGAGCTGCCAGACAGCATTCAAGGGATGGACACGACCACCATGTTCGAGCACGATCATCATCCAGAGAGAATTATCGAGGTGAGTACCGTCAATCAAGGAGCGCCTCACAAGTGCGCGTTCCTACTGTATTTCATAAGAAGAGAGTTGAACCATTAACAGTTCCTCCAGCACCTAAAGACATATGTCCGACCTTGAAAAAAGGATTTTTGTGTGACAGTAGTTTTTGCAAAAAAGACCACCAGTTAGAAAGTTTAACTGATAGGGAATTACTCCTACTAATCGCCCGTAAGACTTGTGGATCAGTAGAACAACAATTAAATATAACTGCACCCAAGGACTCGCGCTTAGCAAATCCAACGGCTGATGATTTCCAGCAAGAGGAAGGTCCAAAAATTACCTTGTTGACACTGATCAAGACGGCAGAACACTGGGCGAGACAAGACATCCGAACCATAGAGGATTCCAAATTAAGGGCATTGTTAACTCTATGTGCTGTGATGACGAGGAAATTCTCAAAATCCCAGCTGAGTCTTTTGTGTGAGACACACCTAAGGCGCGAAGGGCTTGGGCAAGATCAGGCAGAACCCGTTCTCGAAGTATATCAACGATTACACAGTGATAAAGGAGGCAGTTTTGAAGCTGCACTATGGCAACAATGGGACCGACAATCCCTAATTATGTTTATCACTGCATTCTTGAATATCGCTCTCCAGTTACCGTGTGAAAGTTCTGCTGTCGTTGTTTCAGGGTTAAGAACATTGGTTCCTCAATCAGATAATGAGGAAGCTTCAACCAACCCGGGGACATGCTCATGGTCTGATGAGGGTACCCCTTAATAAGGCTGACTAAAACACTATATAACCTTCTACTTGATCACAATACTCCGTATACCTATCATCATATATTTAATCAAGACGATATCCTTTAAAACTTATTCAGTACTATAATCACTCTCATTTCAAATTGATAAGATATGCATAATTGCCTTAATATATAAAGAGGTATGATATAACCCAAACATTGACCAAAGAAAATCATAATCTCGTATCGCTCGCAATATAACCTGCCAAGCATACCTCTTGCACAAAGTGATTCTTGTACACAAATAATGTTTGACTCTACAGGAGGTAGCAACGATCCATCTCATCAAAAAATAAGTATTTTATGATTTACTAATGATCTCTTAAAATATTAAGAAAAACTGACGGAACATAAATTCTTTCTGCTTCAAGTTGTGGAGGAGGTCTATGGTATTCGCTATTGTTATATTACAATCAATAACAAGCTTGTAAAAATATTGTTCTTGTTTCAGGAGGTATATTGTGACCGGAAAAGCTAAACTAATGATGAAGATTAATGCGGAGGTCTGATGAGAATAAACCTTATTATTCAGATTAGGCCCCAAGAGGCATTCTTCATCTCCTTTTAGCAAAATACTATTTCAGGATAGTCCAGCTAGTGACACGTCTTTTAGCTGTATACCAGTTGCCCCTGAGATACGCCACAAAAGTGTCTCTGAGCTAAAGTGGTCTGTACACATCTCATACATTGTATTAGGGGCAATAATATCTAATTGAACTTAGCCATTTAAAATTTAGTGCATAAATCTGGGCTAACTCCACCAGGTCAACTCCATTGGCTGAAAAGAAGCCCACCTACAACGAACATTACTTTGAGCGCCCTCACAATTAAAAAATAAGAGCGTCGTTCCAACAATCGAGCGCAAGGTTACAAGGTTGAACTGAGAGTGTCTAGACAACAAAATATCGATACTCCAGACACCAAGCAAGACCTGAGAAAAAACCATGGCCAAAGCTACGGGACGATACAATCTAATATCGCCCAAAAAGGACCTGGAGAAAGGGGTTGTCTTAAGCGACCTCTGTAACTTCTTAGTTAGTCAAACTATTCAAGGGTGGAAAGTTTATTGGGCTGGTATTGAGTTTGATGTGACTCACAAAGGAATGGCCCTATTGCATAGACTGAAAACTAATGACTTTGCCCCTGCATGGTCAATGACAAGGAACCTATTTCCCCATTTATTTCAAAATCCGAATTCCACTATTGAATCACCGCTGTGGGCACTGAGAGTCATCCTTGCAGCAGGGATACAGGACCAGTTAATTGACCAGTCTTTGATTGAACCCTTAGCAGGAGCCCTTGGTCTGATCTCTGATTGGCTGCTAACAACCAACACTAACCATTTCAACATGCGAACACAACGTGTCAAGGAACAATTGAGCCTAAAAATGCTGTCGTTGATTCGATCCAATATTCTCAAGTTTATTAACAAATTGGATGCTCTACATGTCGTGAACTACAATGGATTATTGAGCAGTATTGAAATTGGAACTCAAAATCATACAATCATCATAACTCGAACTAACATGGGTTTTCTGGTGGAGCTCCAAGAACCCGACAAATCGGCAATGAACCGCAAGAAGCCTGGGCCGGCGAAATTTTCCCTCCTTCATGAGTCCACACTGAAAGCATTTACACAAGGGTCCTCGACACGAATGCAAAGTTTAATTCTTGAATTCAATAGCTCTCTTGCTATCTAACTAAGATGGAATACTTCATATTGGGCTAACTCATATATGCTGACTCAATAGTTAACTTGACATCTCTGCCTTCATAATCAGATATATAAGCATAATAAATAAATACTCATATTTCTTGATAATTTGTTTAACCACAGATAAATCCTCACTGTAAGCCAGCTTCCAAGTTGACACCCTTACAAAAACCAGGACTCAGAATCCCTCAAATAAGAGATTCCAAGACAACATCATAGAATTGCTTTATTATATTAATAAGCATTTTATCACTAGAAATCCAATATACGAAATGGTTAATTGTAACTAAACCCGCAGGTCATGTGTGTTAGGTTTCACAAATTATATATATTACTAACTCCATACTCGTAACTAACATTAGATAAGTAGGTTAAGAAAAAAGCTTGAGGAAGATTAAGAAAAACTGCTTATTGGGTCTTTCCGTGTTTTAGATGAAGCAGTTGACATTCTTCCTCTTGATATTAAATGGCTACACAACATACCCAATACCCAGACGCCAGGTTATCATCACCAATTGTATTGGACCAATGTGACCTTGTCACTAGAGCTTGCGGGTTGTATTCATCATACTCCCTTAATCCGCAACTACGCAACTGTAAACTCCCGAAACATATATACCGTTTAAAATATGATGTAACTGTTACCAAGTTCTTAAGTGATGTACCAGTGGCGACATTGCCCATAGATTTCATAGTCCCAATTCTTCTCAAGGCACTATCAGGCAATGGGTTCTGTCCTGTTGAGCCGCGGTGCCAACAGTTCTTAGATGAAATTATTAAGTACACAATGCAAGATGCTCTCTTCCTGAAATATTATCTCAAAAATGTGGGTGCTCAAGAAGACTGTGTTGATGACCACTTTCAAGAAAAAATCTTATCTTCAATTCAGGGCAATGAATTTTTACATCAAATGTTTTTCTGGTATGACCTGGCTATTTTAACTCGAAGGGGTAGATTAAATCGAGGAAACTCTAGATCAACGTGGTTTGTTCATGATGATTTAATAGACATCTTAGGCTATGGGGACTATGTTTTTTGGAAGATCCCAATTTCACTGTTACCACTGAACACACAAGGAATCCCCCATGCTGCTATGGATTGGTATCAGACATCAGTATTCAAAGAAGCGGTTCAAGGGCATACACACATTGTTTCTGTTTCTACTGCCGATGTCTTGATAATGTGCAAAGATTTAATTACATGTCGATTCAACACAACTCTAATCTCAAAAATAGCAGAGGTTGAGGACCCAGTTTGCTCTGATTATCCCAATTTTAAGATTGTGTCTATGCTTTACCAGAGCGGAGATTACTTACTCTCCATATTAGGGTCTGATGGGTATAAAATCATTAAGTTTCTCGAACCATTGTGCTTGGCTAAAATTCAATTGTGCTCAAAGTACACCGAGAGGAAGGGCCGATTCTTAACACAAATGCATTTAGCTGTAAATCACACCCTGGAAGAAATTACAGAAATACGTGCACTAAAGCCTTCACAGGCTCACAAGATCCGTGAATTCCATAGAACATTGATAAGGCTGGAGATGACGCCACAACAACTTTGTGAGCTATTTTCCATACAAAAACACTGGGGGCATCCTGTGCTACATAGTGAAACAGCAATCCAAAAAGTTAAAAAACATGCTACGGTGCTAAAAGCATTACGCCCTATCGTGATTTTCGAGACATATTGTGTTTTTAAATATAGCATTGCAAAACATTATTTTGATAGTCAAGGATCTTGGTACAGTGTTACCTCAGATAGAAATCTAACACCAGGTCTTAATTCTTATATCAAAAGAAATCAATTCCCTCCGTTGCCAATGATTAAAGAACTGCTATGGGAATTTTACCACCTTGACCATCCTCCACTTTTCTCAACCAAAATTATTAGTGACTTAAGTATTTTTATAAAAGACAGAGCTACTGCAGTAGAAAGGACATGCTGGGATGCAGTATTCGAGCCTAATGTTCTGGGATATAATCCACCTCACAAATTCAGTACCAAACGTGTACCGGAACAATTTTTAGAGCAAGAAAACTTTTCTATTGAGAATGTTCTTTCCTACGCGCAAAAACTCGAGTATCTACTACCACAATATCGGAATTTTTCTTTCTCATTGAAAGAGAAAGAGTTGAATGTAGGTAGAACTTTCGGAAAATTGCCTTATCCGACTCGCAATGTTCAAACACTTTGTGAAGCTCTGTTAGCTGATGGTCTTGCTAAAGCATTTCCTAGCAATATGATGGTAGTTACGGAACGTGAACAAAAAGAAAGCTTATTGCATCAAGCATCATGGCACCACACAAGTGATGATTTCGGTGAGCATGCCACAGTTAGAGGGAGTAGCTTTGTAACTGATTTAGAGAAATACAATCTTGCATTTAGGTATGAGTTTACAGCACCTTTTATAGAATATTGCAACCGTTGCTATGGTGTTAAGAATGTTTTTAATTGGATGCATTATACAATCCCACAGTGTTATATGCATGTCAGTGATTATTATAATCCACCGCATAACCTCACACTGGAAAATCGAAACAACCCCCCTGAAGGGCCTAGTTCATACAGGGGTCATATGGGAGGGATTGAAGGACTGCAACAAAAACTCTGGACAAGTATTTCATGTGCTCAAATTTCTTTAGTTGAAATTAAGACTGGTTTTAAGTTGCGCTCAGCTGTGATGGGTGACAATCAGTGCATTACCGTTTTATCAGTCTTCCCCTTAGAGACTGATGCAGACGAGCAGGAACAGAGCGCCGAGGACAATGCAGCGAGGGTGGCCGCCAGCCTAGCAAAAGTTACAAGTGCCTGTGGAATCTTTTTAAAACCTGATGAAACATTTGTACATTCAGGTTTTATCTATTTTGGAAAAAAACAATATTTGAATGGGGTCCAATTGCCTCAGTCCCTTAAAACGGCTACAAGAATGGCACCATTGTCTGATGCAATTTTTGATGATCTTCAAGGGACCCTGGCTAGTATAGGTACTGCTTTTGAGCGATCCATCTCTGAGACACGACATATCTTTCCTTGCAGAATAACCGCAGCTTTCCATACGTTCTTTTCGGTGAGAATCTTGCAATATCATCACCTCGGATTTAATAAAGGTTTTGACCTTGGACAGTTAACACTCGGCAAACCTCTGGATTTCGGAACAATATCATTGGCACTAGCGGTACCGCAGGTGCTTGGAGGGTTATCCTTCTTGAATCCTGAGAAATGTTTCTACCGGAATCTAGGAGATCCAGTTACCTCAGGTTTATTCCAGTTAAAAACTTATCTCCGAATGATTGAGATGGATGATTTATTCTTACCTTTAATTGCGAAGAACCCTGGGAACTGCACTGCCATTGACTTTGTGCTAAATCCTAGCGGATTAAATGTTCCTGGGTCGCAAGACTTAACTTCATTTCTGCGCCAGATTGTACGTAGGACTATCACCCTAAGTGCGAAAAACAAACTTATTAATACCTTATTTCATGCATCAGCTGACTTCGAAGACGAAATGGTTTGTAAGTGGCTCTTATCATCAACTCCTGTTATGAGTCGTTTCGCAGCCGATATATTTTCACGCACGCCGAGCGGGAAGCGATTGCAAATTCTAGGATACTTGGAAGGAACACGCACATTATTAGCCTCTAAGATCATCAACAATAATACAGAGACGCCGGTTTTGGACAGACTGAGGAAGATAACATTGCAAAGGTGGAGTCTATGGTTTAGTTATCTTGATCATTGTGATAATATCCTGGCGGAGGCTTTAACCCAAATAACTTGCACAGTTGATTTAGCACAGATCCTGAGGGAATATTCATGGGCACATATTTTAGAGGGGAGACCTCTTATTGGAGCCACACTCCCATGTATGATTGAGCAATTCAAAGTGGTTTGGCTGAAACCCTACGAACAATGTCCGCAGTGTTCAAATGCCAAGCAACCTGGTGGGAAACCATTCGTGTCAGTAGCAGTCAAGAAACATATTGTTAGTGCATGGCCAAATGCATCCCGAATAAGCTGGACTATCGGGGATGGAATCCCATACATTGGATCAAGGACAGAAGATAAGATAGGGCAACCTGCTATTAAACCAAAATGTCCTTCCGCAGCCTTAAGAGAGGCCATTGAATTGGCGTCCCGTTTAACATGGGTAACTCAAGGCAGTTCGAACAGTGACTTGCTAATAAAACCATTTTTGGAAGCACGAGTAAATTTAAGTGTTCAAGAAATACTTCAAATGACCCCTTCACATTACTCGGGAAATATTGTTCATAGGTACAACGATCAATACAGTCCTCATTCTTTCATGGCCAATCGTATGAGTAACTCAGCAACGCGATTGATTGTTTCTACAAACACTTTAGGTGAGTTTTCAGGAGGTGGCCAATCGGCACGCGACAGCAATATTATTTTCCAGAATGTTATAAATTATGCAGTTGCACTGTTCGATATTAAATTTAGAAACACTGAGGCTACAGATATCCAGTATAATCGTGCTCACCTTCATCTAACTAAGTGTTGCACCCGGGAGGTACCAGCTCAGTATTTAACATACACATCTACATTGGATTTAGATTTAACAAGATACCGAGAAAATGAATTGATTTATGACAATAATCCTCTAAAAGGAGGACTCAATTGCAATATCTCATTTGATAACCCATTTTTCCAAGGCAAACAGCTGAACATTATAGAAGATGACCTTATTCGACTGCCTCACTTATCTGGATGGGAGCTAGCTAAGACCATCATGCAATCAATTATTTCAGATAGCAATAATTCGTCTACAGACCCAATTAGCAGTGGAGAAACAAGATCATTCACTACCCATTTCTTAACTTATCCCAAGATAGGACTTCTGTACAGTTTTGGGGCCTTTGTAAGTTATTATCTTGGCAATACAATTCTTCGGACTAAGAAATTAACACTTGACAATTTTTTATATTACTTAACTACCCAAATTCATAATCTACCACATCGCTCATTGCGAATACTTAAGCCAACATTCAAACATGCAAGCGTTATGTCACGATTAATGAGTATTGATCCCCATTTTTCTATTTACATAGGCGGTGCTGCAGGTGACAGAGGACTCTCAGATGCGGCCAGGTTATTTTTGAGAACGTCCATTTCATCTTTTCTTACATTTGTAAAGGAATGGATAATTAATCGCGGAACAATTGTCCCTTTATGGATAGTATATCCATTAGAGGGTCAAAATCCAACACCTGTTAATAATTTCCTCCATCAGATCGTAGAACTGCTGGTGCATGATTCATCAAGACACCAGGCTTTTAAAACTACCATAAATGATCATGTACATCCTCACGACAATCTTGTTTACACATGTAAGAGTACAGCCAGCAATTTCTTCCATGCGTCATTGGCGTACTGGAGGAGCAGGCACAGAAACAGCAACCGAAAAGACTTGACAAGAAACTCTTCAACTGGATCAAGCACAAACAACAGTGATGGTCATATTAAGAGAAGTCAAGAACAAACCACCAGAGATCCACATGATGGCACTGAACGGAGTCTAGTCCTGCAAATGAGCCATGAAATAAAAAGAACGACAATTCCACAAGAGAACACGCACCAGGGTCCGTCGTTCCAGTCATTTCTAAGTGACTCTGCTTGCGGTACAGCAAACCCAAAACTAAATTTCGATAGATCGAGACACAATGTGAAATCTCAGGATCATAACTCAGCATCCAAGAGGGAAGGTCATCAAATAATCTCACATCGTCTAGTCCTACCTTTCTTTACATTATCTCAAGGGACACGCCAATTAACGTCATCCAATGAGTCACAAACCCAAGATGAGATATCAAAGTACTTACGGCAATTGAGATCCGTCATTGATACCACAGTTTATTGTAGGTTTACCGGTATAGTCTCGTCCATGCATTACAAACTTGATGAGGTCCTTTGGGAAATAGAGAATTTTAAGTCGGCTGTGACGCTGGCAGAGGGAGAAGGTGCTGGTGCCTTACTATTGATTCAGAAATACCAAGTTAAGACCTTATTTTTCAACACGCTAGCTACTGAGTCCAGTATAGAGTCAGAAATAGTATCAGGAATGACTACTCCTAGGATGCTTCTACCTGTTATGTCAAAATTCCATAATGACCAAATTGAGATTATTCTTAACAACTCAGCAAGCCAAATAACAGACATAACAAATCCTACTTGGTTTAAAGACCAAAGAGCAAGGCTACCTAGGCAAGTCGAGGTTATAACCATGGATGCAGAGACGACAGAGAATATAAACAGATCGAAATTGTACGAAGCTGTACATAAATTGATCTTACACCATGTTGATCCCAGCGTATTGAAAGCAGTGGTCCTTAAAGTCTTTCTAAGTGATACCGAGGGTATGTTATGGCTAAATGATAATCTAGCCCCGTTTTTTGCCACTGGGTATTTAATTAAGCCAATAACGTCAAGTGCCAGGTCTAGTGAGTGGTATCTTTGTCTGACGAACTTCTTATCAACTACACGTAAGATGCCACACCAAAACCATCTCAGTTGTAAGCAGGTAATACTTACGGCATTGCAACTGCAAATTCAACGGAGCCCATACTGGCTAAGTCATTTAACTCAGTATGCTGACTGCGATTTACATTTAAGCTATATCCGCCTTGGTTTTCCATCATTAGAGAAAGTACTATACCACAGGTATAACCTTGTCGATTCAAAAAGAGGTCCACTAGTCTCTGTCACTCAGCACTTAGCACATCTTAGGGCAGAGATTCGAGAATTGACCAATGATTATAATCAACAGCGACAAAGTCGGACTCAAACATATCACTTTATTCGTACTGCAAAAGGACGAATCACAAAACTAGTCAATGATTATTTAAAATTCTTTCTTATTGTACAAGCATTAAAACATAATGGGACATGGCAAGCTGAGTTTAAGAAATTACCAGAGTTGATTAGTGTGTGCAATAGGTTCTATCATATTAGAGATTGTAATTGTGAAGAACGTTTCTTAGTTCAAACCTTATATTTACATAGAATGCAGGATTCTGAAGTTAAGCTTATCGAAAGGCTGACAGGGCTTCTGAGTTTATTTCCAGATGGTCTCTACAGGTTCGATTGAATAACCGTGCATAGTATTTTGATACTTGTAAAGGTTGGTTATCAACATACAGATTATAAAAAACTCATAAATTGCTCTCATACATCATCTTGATCTGATTTCAATAAATAACTATTTAGATAACGAAAGGAGTCCTTACATTATACACTATATTTGGCCTCTCTCCCTGCGTGATAATCAAAAAATTCACAATACAGCATGTGTGACATATTACTGCTGCAATGAGTCTAACGCAACATAATAAACTCCGCACTCTTTATAATTAAGCTTTAACGATAGGTCTGGGCTCATATTGTTATTGATATAGTAATGTTGTATCAATATCTTGCCAGATGGAATAGTGCTTTGGTTGATAACACGACTTCTTAAAACAAAACTGATCTTTAAGATTAAGTTTTTTATAATTGTCATTGCTTTGATTTGTCGATTTAAAAATGGTGATAGCCTTAATCTTTGTGTAAAATAAGAGATTAGGTGTAATAACTTTAACATTTTTGTCTAGTAAGCTACTATTCCATTCAGAATGATAAAATTAAAAGAAAAGACATGACTGTAAAATCAGAAATACCTTCTTTACAATATAGCAGACTAGATAATAATCTTCGTGTTAATGATAATTAAGGCATTGACCACGCTCATCAGAAGGCTCACTAGAATAAACGTTGCAAAAAGGATCCCTGGAAAAATGGTCGCACACAAAAATTTAAAAATAAATCTATTTCTTCTTTTTTGTGTGCCCC',
		}

class ebola_clean(virus_clean):
	def __init__(self,**kwargs):
		virus_clean.__init__(self, **kwargs)

	def clean(self):
		self.unique_date()	
		self.remove_insertions()	
		self.clean_ambiguous()			
		print "Number of viruses after cleaning:",len(self.viruses)

class ebola_refine(tree_refine):
	def refine(self):
		self.refine_generic()

class ebola_process(process, ebola_filter, ebola_clean, ebola_refine):
	"""docstring for ebola, """
	def __init__(self,verbose = 0, force_include = None, 
				force_include_all = False, max_global= True, **kwargs):
		self.force_include = force_include
		self.force_include_all = force_include_all
		self.max_global = max_global
		process.__init__(self, **kwargs)
		ebola_filter.__init__(self,**kwargs)
		ebola_clean.__init__(self,**kwargs)
		ebola_refine.__init__(self,**kwargs)
		self.verbose = verbose

	def run(self, steps, viruses_per_month=50, raxml_time_limit = 1.0):
		if 'filter' in steps:
			print "--- Virus filtering at " + time.strftime("%H:%M:%S") + " ---"
			self.filter()
			if self.force_include is not None and os.path.isfile(self.force_include):
				with open(self.force_include) as infile:
					forced_strains = [line.strip().lower() for line in infile]
			else:
				forced_strains = []
			self.subsample(viruses_per_month, 
				prioritize=forced_strains, all_priority=self.force_include_all, 
				region_specific = self.max_global)
			self.dump()
		else:
			self.load()
		if 'align' in steps:
			self.align()   	# -> self.viruses is an alignment object
			self.dump()			
		if 'clean' in steps:
			print "--- Clean at " + time.strftime("%H:%M:%S") + " ---"
			self.clean()   # -> every node as a numerical date
			self.dump()
		if 'tree' in steps:
			print "--- Tree	 infer at " + time.strftime("%H:%M:%S") + " ---"
			self.infer_tree(raxml_time_limit)  # -> self has a tree
			self.dump()
		if 'ancestral' in steps:
			print "--- Infer ancestral sequences " + time.strftime("%H:%M:%S") + " ---"
			self.infer_ancestral()  # -> every node has a sequence
		if 'refine' in steps:
			print "--- Tree refine at " + time.strftime("%H:%M:%S") + " ---"
			self.refine()
			self.dump()
		if 'frequencies' in steps:
			print "--- Estimating frequencies at " + time.strftime("%H:%M:%S") + " ---"
			self.determine_variable_positions()
			self.estimate_frequencies(tasks = ["nuc_mutations", "clades", "tree"])
			if 'genotype_frequencies' in steps: 
					self.estimate_frequencies(tasks = ["genotypes"])
			self.dump()
		if 'export' in steps:
			self.temporal_regional_statistics()
			# exporting to json, including the H1N1pdm specific fields
			self.export_to_auspice(tree_fields = ['nuc_muts','accession','isolate_id', 'lab','db', 'country'] 
													+ self.fasta_fields.values(), 
			                       annotations = [], export_entropy = 'nuc')
			self.generate_indexHTML()

if __name__=="__main__":
	all_steps = ['filter', 'align', 'clean', 'tree', 'ancestral', 'refine', 'frequencies', 'export']
	from process import parser, shift_cds
	params = parser.parse_args()

	lt = time.localtime()
	num_date = round(lt.tm_year+(lt.tm_yday-1.0)/365.0,2)
	params.time_interval = (num_date-params.years_back, num_date) 
	if params.interval is not None and len(params.interval)==2 and params.interval[0]<params.interval[1]:
		params.time_interval = (params.interval[0], params.interval[1])
	dt= params.time_interval[1]-params.time_interval[0]
	params.pivots_per_year = 12.0 if dt<5 else 6.0 if dt<10 else 3.0
	steps = all_steps[all_steps.index(params.start):(all_steps.index(params.stop)+1)]
	if params.skip is not None:
		for tmp_step in params.skip:
			if tmp_step in steps:
				print "skipping",tmp_step
				steps.remove(tmp_step)

	# add all arguments to virus_config (possibly overriding)
	virus_config.update(params.__dict__)
	# pass all these arguments to the processor: will be passed down as kwargs through all classes
	ebola = ebola_process(**virus_config) 
	if params.test:
		ebola.load()
	else:
		ebola.run(steps,viruses_per_month = virus_config['viruses_per_month'], 
			raxml_time_limit = virus_config['raxml_time_limit'])
