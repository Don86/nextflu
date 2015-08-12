import dendropy, time
import numpy as np
from itertools import izip
from seq_util import translate
from io_util import read_json
from io_util import write_json
from tree_util import json_to_dendropy
from tree_util import dendropy_to_json
from fitness_tolerance import load_mutational_tolerance, calc_fitness_tolerance

epitope_mask = np.fromstring("00000000000000000000000000000000000000000000000000000000000011111011011001010011000100000001001011110011100110101000001100000100000001000110101011111101011010111110001010011111000101011011111111010010001111101110111001010001110011111111000000111110000000101010101110000000000011100100000001011011100000000000001001011000110111111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", dtype='S1')
# shift left by 1 to improve fit. This should probably become baseline across the board.
epitope_mask = np.fromstring("00000000000000000000000000000000000000000000000000000000000111110110110010100110001000000010010111100111001101010000011000001000000010001101010111111010110101111100010100111110001010110111111110100100011111011101110010100011100111111110000001111100000001010101011100000000000111001000000010110111000000000000010010110001101111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", dtype='S1')


def epitope_sites(aa):
	aaa = np.fromstring(aa, 'S1')
	return ''.join(aaa[epitope_mask[:len(aa)]=='1'])

def nonepitope_sites(aa):
	aaa = np.fromstring(aa, 'S1')
	return ''.join(aaa[epitope_mask[:len(aa)]=='0'])
	
def epitope_distance(aaA, aaB):
	"""Return distance of sequences aaA and aaB by comparing epitope sites"""
	epA = epitope_sites(aaA)
	epB = epitope_sites(aaB)
	distance = sum(a != b for a, b in izip(epA, epB))
	return distance

def nonepitope_distance(aaA, aaB):
	"""Return distance of sequences aaA and aaB by comparing non-epitope sites"""
	neA = nonepitope_sites(aaA)
	neB = nonepitope_sites(aaB)
	distance = sum(a != b for a, b in izip(neA, neB))
	return distance

def calc_epitope_distance(tree, attr='ep', ref = None):
	'''
	calculates the distance at epitope sites of any tree node  to ref
	tree   --   dendropy tree
	attr   --   the attribute name used to save the result
	'''
	if not hasattr(tree, "epitope_distance_assigned") or tree.epitope_distance_assigned==False:
		if ref == None:
			ref = translate(tree.seed_node.seq)
		for node in tree.postorder_node_iter():
			if not hasattr(node, 'aa'):
				node.aa = translate(node.seq)
			node.__setattr__(attr, epitope_distance(node.aa, ref))
		tree.epitope_distance_assigned=True

def calc_tolerance(tree, attr='tol'):
	'''
	calculates the distance at epitope sites of any tree node  to ref
	tree   --   dendropy tree
	attr   --   the attribute name used to save the result
	'''
	from Bio import AlignIO
	aa, sites, wt_aa, aa_prob = load_mutational_tolerance()
	aln = AlignIO.read('source-data/H1_H3.fasta', 'fasta')
	# returns true whenever either of the sequences have a gap
	aligned = (np.array(aln)!='-').min(axis=0)
	# map alignment positions to sequence positions, subset to aligned amino acids
	indices = {}
	for seq in aln:
		indices[seq.name] = (np.cumsum(np.fromstring(str(seq.seq), dtype='S1')!='-')-1)[aligned]

	# make a reduced set of amino-acid probabilities that only contains aligned positions
	aa_prob=aa_prob[indices['H1'],:]
	# attach another column for non-canonical amino acids

	aa_prob = np.hstack((aa_prob, 1e-5*np.ones((aa_prob.shape[0],1))))	
	if not hasattr(tree, "tolerance_assigned") or tree.tolerance_assigned==False:
		for node in tree.postorder_node_iter():
			if not hasattr(node, 'aa'):
				node.aa = translate(node.seq)
			node.__setattr__(attr, calc_fitness_tolerance(node.aa, aa_prob, aa, indices['H3']))
		tree.tolerance_assigned=True


def calc_nonepitope_distance(tree, attr='ne', ref = None):
	'''
	calculates the distance at nonepitope sites of any tree node to ref
	tree   --   dendropy tree
	attr   --   the attribute name used to save the result
	'''
	if not hasattr(tree, "nonepitope_distance_assigned") or tree.nonepitope_distance_assigned==False:
		if ref == None:
			ref = translate(tree.seed_node.seq)
		for node in tree.postorder_node_iter():
			if not hasattr(node, 'aa'):
				node.aa = translate(node.seq)
			node.__setattr__(attr, nonepitope_distance(node.aa, ref))
		tree.nonepitope_distance_assigned=True

def calc_nonepitope_star_distance(tree, attr='ne_star', seasons = []):
	'''
	calculates the distance at nonepitope sites of any tree node to ref
	tree   --   dendropy tree
	attr   --   the attribute name used to save the result
	'''
	if not hasattr(tree, "nonepitope_star_distance_assigned") or tree.nonepitope_star_distance_assigned==False:
		for node in tree.postorder_node_iter():
			if len(node.tips) and node!=tree.seed_node:
				if not hasattr(node, 'aa'):
					node.aa = translate(node.seq)
				tmp_node = node.parent_node
				cur_season = min(node.tips.keys())
				prev_season = seasons[max(0,seasons.index(cur_season)-1)]
				while True:
					if tmp_node!=tree.seed_node:
						if prev_season in tmp_node.tips and len(tmp_node.tips[prev_season])>0:
							break
						else:
							tmp_node=tmp_node.parent_node
					else:
						break
				if not hasattr(tmp_node, 'aa'):
					tmp_node.aa = translate(tmp_node.seq)
				node.__setattr__(attr, nonepitope_distance(node.aa, tmp_node.aa))
			else:
				node.__setattr__(attr, np.nan)				
		tree.nonepitope_star_distance_assigned=True


def calc_LBI(tree, attr = 'lbi', tau=0.0005, transform = lambda x:x):
	'''
	traverses the tree in postorder and preorder to calculate the
	up and downstream tree length exponentially weighted by distance.
	then adds them as LBI
	tree -- dendropy tree for whose node the LBI is being computed
	attr	 -- the attribute name used to store the result
	'''
	# traverse the tree in postorder (children first) to calculate msg to parents
	for node in tree.postorder_node_iter():
		node.down_polarizer = 0
		node.up_polarizer = 0
		for child in node.child_nodes():
			node.up_polarizer += child.up_polarizer
		bl =  node.edge_length/tau
		node.up_polarizer *= np.exp(-bl)
		if node.alive: node.up_polarizer += tau*(1-np.exp(-bl))

	# traverse the tree in preorder (parents first) to calculate msg to children
	for node in tree.preorder_internal_node_iter():
		for child1 in node.child_nodes():
			child1.down_polarizer = node.down_polarizer
			for child2 in node.child_nodes():
				if child1!=child2:
					child1.down_polarizer += child2.up_polarizer

			bl =  child1.edge_length/tau
			child1.down_polarizer *= np.exp(-bl)
			if child1.alive: child1.down_polarizer += tau*(1-np.exp(-bl))

	# go over all nodes and calculate the LBI (can be done in any order)
	for node in tree.postorder_node_iter():
		tmp_LBI = node.down_polarizer
		for child in node.child_nodes():
			tmp_LBI += child.up_polarizer
		node.__setattr__(attr, transform(tmp_LBI))

def geo_fitness(region):
	'''
	Assign fitness based on historical geographic spread
	Numbers are from https://github.com/blab/global-migration/blob/master/geo/h3_large_geo/stats/h3_large_geo_origin.tsv
	'''
	fitness = 0.0
	if region == "NorthAmerica":
		fitness = 0.116
	elif region == "SouthAmerica":
		fitness = 0.041
	elif region == "Europe":
		fitness = 0.072
	elif region == "India":
		fitness = 0.170
	elif region == "China":
		fitness = 0.413
	elif region == "JapanKorea":
		fitness = 0.038
	elif region == "SoutheastAsia":
		fitness = 0.133
	elif region == "Oceania":
		fitness = 0.016
	elif region == "WestAsia":
		fitness = 0.0
	elif region == "Africa":
		fitness = 0.0
	return fitness																

def calc_geo_fitness(tree, attr='geo'):
	'''
	calculates the "geographic fitness" of any tree node
	tree   --   dendropy tree
	attr   --   the attribute name used to save the result
	'''
	if not hasattr(tree, "geo_fitness_assigned") or tree.geo_fitness_assigned==False:
		for node in tree.postorder_node_iter():
			if hasattr(node, 'region'):
				node.__setattr__(attr, geo_fitness(node.region))
		tree.geo_fitness_assigned=True

def main(tree_fname = 'data/tree_refine.json'):

	print "--- Testing predictor evaluations ---"
	tree =  json_to_dendropy(read_json(tree_fname))

	print "Calculating epitope distances"
	calc_epitope_distance(tree)

	print "Calculating nonepitope distances"
	calc_nonepitope_distance(tree)

	print "Calculating LBI"
#	calc_LBI(tree)

	print "Calculating geo fitnesses"
	calc_geo_fitness(tree)

	print "Writing decorated tree"
	out_fname = "data/tree_predictors.json"
	write_json(dendropy_to_json(tree.seed_node), out_fname)
	return out_fname

if __name__=='__main__':
	main()
