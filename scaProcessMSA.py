#!/usr/bin/env python
"""
The scaProcessMSA script conducts the basic steps in multiple sequence alignment (MSA) pre-processing for SCA, and stores the results using the python tool pickle:  

     1)  Trim the alignment, either by truncating to a reference sequence (specified with the -t flag) or by removing
         excessively gapped positions (set to positions with more than 40% gaps)
     2)  Identify/designate a reference sequence in the alignment, and create a mapping of the alignment numberings to position numberings 
         for the reference sequence. The reference sequence can be specified in one of four ways:
              a)  By supplying a PDB file - in this case, the reference sequence is taken from the PDB (see the pdb kwarg)
              b)  By supplying a reference sequence directly (as a fasta file - see the refseq kwarg)
              c)  By supplying the index of the reference sequence in the alignment (see the refseq kwarg)
              d)  If no reference sequence is supplied by the user, one is automatically selected using the scaTools function chooseRef.
         The position numbers (for mapping the alignment) can be specified in one of three ways:
              a) By supplying a PDB file - in this case the alignment positions are mapped to structure positions 
              b) By supplying a list of reference positions (see the refpos kwarg)
              c) If no reference positions are supplied by the user, sequential numbering (starting at 1) is assumed.
     3)  Filter sequences to remove highly gapped sequences, and sequences with an identity below or above some minimum 
         or maximum value to the reference sequence (see the parameters kwarg)
     4)  Filter positions to remove highly gapped positions (default 20% gaps, can also be set using --parameters)
     5)  Calculate sequence weights and write out the final alignment and other variables
              

:Arguments: 
     Input_MSA.fasta (the alignment to be processed, typically the headers contain taxonomic information for the sequences).

:Keyword Arguments:
     --pdb, -s         PDB identifier (ex: 1RX2) // keep
     --chainID, -c     chain ID in the PDB for the reference sequence // keep
     --species, -f     species of the reference sequence // drop
     --refseq, -r      reference sequence, supplied as a fasta file // drop
     --refpos, -o      reference positions, supplied as a text file with one position specified per line // drop
     --refindex, -i    reference sequence number in the alignment, COUNTING FROM 0 // drop
     --parameters, -p  list of parameters for filtering the alignment:  // drop
                       [max_frac_gaps for positions, max_frac_gaps for sequences, min SID to reference seq, max SID to reference seq]
                       default values: [0.2, 0.2, 0.2, 0.8] (see filterPos and filterSeq functions for details)
     --selectSeqs, -n  subsample the alignment to (1.5 * the number of effective sequences) to reduce computational time, default: False // drop
     --truncate, -t    truncate the alignment to the positions in the reference PDB, default: False // keep
     --matlab, -m      write out the results of this script to a matlab workspace for further analysis  // drop
     --output          specify a name for the outputfile  // keep

:Example: 
>>> ./scaProcessMSA.py Inputs/PF00071_full.an -s 5P21 -c A -f 'Homo sapiens' 

:By: Rama Ranganathan
:On: 8.5.2014
Copyright (C) 2015 Olivier Rivoire, Rama Ranganathan, Kimberly Reynolds
This program is free software distributed under the BSD 3-clause
license, please see the file LICENSE for details.
"""
from __future__ import division
import time

import os
import pickle
import argparse
import re

import numpy as np
from Bio.pairwise2 import align
from Bio.PDB import PDBParser

import scaTools as sca


FINAL_MSG = """Final alignment parameters:
    Number of sequences, M: {}
    Number of effective sequences, M': {}
    Number of alignment positions, L: {}
    Number of positions in the ats: {}
    Number of structure positions mapped: {}
    Size of the distance matrix: {} x {}
"""
# Gap limiting parameters
PARAMETERS = [0.2, 0.2, 0.2, 0.8]
# Default standard amino acids
DEFAULT_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
# Table of 3-letter to 1-letter code for amino acids
AMINO_ACID_TABLE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}


def get_pdb_residues(pdb_id, chain_id, input_dir='Inputs/', QUIET=False):
    """
    Read the PDB structure using Bio.PDB.PDBParser
    :param pdb_id:
    :param input_dir:
    :return:
    """
    model_index = 0
    hetero_atom_index = 0
    file_name = os.path.join(input_dir, pdb_id + '.pdb')
    structure = PDBParser(QUIET=True).get_structure(pdb_id, file_name)
    return [res for res in structure[model_index][chain_id] if res.get_id()[hetero_atom_index] == ' ']


def get_pdb_sequence(residues, substitute_unknown='X'):
    """
    Get the sequence for the pdb residues, and the labels
    :param residues:
    :param substitute_unknown:
    :return:
    """
    sequence, labels = list(), list()
    for residue in residues:
        residue_id = residue.get_id()
        sequence_id = residue_id[1]
        insertion_code = residue_id[2]
        labels.append(str(sequence_id) + str(insertion_code).strip())
        sequence.append(AMINO_ACID_TABLE.get(residue.get_resname(), substitute_unknown))
    return str.join('', sequence), labels


def get_pdb_distances(residues):
    """
    Calculate the distances between residues
    :param residues:
    :return:
    """
    # Distances between residues (minimal distance between atoms, in angstrom):
    dist = np.zeros((len(residues), len(residues)))
    for n0, res0 in enumerate(residues):
        for n1, res1 in enumerate(residues):
            dist[n0, n1] = min([atom0 - atom1 for atom0 in res0 for atom1 in res1])
    return dist


def locate_reference(sequences, pdb_sequence):
    """
    Get the index of the reference sequence in sequences using a global pairwise alignment
    :param sequences:
    :param pdb_sequence:
    :return:
    """
    score = list()
    for index, sequence in enumerate(sequences):
        score.append(align.globalxx(pdb_sequence, sequence, one_alignment_only=1, score_only=1))
    return score.index(max(score))


def get_options():
    """
    Use argparse to parse the args.
    :return: output from ArgumentParser.parse_args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment", help='Input Sequence Alignment')
    parser.add_argument("-s", "--pdb", dest="pdbid", help="PDB identifier (ex: 1RX2)")
    parser.add_argument("-c", "--chainID", dest="chainID", default='A',
                        help="chain ID in the PDB for the reference sequence")
    parser.add_argument("-t", "--truncate", action="store_true", dest="truncate", default=False,
                        help="truncate the alignment to the positions in the reference PDB, default: False")
    parser.add_argument("--output", dest="outputfile", default=None, help="specify an outputfile name")
    return parser.parse_args()


def filter_non_standard(alignment, headers, amino_acids):
    """
    Loop through the alignment and check for non-standard amino acids in the sequences based on the parameter
    :param alignment: (List[String]) The multiple sequence alignment sequences as a list
    :param headers: (List[String]) The multiple sequence alignment headers as a list
    :param amino_acids: (String) The amino acid codes to consider standard as a single string
    :return: (Tuple(List[String], List[String]) the sequences and headers that made the cut
    """
    non_standard = re.compile(f'[^{amino_acids}]')
    align_clean, headers_clean = list(), list()
    for i, sequence in enumerate(alignment):
        if non_standard.search(sequence) is not None:
            continue
        else:
            align_clean.append(sequence)
            headers_clean.append(headers[i])
    return align_clean, headers_clean


if __name__ == '__main__':
    _start = time.time()
    options = get_options()

    headers_full, sequences_full = sca.readAlg(options.alignment)
    print("Loaded alignment of {} sequences, {} positions.".format(len(headers_full), len(sequences_full[0])))

    print("Checking alignment for non-standard amino acids")
    standard_sequences, standard_headers = filter_non_standard(sequences_full, headers_full, DEFAULT_AMINO_ACIDS)

    print("Alignment size after removing sequences with non-standard amino acids: {}".format(len(standard_sequences)))
    # Do an initial trimming to remove excessively gapped positions - this is critical for building a correct ATS
    print("Trimming alignment for highly gapped positions (80% or more)")
    position_filtered_sequences, kept_positions = sca.filterPos(standard_sequences, [1], 0.8)

    print("Alignment size post-trimming: {} positions".format(len(position_filtered_sequences[0])))

    print('Collecting residue data from PDB')
    pdb_residues = get_pdb_residues(options.pdbid, options.chainID, QUIET=True)

    print('Parsing sequence and pdb_ats')
    pdb_sequence, pdb_ats = get_pdb_sequence(pdb_residues)

    print('Calculating distances')
    pdb_distances = get_pdb_distances(pdb_residues)

    print("Finding reference sequence using Bio.pairwise2.align.globalxx")
    reference_index = locate_reference(position_filtered_sequences, pdb_sequence)

    print("Index of reference sequence: {}".format(reference_index))
    sequences, ats = sca.makeATS(position_filtered_sequences, pdb_ats, pdb_sequence, reference_index, options.truncate)

    # filtering sequences and positions, calculations of effective number of seqs
    print("Conducting sequence and position filtering: alignment size is {} seqs, {} pos".format(
        len(sequences), len(sequences[0]))
    )
    print(f'ATS size: {len(ats)}')
    print(f'Dim Distance Matrix: {len(pdb_distances)} x {len(pdb_distances[0])}')

    alg0, seqw0, seqkeep = sca.filterSeq(sequences, reference_index, max_fracgaps=PARAMETERS[1],
                                         min_seqid=PARAMETERS[2],
                                         max_seqid=PARAMETERS[3])

    headers = [standard_headers[s] for s in seqkeep]
    alg1, iposkeep = sca.filterPos(alg0, seqw0, PARAMETERS[0])
    ats = [ats[i] for i in iposkeep]
    distance_matrix = pdb_distances[np.ix_(iposkeep, iposkeep)]
    effseqsprelimit = int(seqw0.sum())
    Nseqprelimit = len(alg1)
    print("After filtering: alignment size is {} seqs, {} effective seqs, {} pos".format(
        len(alg1), effseqsprelimit, len(alg1[0]))
    )

    alg = alg1
    hd = headers

    # calculation of final MSA, sequence weights
    seqw = sca.seqWeights(alg)
    effseqs = seqw.sum()
    msa_num = sca.lett2num(alg)
    Nseq, Npos = msa_num.shape
    structPos = [i for (i, k) in enumerate(ats) if k != '-']
    print(FINAL_MSG.format(Nseq, effseqs, Npos, len(ats), len(structPos), len(distance_matrix), len(distance_matrix[0])))

    path_list = os.path.split(options.alignment)
    fn = path_list[-1]
    fn_noext = fn.split(".")[0]

    with open("Outputs/" + fn_noext + "processed" + ".fasta", "w") as f:
        for i in range(len(alg)):
            f.write(">%s\n" % (hd[i]))
            f.write(alg[i] + "\n")

    sequence_dict = {
        'alg': alg,
        'hd': hd,
        'msa_num': msa_num,
        'seqw': seqw,
        'Nseq': Nseq,
        'Npos': Npos,
        'ats': ats,
        'effseqs': effseqs,
        'NseqPrelimit': Nseqprelimit,
        'effseqsPrelimit': effseqsprelimit,
        'pdbid': options.pdbid,
        'pdb_chainID': options.chainID,
        'distmat': distance_matrix,
        'i_ref': reference_index,
        'trim_parameters': PARAMETERS,
        'truncate_flag': options.truncate
    }

    if options.outputfile is not None:
        fn_noext = options.outputfile
    print("Opening database file " + "Outputs/" + fn_noext)
    data_bank = {
        'sequence': sequence_dict
    }
    with open("Outputs/" + fn_noext + ".db", "wb") as db_out:
        pickle.dump(data_bank, db_out)
    _end = time.time()
    print(f'Completed in {_end-_start} sec')
