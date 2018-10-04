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
import os
import numpy as np
import scaTools as sca
import pickle
import argparse
from Bio.pairwise2 import align


FINAL_MSG = """Final alignment parameters:
    Number of sequences, M: {}
    Number of effective sequences, M': {}
    Number of alignment positions, L: {}
    Number of positions in the ats: {}
    Number of structure positions mapped: {}
    Size of the distance matrix: {} x {}
"""

DEFAULT_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("alignment", help='Input Sequence Alignment')
        parser.add_argument("-s", "--pdb", dest="pdbid", help="PDB identifier (ex: 1RX2)")
        parser.add_argument("-c", "--chainID", dest="chainID", default='A',
                            help="chain ID in the PDB for the reference sequence")
        parser.add_argument("-t", "--truncate", action="store_true", dest="truncate", default=False,
                            help="truncate the alignment to the positions in the reference PDB, default: False")
        parser.add_argument("--output", dest="outputfile", default=None, help="specify an outputfile name")
        options = parser.parse_args()

        PARAMETERS = [0.2, 0.2, 0.2, 0.8]

        headers_full, sequences_full = sca.readAlg(options.alignment)
        print("Loaded alignment of {} sequences, {} positions.".format(len(headers_full), len(sequences_full[0])))

        print("Checking alignment for non-standard amino acids")
        alg_out, hd_out = list(), list()
        for i, k in enumerate(sequences_full):
            has_invalid = False
            for aa in k:
                if aa not in DEFAULT_AMINO_ACIDS:
                    has_invalid = True
                    break
            if has_invalid:
                continue
            else:
                alg_out.append(k)
                hd_out.append(headers_full[i])
        headers_full = hd_out
        sequences_full = alg_out
        print("Alignment size after removing sequences with non-standard amino acids: {}".format(len(sequences_full)))

        # Do an initial trimming to remove excessively gapped positions - this is critical for building a correct ATS
        print("Trimming alignment for highly gapped positions (80% or more)")
        alg_out, poskeep = sca.filterPos(sequences_full, [1], 0.8)
        sequences_ori = sequences_full
        sequences_full = alg_out
        print("Alignment size post-trimming: {} positions".format(len(sequences_full[0])))

        seq_pdb, ats_pdb, dist_pdb = sca.pdbSeq(options.pdbid, options.chainID)
        print("Finding reference sequence using Bio.pairwise2.align.globalxx")
        score = list()
        for k, s in enumerate(sequences_full):
            score.append(align.globalxx(seq_pdb, s, one_alignment_only=1, score_only=1))
        i_ref = score.index(max(score))
        options.i_ref = i_ref
        print("Index of reference sequence: {}".format(i_ref))
        print(headers_full[i_ref])
        print(sequences_full[i_ref])
        sequences, ats = sca.makeATS(sequences_full, ats_pdb, seq_pdb, i_ref, options.truncate)
        dist_new = np.zeros((len(ats), len(ats)))
        for (j, pos1) in enumerate(ats):
            for (k, pos2) in enumerate(ats):
                if k != j:
                    if (pos1 == '-') or (pos2 == '-'):
                        dist_new[j, k] = 1000
                    else:
                        ix_j = ats_pdb.index(pos1)
                        ix_k = ats_pdb.index(pos2)
                        dist_new[j, k] = dist_pdb[ix_j, ix_k]
        dist_pdb = dist_new
        # filtering sequences and positions, calculations of effective number of seqs
        print("Conducting sequence and position filtering: alignment size is {} seqs, {} pos".format(
            len(sequences), len(sequences[0]))
        )
        print("ATS and distmat size - ATS: {}, distmat: {} x {}".format(len(ats), len(dist_pdb), len(dist_pdb[0])))

        alg0, seqw0, seqkeep = sca.filterSeq(sequences, max_fracgaps=PARAMETERS[1],
                                             min_seqid=PARAMETERS[2],
                                             max_seqid=PARAMETERS[3])
       
        headers = [headers_full[s] for s in seqkeep]
        alg1, iposkeep = sca.filterPos(alg0, seqw0, PARAMETERS[0])
        ats = [ats[i] for i in iposkeep]
        if options.pdbid is not None: 
            distmat = dist_pdb[np.ix_(iposkeep, iposkeep)]
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
        print(FINAL_MSG.format(Nseq, effseqs, Npos, len(ats), len(structPos), len(distmat), len(distmat[0])))

        path_list = options.alignment.split(os.sep)
        fn = path_list[-1]
        fn_noext = fn.split(".")[0]
        f = open("Outputs/" + fn_noext + "processed" + ".fasta", "w")
        for i in range(len(alg)):
            f.write(">%s\n" % (hd[i]))
            f.write(alg[i] + "\n")
        f.close()

        sequence_dict = {
            'alg': alg,
            'hd': hd,
            'msa_num': msa_num,
            'seqw': seqw,
            'Nseq': Nseq,
            'Npos': Npos,
            'ats': ats,
            'effseqs': effseqs,
            'limitseqs': options.Nselect,
            'NseqPrelimit': Nseqprelimit,
            'effseqsPrelimit': effseqsprelimit,
            'pdbid': options.pdbid,
            'pdb_chainID': options.chainID,
            'distmat': distmat,
            'i_ref': i_ref,
            'trim_parameters': options.parameters,
            'truncate_flag': options.truncate
        }

        if options.outputfile is not None:
            fn_noext = options.outputfile
        print("Opening database file " + "Outputs/" + fn_noext)
        data_bank = {
            'sequence': sequence_dict
        }
        pickle.dump(data_bank, open("Outputs/" + fn_noext + ".db", "wb"))

