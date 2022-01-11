#!/usr/bin/env python3

import time # to time calculations for users
import pandas as pd # for writing output to Excel
from collation_parser import collation_parser, tei_collation_parser, vmr_collation_parser
from collation_factorizer import collation_factorizer
import argparse # for parsing command-line input

"""
Entry point to the script. Parses command-line arguments and calls the core functions.
"""
def main():
    parser = argparse.ArgumentParser(description="Reads a collation into a matrix and performs rank estimation for NMF on this matrix.")
    parser.add_argument("-w", metavar="ambiguous_reading_prefix", type=str, help="Prefix used to indicate ambiguous readings (e.g., \"W\" or \"zw-\"). Only used with TEI XML inputs.")
    parser.add_argument("-s", metavar="suffix", type=str, action="append", help="Suffixes for first hand, main text, corrector, alternate text, and multiple attestation (e.g., *, T, C, C1, C2, C3, A, A1, A2, K, K1, K2, /1, /2). Witness sigla with these suffixes will be collapsed with their base witnesses. If more than one suffix is used, this argument can be specified multiple times.")
    parser.add_argument("-t", metavar="trivial_reading_types", type=str, action="append", help="Reading types to treat as trivial and collapse with the previous substantive reading (e.g., defective, orthographic, nomSac). If more than one type is applicable, this argument can be specified multiple times.")
    parser.add_argument("-z", metavar="ignored_reading_types", type=str, action="append", help="Reading types to treat as lacunae and ignore (e.g., lac, ambiguous). If more than one type is applicable, this argument can be specified multiple times.")
    parser.add_argument("-p", metavar="min_extant_proportion", type=float, default=0.95, help="Minimum proportion of variation units at which a witness must have an extant reading in order to be included in the primary matrix to be factored. (Default: 0.95)")
    parser.add_argument("--use-tfidf", action="store_true", help="If set, weigh the final collation matrix by term frequency-inverse document frequency (TF-IDF).")
    parser.add_argument("-nrun", type=int, default=10, help="Number of NMF trials to run when evaluating each rank. (Default: 10)")
    parser.add_argument("--verbose", action="store_true", help="If set, enable logging for debugging and performance.")
    parser.add_argument("-o", metavar="output", type=str, help="Filename for the Excel (.xlsx) or JSON (.json) output containing rank estimation metrics (if none is specified, then the output is written to the console).")
    parser.add_argument("input", type=str, help="Collation input. A TEI XML collation file (.xml) or an index for the ECM collation hosted in the New Testament Virtual Manuscript Room (NTVMR) is expected. (For the latter, specify indices as Acts [for the whole book], Acts.1 or Acts.1-5 [for one or more chapters], or Acts.1.1-5 [for one or more verses].)")
    parser.add_argument("minrank", type=int, help="Minimum rank to evaluate.")
    parser.add_argument("maxrank", type=int, help="Maximum rank to evaluate.")
    args = parser.parse_args()
    # Parse the optional arguments:
    min_extant_proportion = args.p
    if min_extant_proportion < 0 or min_extant_proportion > 1:
        print("Error: Parameter -p must be value between 0 and 1.")
        exit(1)
    use_tfidf = args.use_tfidf
    ambiguous_reading_prefix = "" if args.w is None else args.w
    subwitness_suffixes = [] if args.s is None else args.s
    trivial_reading_types = [] if args.t is None else args.t
    ignored_reading_types = [] if args.z is None else args.z
    n_run = args.nrun
    output_addr = args.o
    verbose = args.verbose
    # Parse the positional arguments:
    input_addr = args.input
    min_rank = args.minrank
    max_rank = args.maxrank
    # Initialize the collation_parser instance and use it to read in the collation input:
    cp = None # the collation_parser instance; depending on the type of input, it will either be a tei_collation_parser or a vmr_collation_parser
    if input_addr.endswith(".xml"):
        cp = tei_collation_parser(min_extant_proportion, use_tfidf, ambiguous_reading_prefix, subwitness_suffixes, trivial_reading_types, ignored_reading_types, verbose)
        try:
            cp.read(input_addr)
        except Exception as e:
            print("Error reading XML input %s: %s: %s" % (input_addr, type(e), e))
            exit(1)
    else:
        cp = vmr_collation_parser(min_extant_proportion, use_tfidf, ambiguous_reading_prefix, subwitness_suffixes, trivial_reading_types, ignored_reading_types, verbose)
        try:
            cp.read(input_addr)
        except Exception as e:
            print("Error reading input for index %s: %s: %s" % (input_addr, type(e), e))
            exit(1)
    # Then initialize a collation_factorizer instance and perform rank estimation:
    cf = collation_factorizer(cp, verbose=verbose)
    rank_metrics = cf.estimate_rank(min_rank, max_rank, n_run)
    # Then write the rank estimation metrics to the appropriate output:
    if output_addr is None:
        for rank_metrics_dict in rank_metrics:
            print(rank_metrics_dict)
    else:
        if output_addr.endswith(".xlsx"):
            if verbose:
                print("Writing final collation matrix to Excel...")
            t0 = time.time()
            rank_metrics_df = pd.DataFrame(data=rank_metrics)
            #Then write the Excel collation table to output:
            rank_metrics_df.to_excel(output_addr, sheet_name="Rank Estimation", index=False)
            t1 = time.time()
            if verbose:
                print("Done in %0.4fs." % (t1 - t0))
        elif output_addr.endswith(".json"):
            if verbose:
                print("Writing final collation matrix to JSON...")
            t0 = time.time()
            rank_metrics_df = pd.DataFrame(data=rank_metrics)
            #Then write the Excel collation table to output:
            rank_metrics_df.to_json(output_addr, orient="records")
            t1 = time.time()
            if verbose:
                print("Done in %0.4fs." % (t1 - t0))
        else:
            print("Error: Unrecognized output file format. Please specify a file with format .xlsx or .json.")
            exit(1)
    exit(0)

if __name__=="__main__":
    main()