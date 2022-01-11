#!/usr/bin/env python3

import time # to time calculations for users
import numpy as np # matrix support
import nimfa as nf # for performing non-negative matrix factorization (NMF)
import scipy as sp # for solving optimization problems behind classifying lacunose witnesses
import pandas as pd # for writing output to Excel
import json # for writing output to Excel
from collation_parser import *

"""
Base class for applying non-negative matrix factorization (NMF) to a collation matrix.
"""
class collation_factorizer():
    """
	Constructs a new collation_factorizer with the given settings.
	"""
    def __init__(self, collation_parser, verbose=False):
        self.collation_parser = collation_parser # internal instance of the parser for the input collation data to be factorized
        self.verbose = verbose # flag indicating whether or not to print timing and debugging details for the user
        self.rank = 1 # number of latent groups
        self.factorizer = nf.Lsnmf(self.collation_parser.collation_matrix, seed="nndsvd", max_iter=10, rank=self.rank, track_error=True) # NMF least-squares factorizer to be applied to the collation matrix
        self.fit_summary = {} # dictionary of NMF fitness and performance metrics keyed by name
        self.basis_factor = np.zeros((len(self.collation_parser.readings), self.rank)) # "profile" (readings x rank) factor matrix
        self.coef_factor = np.zeros((self.rank, len(self.collation_parser.witnesses))) # "mixture" (rank x witnesses) factor matrix
        self.fragmentary_coef_factor = np.zeros((self.rank, len(self.collation_parser.fragmentary_witnesses))) # "mixture" (rank x fragmentary_witnesses) factor matrix for fragmentary witnesses

    """
    Performs rank estimation on the primary collation matrix for the ranks in the given range.
    Optionally, a number of trials to run for each rank can be specified.
    The output is a list of rank estimation results (in dictionary form).
    """
    def estimate_rank(self, min_rank, max_rank, n_run=10):
        if self.verbose:
            print("Estimating rank in range [%d, %d] using %d trials for each rank (this may take some time)..." % (min_rank, max_rank, n_run))
        t0 = time.time()
        rank_metrics = []
        metrics = ["cophenetic", "rss", "evar", "sparseness"]
        # For rank estimation, use random seeding and a small number of iterations:
        self.factorizer = nf.Lsnmf(self.collation_parser.collation_matrix, seed="random_vcol", max_iter=10, rank=self.rank, track_error=True)
        rank_est_dict = self.factorizer.estimate_rank(rank_range=range(min_rank, max_rank + 1), what=metrics, n_run=n_run) # evaluate the specified metrics for each rank
        for r in range(min_rank, max_rank + 1):
            rank_est_metrics = rank_est_dict[r]
            rank_metric_dict = {}
            for metric in metrics:
                rank_metric_dict["rank"] = r
                if metric == "sparseness":
                    # Separate the sparseness coefficients into their own named entries:
                    rank_metric_dict["basis_sparseness"] = rank_est_metrics[metric][0]
                    rank_metric_dict["mixture_sparseness"] = rank_est_metrics[metric][1]
                else:
                    rank_metric_dict[metric] = rank_est_metrics[metric]
            rank_metrics.append(rank_metric_dict)
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        return rank_metrics

    """
    Factors the collation into factors of a given rank using NMF
    and finds the optimal mixture coefficients for fragmentary witnesses using the best-found basis matrix.
    The best-found factors are stored internally.
    """
    def factorize_collation(self, rank):
        if self.verbose:
            print("Factorizing collation matrix into factors of rank %d..." % rank)
        t0 = time.time()
        # For factorization, use NNDSVD seeding and a larger number of iterations:
        self.rank = rank
        self.factorizer = nf.Lsnmf(self.collation_parser.collation_matrix, seed="nndsvd", max_iter=100, rank=self.rank, track_error=True)
        nmf_fit = self.factorizer()
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        # Populate the fitness and performance metrics:
        self.fit_summary = {"rank": self.rank, "time (s)": t1 - t0, "n_iter": nmf_fit.fit.n_iter, "rss": nmf_fit.fit.rss(), "evar": nmf_fit.fit.evar(), "basis_sparseness": nmf_fit.fit.sparseness()[0], "mixture_sparseness": nmf_fit.fit.sparseness()[1]}
        # Get the factor matrices:
        self.basis_factor = nmf_fit.basis()
        self.coef_factor = nmf_fit.coef()
        # Then evaluate the mixture coefficients for the fragmentary witnesses using non-negative least squares (NNLS) optimization with the basis factor:
        if self.verbose:
            print("Finding optimal mixture coefficients for fragmentary witnesses...")
        t0 = time.time()
        self.fragmentary_coef_factor = np.zeros((self.rank, len(self.collation_parser.fragmentary_witnesses)))
        for j in range(len(self.collation_parser.fragmentary_witnesses)):
            witness_vector = np.array([self.collation_parser.fragmentary_collation_matrix[i, j] for i in range(len(self.collation_parser.readings))]) # because for some reason, numpy.ndarray.flatten() leaves the column slice as a 2D array
            witness_coefs, rnorm = sp.optimize.nnls(self.basis_factor, witness_vector)
            self.fragmentary_coef_factor[:, j] = witness_coefs[:]
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        return

    """
    Writes the NMF factors and the fragmentary witness mixture coefficients for the current rank to a specified Excel file.
    """
    def to_excel(self, output_addr):
        # Then convert the NumPy collation matrix to a Pandas DataFrame:
        if self.verbose:
            print("Writing NMF results to Excel...")
        t0 = time.time()
        # First, convert all NumPy matrices to Pandas DataFrames:
        fit_summary_df = pd.DataFrame(data=[self.fit_summary])
        basis_factor_df = pd.DataFrame(data=self.basis_factor, index=self.collation_parser.readings, columns=["Cluster " + str(r) for r in range(1, self.rank + 1)])
        coef_factor_df = pd.DataFrame(data=self.coef_factor, index=["Cluster " + str(r) for r in range(1, self.rank + 1)], columns=self.collation_parser.witnesses)
        fragmentary_coef_factor_df = pd.DataFrame(data=self.fragmentary_coef_factor, index=["Cluster " + str(r) for r in range(1, self.rank + 1)], columns=self.collation_parser.fragmentary_witnesses)
        #Then write them to separate sheets in the Excel output:
        writer = pd.ExcelWriter(output_addr)
        fit_summary_df.to_excel(writer, sheet_name="Summary", index=False)
        basis_factor_df.to_excel(writer, sheet_name="Group Profiles")
        coef_factor_df.to_excel(writer, sheet_name="Witness Groupings")
        fragmentary_coef_factor_df.to_excel(writer, sheet_name="Fragmentary Witness Groups")
        writer.save()
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        return

    """
    Writes the NMF factors and the fragmentary witness mixture coefficients for the current rank to JSON strings.
    A dictionary mapping each table's name to its JSON serialization is returned.
    """
    def to_json(self):
        # Then convert the NumPy collation matrix to a Pandas DataFrame:
        if self.verbose:
            print("Writing basis and mixture matrix factors to JSON...")
        t0 = time.time()
        # First, convert all NumPy matrices to Pandas DataFrames:
        fit_summary_df = pd.DataFrame(data=[self.fit_summary])
        basis_factor_df = pd.DataFrame(data=self.basis_factor, index=self.collation_parser.readings, columns=["Cluster " + str(r) for r in range(1, self.rank + 1)])
        coef_factor_df = pd.DataFrame(data=self.coef_factor, index=["Cluster " + str(r) for r in range(1, self.rank + 1)], columns=self.collation_parser.witnesses)
        fragmentary_coef_factor_df = pd.DataFrame(data=self.fragmentary_coef_factor, index=["Cluster " + str(r) for r in range(1, self.rank + 1)], columns=self.collation_parser.fragmentary_witnesses)
        #Then combine their JSON serializations in a JSON object:
        fit_summary_json = fit_summary_df.to_json(orient="records")
        basis_factor_json = basis_factor_df.to_json(orient="records")
        coef_factor_json = coef_factor_df.to_json(orient="records")
        fragmentary_coef_factor_json = fragmentary_coef_factor_df.to_json(orient="records")
        json_output = json.dumps({"Group Profiles": basis_factor_json, "Witness Groupings": coef_factor_json, "Fragmentary Witness Groups": fragmentary_coef_factor_json})
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        return json_output