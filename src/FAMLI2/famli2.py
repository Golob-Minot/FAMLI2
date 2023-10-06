import scipy
import pandas as pd
import anndata as ad
import numpy as np
import logging
import argparse
import sys
import os
import time
import gzip
import json
import csv


class FAMLI2():
    def __init__(
        self,
        subjects=list(),
        queries=list(),
        slens=list(),
        alignments=list(),
        SD_MEAN_CUTOFF=1.0,
        STRIM_5=18,
        STRIM_3=18,
        ALN_SCORE_SCALE=0.9,
    ):
        """
            Most often invoked indirectly via a loader.
            subjects: A list of unique subject IDs corresponding to possible 'protein coding sequences'
            queries: A list of unique query IDs corresponding to observed short reads.
            slens: A list of integers corresponding to the subject lengths. Should be the same order and length
                    as subjects as above. Failure to do so will result in an assertion error
            alignments: A list of tuples: [
                (subject_index, query_index, sstart, send, bitscore)
                    0             1            2       3    4
            ]
            subject/query index is the index into subject / query lists as above...
        """
        # Store some hyperparamters
        self.SD_MEAN_CUTOFF = SD_MEAN_CUTOFF
        self.STRIM_5 = STRIM_5
        self.STRIM_3 = STRIM_3
        self.ALN_SCORE_SCALE = ALN_SCORE_SCALE

        # State
        self.filtered = False

        logging.info("Building anndata")
        self.aln_ad = ad.AnnData(
            scipy.sparse.coo_matrix(
                (
                    [True] * len(alignments),
                    (
                        [v[0] for v in alignments],
                        [v[1] for v in alignments],
                    )
                ),
                shape=(
                    len(subjects),
                    len(queries),
                ),
                dtype=bool
            ).tocsr(),
            obs=pd.DataFrame(index=subjects),
            var=pd.DataFrame(index=queries),
            dtype=bool,
        )
        # X is a boolean matrix of the current query-subject alignments
        logging.info("Adding bitscores")
        self.aln_ad.layers['bitscore'] = scipy.sparse.coo_matrix(
            (
                [v[4] for v in alignments],
                (
                    [v[0] for v in alignments],
                    [v[1] for v in alignments],
                )
            ),
            shape=(
                len(subjects),
                len(queries),
            ),
            dtype=np.float16
        ).tocsr()
        logging.info("Adding alignment starts")
        self.aln_ad.layers['sstart'] = scipy.sparse.coo_matrix(
            (
                [v[2] for v in alignments],
                (
                    [v[0] for v in alignments],
                    [v[1] for v in alignments],
                )
            ),
            shape=(
                len(subjects),
                len(queries),
            ),
            dtype=np.int32
        ).tocsr()
        logging.info("Adding alignment ends")
        self.aln_ad.layers['send'] = scipy.sparse.coo_matrix(
            (
                [v[3] for v in alignments],
                (
                    [v[0] for v in alignments],
                    [v[1] for v in alignments],
                )
            ),
            shape=(
                len(subjects),
                len(queries),
            ),
            dtype=np.int32
        ).tocsr()
        # Subject / obs length (used to calculate the coverage)
        self.aln_ad.obs['slen'] = slens
        self.update_mapping_state()
        logging.info("FAMLI2 anndata built.")

    # -- Bitscore filter -- #
    # Helper method for coverage filter.
    def make_bool_cov(self, slen, ss, se):
        cov = np.zeros(slen, dtype=bool)
        cov[ss:se + 1] = True
        return cov

    def make_subject_coverage(self, slen, sstarts, sends):
        # convert to dense as this point
        sstarts = np.ravel(sstarts.astype(int).toarray())
        sends = np.ravel(sends.astype(int).toarray())
        # Build up our boolean coverageograms for each read for this subject.
        s_bool_cov = [
            self.make_bool_cov(slen, ss, se)
            for (ss, se) in zip(
                sstarts, sends
            )
            if se != ss
        ]
        # Lots of edge case checking

        if len(s_bool_cov) == 1:
            # IF it's only one query for this subject..
            s_cov = s_bool_cov[0].astype(int)
        elif len(s_bool_cov) > 1:
            # More than one query for this subject
            s_cov = np.sum(s_bool_cov, axis=0)
        else:
            s_cov = np.zeros(slen)

        return s_cov

    # subject coverage filter
    def subject_coverage_filter(
        self,
        slen,
        sstarts,
        sends,
        strim_5,
        strim_3,
        sd_mean_cutoff
    ):
        s_cov = self.make_subject_coverage(
            slen,
            sstarts,
            sends
        )
        if s_cov.max() == 0:
            return False
        # Implicit else
        # Trim off the ends IF the subject is long enough
        if len(s_cov) > strim_3 + strim_5 + 10:
            s_cov = s_cov[strim_5: -strim_3]
        # Get our filter result (is there some coverage and is it above zero)
        s_filter_res = (np.mean(s_cov) > 0) & (np.std(s_cov) / np.mean(s_cov) <= sd_mean_cutoff)
        return s_filter_res

    def coverage_filter(self):
        logging.info("Starting Coverage Filter")
        # Add a column to our obs (subjects) if they passed
        self.aln_ad.obs['coverage_filter_pass'] = [
            self.subject_coverage_filter(
                slen,
                sstarts,
                sends,
                self.STRIM_5,
                self.STRIM_3,
                self.SD_MEAN_CUTOFF
            )
            for slen, sstarts, sends in zip(
                self.aln_ad.obs.slen,
                self.aln_ad.layers['sstart'],
                self.aln_ad.layers['send'],
            )
        ]
        # Do a bit of conversion here to deal with which sparse formats work the best
        new_mask = self.aln_ad.X.tolil()
        new_mask[
            ~self.aln_ad.obs.coverage_filter_pass
        ] = False
        self.aln_ad.X = new_mask.tocsr()
        logging.info("Completed Coverage Filter")
        self.apply_mask()
        self.update_mapping_state()

    # -- Bitscore filter -- #
    def bitscore_filter_iteration(self):
        # Generate a current 'subject weights' based on the alignment scores...
        self.aln_ad.obs['s_weight'] = np.ravel(self.aln_ad.layers['aln_score'].sum(axis=1)) / self.aln_ad.obs.slen
        # Then adjust the aln_score based on these revised weights
        self.aln_ad.layers['aln_score'] = self.aln_ad.layers['aln_score'].T.multiply(
            self.aln_ad.obs['s_weight']
        ).T
        # Renormalize the scores to sum up to 1
        self.aln_ad.layers['aln_score'] = self.aln_ad.layers['aln_score'] / self.aln_ad.layers['aln_score'].sum(axis=0)
        # The actual filter step in which we take normalize each alignment score to the max
        # for this read (query / var) and then only keep those that are above the ALN_SCORE_SCALE threshold..

        # self.aln_ad.layers['aln_score'].max(axis=0).toarray()
        max_score_per_query = self.aln_ad.layers['aln_score'].max(axis=0).toarray()
        max_score_per_query = max_score_per_query.clip(
            min=max_score_per_query[max_score_per_query > 0].min() / 10
        )
        self.aln_ad.X = self.aln_ad.X.multiply(
            self.aln_ad.layers['aln_score'].multiply(1 / max_score_per_query) >= self.ALN_SCORE_SCALE
        )
        self.apply_mask()
        self.update_mapping_state()

    def bitscore_filter(self, MAX_ITERATIONS=1000):
        logging.info("Starting bitscore filter (Iterative)")
        # Initialize an alignment score based on the bitscores
        self.aln_ad.layers['aln_score'] = self.aln_ad.layers['bitscore'] / self.aln_ad.layers['bitscore'].sum(axis=0)

        ix = 0
        while ix < MAX_ITERATIONS:
            ix += 1
            pre_n_aln = self.aln_ad.X.sum()
            self.bitscore_filter_iteration()
            post_n_aln = self.aln_ad.X.sum()
            logging.info(
                f'Iteration {ix}: {pre_n_aln:,d} to {post_n_aln:,d} Alignments.'
            )
            if pre_n_aln == post_n_aln:
                break
        logging.info(
            f'Completed bitscore filter after {ix} iterations.'
        )

    # -- Full pathway -- #
    def run_famli(self):
        # Step 1
        logging.info("Step 1: Initial Coverage Filter")
        self.coverage_filter()
        # Step 2
        logging.info("Step 2: Bitscore filtering")
        self.bitscore_filter()
        # Step 3
        logging.info("Step 3: Final coverage filter")
        self.coverage_filter()

        self.filtered = True

    # -- State updaters -- #

    def update_mapping_state(self):
        # Get the current mapping state of each read (var)
        logging.info("Updating state of alignments")
        self.aln_ad.var['n_subj_per_read'] = np.ravel(self.aln_ad.X.sum(axis=0))
        self.aln_ad.var['multimapped'] = self.aln_ad.var.n_subj_per_read > 1
        self.aln_ad.var['unique'] = self.aln_ad.var.n_subj_per_read == 1
        self.aln_ad.var['pruned'] = self.aln_ad.var.n_subj_per_read == 0
        logging.info(
            f'There are {self.aln_ad.var.multimapped.sum():,d} multimapped, {self.aln_ad.var.unique.sum():,d} uniquely mapped reads. {self.aln_ad.var.pruned.sum():,d} reads pruned.'
        )
        self.aln_ad.obs['nreads'] = np.ravel(self.aln_ad.X.sum(axis=1))

    def apply_mask(self):
        logging.info("Applying mask to alignments")
        # And then apply our mask...
        self.aln_ad.layers['bitscore'] = self.aln_ad.layers['bitscore'].multiply(self.aln_ad.X)
        self.aln_ad.layers['sstart'] = self.aln_ad.layers['sstart'].multiply(self.aln_ad.X)
        self.aln_ad.layers['send'] = self.aln_ad.layers['send'].multiply(self.aln_ad.X)

    def output_list(self):
        final_aln = self.aln_ad[self.aln_ad.obs.nreads > 0]

        final_cov = [
            self.make_subject_coverage(
                slen,
                sstarts,
                sends,
            )
            for slen, sstarts, sends in zip(
                final_aln.obs.slen,
                final_aln.layers['sstart'],
                final_aln.layers['send'],
            )
        ]

        return [
            {
                'id': subj,
                'nreads': nr,
                'coverage': c_f,
                'depth': c_m,
                'std': c_std,
                'length': slen
            } for (
                subj,
                nr,
                c_f,
                c_m,
                c_std,
                slen
            ) in zip(
                final_aln.obs_names,
                final_aln.obs.nreads,
                [(c > 0).mean() for c in final_cov],
                [c.mean() for c in final_cov],
                [c.std() for c in final_cov],
                final_aln.obs.slen,
            )
        ]

    def __repr__(self):
        return self.aln_ad.__repr__()


def load_diamond_blast6_lowmem(
        file,
        columns=[
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore",
            "qlen",
            "slen"
        ]):
    logging.info("Attempting to open alignment file")
    if file.endswith('.gz') or file.endswith('.gzip'):
        fh = gzip.open(file, 'rt')
    else:
        fh = open(file, 'rt')
    fr = csv.DictReader(
        fh,
        fieldnames=columns,
        delimiter='\t'
    )
    logging.info("Load alignments into memory")
    aln = [
        (
            r['qseqid'],  # 0
            r['sseqid'],  # 1
            int(r['slen']),  # 2
            int(r['sstart']),  # 3
            int(r['send']),  # 4
            float(r['bitscore'])  # 5
        ) for r in fr
    ]
    logging.info(f"Loading of {len(aln):,d} alignments complete")
    logging.info("Load in queries")
    queries = sorted({
        r[0] for r in aln
    })
    query_i = {
        s: i
        for (i, s) in enumerate(queries)
    }

    logging.info("Load in subject IDs")
    subjects = sorted({
        r[1] for r in aln
    })
    subject_i = {
        s: i
        for (i, s) in enumerate(subjects)
    }

    logging.info("Subject Lengths ordering..")
    slen_dict = {
        r[1]: r[3]
        for r in aln
    }
    slens = [
        int(slen_dict.get(s, 0))
        for s in subjects
    ]
    logging.info(
        f'There were {len(subjects):,d} protein-coding sequences and {len(queries):,d} reads with alignments'
    )

    logging.info("Building FAMLI2 object")
    return FAMLI2(
        subjects=subjects,
        queries=queries,
        slens=slens,
        alignments=[
            (
                subject_i.get(r[1]),
                query_i.get(r[0]),
                r[3],
                r[4],
                r[5],
            )
            for r in aln

        ]
    )


def load_diamond_blast6(file):
    logging.info("Using pandas to load blast6 format alignment")
    aln = pd.read_csv(
        file,
        delimiter='\t',
        names=[
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore",
            "qlen",
            "slen"
        ]
    )
    logging.info(f"Loading of {len(aln):,d} alignments complete")
    subjects = list(aln.sseqid.unique())
    subject_i = {
        s: i
        for (i, s) in enumerate(subjects)
    }
    aln['subject_i'] = aln.sseqid.apply(subject_i.get)
    queries = list(aln.qseqid.unique())
    query_i = {
        s: i
        for (i, s) in enumerate(queries)
    }
    aln['query_i'] = aln.qseqid.apply(query_i.get)
    logging.info(
        f'There were {len(subjects):,d} protein-coding sequences and {len(queries):,d} reads with alignments'
    )
    slens = aln.groupby('sseqid').max(numeric_only=True).slen.loc[
        subjects
    ]
    logging.info("Building FAMLI2 object")
    return FAMLI2(
        subjects=subjects,
        queries=queries,
        slens=slens,
        alignments=[
            (
                subject_i.get(r.sseqid),
                query_i.get(r.qseqid),
                r.sstart,
                r.send,
                r.bitscore
            )
            for i, r in aln.iterrows()
        ]
    )


def main():
    """Filter a set of alignments with FAMLI."""
    parser = argparse.ArgumentParser(
        description="""Filter a set of existing alignments in tabular
        format with FAMLI2""")

    parser.add_argument("--input",
                        type=str,
                        help="Location for input alignment file. For now, DIAMOND BLAST6 format please")
    parser.add_argument("--output",
                        type=str,
                        help="Location for output JSON file.")
    parser.add_argument("--logfile",
                        type=str,
                        help="""(Optional) Write log to this file.""")
    parser.add_argument("--sd-mean-cutoff",
                        default=1.0,
                        type=float,
                        help="""Threshold for filtering max SD / MEAN""")
    parser.add_argument("--strim-5",
                        default=18,
                        type=int,
                        help="""Amount to trim from 5' end of subject""")
    parser.add_argument("--strim-3",
                        default=18,
                        type=int,
                        help="""Amount to trim from 3' end of subject""")

    args = parser.parse_args()

    start_time = time.time()

    assert os.path.exists(args.input)

    # Set up logging
    logFormatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [FAMLI2] %(message)s'
    )
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    if args.logfile:
        # Write to file
        fileHandler = logging.FileHandler(args.logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    # Write to STDOUT
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info("Attemping to load input file")
    famli2 = load_diamond_blast6_lowmem(args.input)

    logging.info("Starting FAMLI (v1) filter")
    famli2.run_famli()
    logging.info("Filtering completed")

    if args.output:
        with open(args.output, "wt") as fo:
            json.dump(
                famli2.output_list(),
                fo,
                indent=4
            )

    elapsed = round(time.time() - start_time, 2)
    logging.info("Time elapsed: {:,}".format(elapsed))


if __name__ == "__main__":
    main()
