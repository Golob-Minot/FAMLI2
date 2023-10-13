from FAMLI2 import load_diamond_blast6, FAMLI2
import taichi as ti

ti.init(debug=True)


def test_load_diamond_blast6():

    # Load data from alignments formatted in BLAST6 format
    result = load_diamond_blast6("tests/data/GGH0010.n3.aln.gz")

    # Assert that the function returns an instance of FAMLI2
    assert isinstance(result, FAMLI2)

    # Reads are aligned to multiple references
    assert result.aln_ad.var['n_subj_per_read'].max() > 1

    # Run the FAMLI analysis
    result.run_famli()

    # Reads are aligned uniquely to a single reference
    assert result.aln_ad.var['n_subj_per_read'].max() == 1
