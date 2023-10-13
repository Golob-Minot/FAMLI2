from FAMLI2 import load_diamond_blast6, FAMLI2
import taichi as ti

ti.init(debug=True)


def test_load_diamond_blast6():

    # Load data from alignments formatted in BLAST6 format
    result = load_diamond_blast6("tests/data/GGH0010.n3.aln.gz")

    # Assert that the function returns an instance of FAMLI2
    assert isinstance(result, FAMLI2)

    # Run the FAMLI analysis
    result.run_famli()
