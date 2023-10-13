from FAMLI2 import load_diamond_blast6_lowmem, FAMLI2


def test_load_diamond_blast6_lowmem():

    # Load data from alignments formatted in BLAST6 format
    result = load_diamond_blast6_lowmem("tests/data/GGH0010.n3.aln.gz")

    # Assert that the function returns an instance of FAMLI2
    assert isinstance(result, FAMLI2)
