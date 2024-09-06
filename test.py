from src.main import StructureData, AntiInterNet, AntiScout

from pathlib import Path
PDBDIR = Path("/home/projects/vaccine/people/joacli/AntiInterNet-1.0/sample_pdbs")
ENCDIR = Path("/home/projects/vaccine/people/joacli/AntiInterNet-1.0/output/encodings")
TMPDIR = Path("/home/projects/vaccine/people/joacli/AntiInterNet-1.0/output/temporary")

#making the inverse folding encodings takes a while...
data = StructureData()
data.encode_proteins(PDBDIR, ENCDIR, TMPDIR)