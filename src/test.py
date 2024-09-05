
from main import StructureData
from pathlib import Path
PDBDIR = Path("/home/projects/vaccine/people/joacli/AntiInterNet-1.0/sample_pdbs")
ENCDIR = Path("/home/projects/vaccine/people/joacli/AntiInterNet-1.0/output/encodings")
TMPDIR = Path("/home/projects/vaccine/people/joacli/AntiInterNet-1.0/output/temporary")


data = StructureData()
data.encode_proteins(PDBDIR, ENCDIR, TMPDIR)