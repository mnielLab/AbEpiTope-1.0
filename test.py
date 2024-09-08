from src.main import StructureData, EvalAbAgs
from pathlib import Path
PDBDIR = Path.cwd() / "sample_pdbs"
ENCDIR = Path.cwd() / "encodings"
TMPDIR = Path.cwd() / "temporary"
OUTDIR = Path.cwd() / "output"
#making the inverse folding encodings takes a while...
data = StructureData()
data.encode_proteins(PDBDIR, ENCDIR, TMPDIR)
eval_abags = EvalAbAgs(data)
eval_abags.predict(OUTDIR)