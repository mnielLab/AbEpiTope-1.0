from abepitope.main import StructureData, EvalAbAgs
from pathlib import Path
PDBDIR = Path.cwd() / "abag_exampledata" / "Cancer"
ENCDIR = Path.cwd() / "encodings"
TMPDIR = Path.cwd() / "temporary"
OUTDIR = Path.cwd() / "output"
#making the inverse folding encodings takes a while...
data = StructureData()
data.encode_proteins(PDBDIR, ENCDIR, TMPDIR)
eval_abags = EvalAbAgs(data, decimal_precision=6)
eval_abags.predict(OUTDIR)
