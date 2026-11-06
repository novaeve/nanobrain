from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "Geneformer" / "geneformer_seaad_joint_finetune.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
