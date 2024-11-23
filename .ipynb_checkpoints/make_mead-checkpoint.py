import os
import glob
import subprocess

files = glob.glob("/scratch/MEAD_DATA/MEAD_extracted/M003/**/*.mp4",recursive = True)
def convert_path_to_filename(path):
    # Remove the base directory and file extension
    relative_path = path.split("/MEAD_extracted/")[1].rsplit(".", 1)[0]
    
    # Replace slashes with underscores
    filename = relative_path.replace("/", "_")
    
    return filename

for f in files:
    referencefile = convert_path_to_filename(f)
    command = [
            "python", "run_pipeline.py",
            "--videofile", f,
            "--reference",referencefile,
            "--data_dir", "/scratch/souvik/data/"
        ]
    
    result = subprocess.run(command, capture_output=True, text=True)
            
    print(f"Output", result.stdout)
    print(f"Error ", result.stderr)