import subprocess
import os
import shutil
import time

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Configuration
    n_iter = 10000 # Increased iterations
    r_iter = 16    # Restarts
    iterations = 2 # Number of full loops
    
    # Paths
    packer_exe = "./tree_packer_v21"
    bp_exe = "./bp"
    rot_script = "optimize_rotation.py"
    gen_script = "lattice_generator.py"
    submission_file = "submission.csv"
    
    # 0. Generate Lattice Submission
    print("--- Step 0: Generating Lattice Submission ---")
    run_command(f"python {gen_script}")
    
    # 1. Initial Packing
    print("--- Step 1: Initial Packing ---")
    try:
        run_command(f"{packer_exe} -n {n_iter} -r {r_iter}")
    except subprocess.CalledProcessError:
        print("Packer failed.")
        raise

    # Check output
    if os.path.exists("submission_v21.csv"):
        print("Found submission_v21.csv, using it.")
        shutil.copy("submission_v21.csv", submission_file)
    else:
        print("Warning: submission_v21.csv not found. Checking if submission.csv was modified.")
        pass

    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")
        
        # 2. Backward Propagation
        print("--- Step 2: Backward Propagation ---")
        run_command(f"{bp_exe} {submission_file} {submission_file}")
        
        # 3. Rotation Optimization
        print("--- Step 3: Rotation Optimization ---")
        run_command(f"python {rot_script} {submission_file} {submission_file} 1")
        
        # Save intermediate result
        shutil.copy(submission_file, f"submission_iter_{i+1}.csv")
        
    print("\nLattice experiment complete.")

if __name__ == "__main__":
    main()
