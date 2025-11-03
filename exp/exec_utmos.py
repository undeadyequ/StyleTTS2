import subprocess
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run UTMOS pipeline")
    parser.add_argument("--inp_dir", type=str, required=True, help="Input directory for predict.py")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--summary_dir", type=str, default=None, help="File to append summary output")

    args = parser.parse_args()


    # Step 1. Activate conda environment
    conda_env = "utmos"
    project_dir = os.path.expanduser("~/Project/UTMOS-demo/")
    input_dir = args.inp_dir
    output_file = os.path.dirname(input_dir) + "/utmos_{}.txt".format(os.path.basename(input_dir))
    if args.summary_dir is None:
        if input_dir[-1] == "/":
            input_dir = input_dir[:-1]
        utmos_summary = os.path.dirname(input_dir) + "/utmos_summary.txt"
    else:
        utmos_summary = args.summary_dir + "/utmos_summary.txt"
    batch_size = args.bs

    # Step 2. Compose the command to run inside the environment
    cmd = (
        f"conda run -n {conda_env} python predict.py "
        f"--mode predict_dir --inp_dir {input_dir} "
        f"--bs {batch_size} --out {output_file}"
    )

    # Step 3. Run the command in the project directory
    result = subprocess.run(cmd, shell=True, cwd=project_dir)

    # Step 4. Check result
    if result.returncode == 0:
        print("✅ Step 1 UTMOS prediction completed successfully.")
    else:
        print(f"❌ Error running UTMOS prediction (exit code {result.returncode}).")


    # --- Step 4: Compute average and append to summary ---
    cmd2 = (
        f"conda run -n {conda_env} python compute_average.py "
        f"{output_file} >> {utmos_summary}"
    )

    result2 = subprocess.run(cmd2, shell=True, cwd=project_dir)

    if result2.returncode == 0:
        print(f"✅ Step 4 completed. Results appended to {utmos_summary}")
    else:
        print(f"❌ Error in compute_average (exit code {result2.returncode}).")

if __name__ == "__main__":
    main()