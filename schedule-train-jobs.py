import argparse
import os
import subprocess
import itertools
import sys

def write_job_file(tasks, job_name, embedding_model, output_file):
    print(f"#!/bin/bash", file=output_file)
    print(f"", file=output_file)
    print(f"#SBATCH --job-name={job_name}.train", file=output_file)
    print(f"#SBATCH --ntasks=1", file=output_file)
    print(f"#SBATCH --cpus-per-task=2", file=output_file)
    print(f"#SBATCH --ntasks-per-node=1", file=output_file)
    print(f"#SBATCH --time=48:00:00", file=output_file)
    print(f"#SBATCH --mem=16384M", file=output_file)
    print(f"#SBACTH --partition=gpu_shared_course", file=output_file)
    print(f"#SBATCH --gres=gpu:1", file=output_file)
    print(f"", file=output_file)
    print(f"module purge", file=output_file)
    print(f"", file=output_file)
    print(f"module load Miniconda3/4.3.27", file=output_file)
    print(f"module load CUDA/9.0.176", file=output_file)
    print(f"module load cuDNN/7.3.1-CUDA-9.0.176", file=output_file)
    print(f"", file=output_file)
    print(f"export PYTHONIOENCODING=utf", file=output_file)
    print(f"source activate dl", file=output_file)
    print(f"", file=output_file)
    print(f"srun python3 -u train_jmt.py --output {os.path.join('output', args.output, job_name)} --tasks {' '.join(task_set)} --embedding-model {embedding_model}", file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model", type=str, choices=["ELMo+GloVe", "bert-base-cased", "bert-large-cased"], required=True,
        help="The embedding model to use."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="The folder to output the training information (relative to the existing 'output' folder)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, required=False,
        help="Write actions that will be undertaken to standard out but don't perform them."
    )

    args = parser.parse_args()

    train_tasks = ["pos", "vua", "snli"]

    task_sets = itertools.chain(*[itertools.combinations(train_tasks, r) for r in range(1, len(train_tasks) + 1)])

    job_folder = os.path.join("jobs", args.output)

    if not args.dry_run:
        try:
            os.makedirs(job_folder, exist_ok=False)
        except Exception:
            print("Output folder already exists pick another one.")
            exit(1)
    else:
        print(f"Making jobs folder: {job_folder}")

    for task_set in task_sets:
        job_name = "-".join(task_set)
        job_file_location = os.path.join(job_folder, job_name + ".sh")

        if args.dry_run:
            write_job_file(task_set, job_name, args.embedding_model, sys.stdout)
        else:
            with open(job_file_location, "w", encoding="utf-8") as f:
                write_job_file(task_set, job_name, args.embedding_model, f)

        if args.dry_run:
            print(f"Scheduling job: {job_name}")
        else:
            subprocess.call(
                ["sbatch", job_file_location]
            )
