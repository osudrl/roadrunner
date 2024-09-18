import argparse
import io
import logging
import os
import subprocess

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# ray.init(num_cpus=32)


# @ray.remote
# def upload_files(index, start, end, wid, src_path, dest_path):
#     for i in tqdm(range(start, end), desc=f'wid:{wid}, start->end:{start}-{end - 1}', leave=False):
#         filename = index.iloc[i]['filename']
#         subprocess.call(["rsync", "-avz", os.path.join(src_path, filename), os.path.join(dest_path, filename)])


def upload_files(index, start, end, src_path, dest_path):
    command = ""

    for i in range(start, end):
        filename = index.iloc[i]['filename']
        command += f"rsync -avz {os.path.join(src_path, filename)} {os.path.join(dest_path, filename)} & "
    command += "wait"
    # print(command)
    subprocess.run(command, capture_output=True, shell=True)
    # subprocess.call(["rsync", "-avz", os.path.join(src_path, filename), os.path.join(dest_path, filename)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer .pkl files from one system to another")
    parser.add_argument("--start", type=int, help="The starting integer for the file names", required=False, default=0)
    parser.add_argument("--end", type=int, help="The ending integer for the file names", required=False)
    parser.add_argument("--n_workers", type=int, help="Number of workers", required=True)
    parser.add_argument("--src_path", type=str,
                        help="Local or remote path. Eg: /home/bikram/workspace/roadrunner_refactor/offline_data/CassieHfield/20230617-234108/ or panditb@flip3.engr.oregonstate.edu:/nfs/stak/users/panditb/workspace/roadrunner_refactor/offline_data/CassieHfield/20230617-234108/",
                        required=True)
    parser.add_argument("--trg_path", type=str,
                        help="Local or remote path. Eg: panditb@flip3.engr.oregonstate.edu:/nfs/stak/users/panditb/workspace/roadrunner_refactor/offline_data/CassieHfield/20230617-234108/ or /home/bikram/workspace/roadrunner_refactor/offline_data/CassieHfield/20230617-234108/",
                        required=True)
    parser.add_argument("--replace", action='store_true', help="Replace existing files", required=False, default=False)
    args = parser.parse_args()

    logging.info(args.__dict__)

    if ':' in args.src_path:
        logging.info('Loading index from remote')
        # Remote
        var = args.src_path.split(':')
        assert len(var) == 2, f"Invalid src_path: Must be of the form user@host:/path/to/src"
        host, index_path = var
        index_path = os.path.join(index_path, "index.csv")
        ssh_command = f'ssh {host} "cat {index_path}"'
        try:
            output = subprocess.check_output(ssh_command, shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"{index_path} may not exist in {host}")
            raise e
        index = pd.read_csv(io.StringIO(output))

        logging.info(f"Number of files in {host}:{index_path}: {len(index)}")
    else:
        logging.info('Loading index from local')
        # Local
        index_path = os.path.join(args.src_path, "index.csv")
        index = pd.read_csv(index_path)
        # Check if index exist in src_path
        assert os.path.exists(index_path), f"{index_path} does not exist"

        logging.info(f"Number of files in {index_path}: {len(index)}")

    if not args.replace:
        # Check if files exist in trg_path
        if ':' in args.trg_path:
            logging.info('Checking if files exist in remote')
            var = args.trg_path.split(':')
            assert len(var) == 2, f"Invalid trg_path: Must be of the form user@host:/path/to/trg"
            host, trg_path = var
            ssh_command = f'ssh {host} "ls {trg_path}"'
            try:
                output = subprocess.check_output(ssh_command, shell=True, universal_newlines=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"{trg_path} may not exist in {host}")
                raise e
            files = output.split('\n')
            files = [f for f in files if f != '']
            logging.info(f"Number of files in {host}:{trg_path}: {len(files)}")
        else:
            logging.info('Checking if files exist in local')
            files = os.listdir(args.trg_path) if os.path.exists(args.trg_path) else []
            logging.info(f"Number of files in {args.trg_path}: {len(files)}")

        # Remove files that already exist in trg_path
        index = index[~index['filename'].isin(files)]

    logging.info(f"Number of files to be transferred: {len(index)}")

    if args.end is None:
        args.end = len(index) - 1

    files_per_worker = np.ceil((args.end - args.start + 1) / args.n_workers).astype(int)

    workers = []

    os.makedirs(args.trg_path, exist_ok=True)

    # for i in range(args.start, args.end + 1, files_per_worker):
    #     workers.append(
    #         upload_files.remote(index, i, min(i + files_per_worker, len(index)), len(workers), args.src_path,
    #                             args.trg_path))
    #
    # ray.get(workers)

    tq = tqdm(range(args.start, args.end + 1, args.n_workers))
    for i in tq:
        j = min(i + args.n_workers, len(index))
        tq.set_description(f'Transferring file from {i} to {j}')
        upload_files(index, i, j, args.src_path, args.trg_path)
