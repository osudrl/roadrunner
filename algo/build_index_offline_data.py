import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create index.csv containing all files in a directory")
    parser.add_argument("--dir_path", type=str,
                        help="Eg: /home/bikram/workspace/roadrunner_refactor/offline_data/CassieHfield/HEAD/old/",
                        required=True)
    args = parser.parse_args()

    # Check if the directory exists
    assert os.path.exists(args.dir_path), "Directory does not exist"

    files = os.listdir(args.dir_path)

    # Create the index.csv file
    df = pd.DataFrame(files, columns=['filename'])
    df.to_csv(os.path.join(args.dir_path, 'index.csv'), index=True)

    logging.info(f"Index build completed for {args.dir_path}. Total files {len(df)}")
