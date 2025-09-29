import argparse
import os
import pandas as pd
from tdc import Oracle
from rdkit import RDLogger
from in_virtuo_gen.utils.fragments import bridge_smiles_fragments, order_fragments_by_attachment_points, smiles2frags
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
RDLogger.DisableLog('rdApp.*')
import multiprocessing as mp
def calculate_property_batch(smiles_batch, prop, device=0):
    """Calculate property for a batch of SMILES"""
    try:
        oracle = Oracle(prop, device=device)
        return oracle(smiles_batch)
    except Exception as e:
        print(f"Error calculating {prop}: {e}")
        return [None] * len(smiles_batch)


def process_property_parallel(df, prop, batch_size=1000, max_workers=None):
    """Process property calculation in parallel batches"""
    smiles_list = df['smiles'].tolist()
    n_samples = len(smiles_list)

    if max_workers is None:
        max_workers = min(20, mp.cpu_count())

    # Split into batches
    batches = [smiles_list[i:i + batch_size] for i in range(0, n_samples, batch_size)]

    print(f'Processing {len(batches)} batches of ~{batch_size} molecules each using {max_workers} workers...')

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Assign different GPU devices to different workers
        futures = []
        for i, batch in enumerate(batches):
            device = i % max_workers  # Cycle through available GPUs
            future = executor.submit(calculate_property_batch, batch, prop, device)
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            batch_results = future.result()
            results.extend(batch_results)
            print(f'Completed batch {i+1}/{len(batches)}')

    return results
if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--datapath", type=str, default="data/frags_zinc250.csv")
    args.add_argument("--outpath", type=str, default="in_virtuo_reinforce/vocab/zinc250k.csv")
    args.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    args.add_argument("--max_workers", type=int, default=8, help="Max parallel workers")
    args.add_argument("--resume", action="store_true", help="Resume from existing file")
    args = args.parse_args()
    props = ['albuterol_similarity',
                 'amlodipine_mpo',
                 'celecoxib_rediscovery',
                 'deco_hop',
                 'drd2',
                 'fexofenadine_mpo',
                 'gsk3b',
                 'isomers_c7h8n2o2',
                 'isomers_c9h10n2o2pf2cl',
                 'jnk3',
                 'median1',
                 'median2',
                 'mestranol_similarity',
                 'osimertinib_mpo',
                 'perindopril_mpo',
                 'qed',
                 'ranolazine_mpo',
                 'scaffold_hop',
                 'sitagliptin_mpo',
                 'thiothixene_rediscovery',
                 'troglitazone_rediscovery',
                 'valsartan_smarts',
                 'zaleplon_mpo']


    # prepare smiles once
    samples = pd.read_csv(args.datapath, header=None)
    smiles_list = samples.iloc[:, 0].tolist()
    smiles = [bridge_smiles_fragments(s.split()) for s in smiles_list]
    df = pd.DataFrame({'smiles': smiles})

    for prop in props:
        if prop not in df.columns:
            print(f'\nCalculating {prop}...')

            # Use parallel processing for property calculation
            df[prop] = process_property_parallel(df, prop, args.batch_size, args.max_workers)

            # Save intermediate results
            os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
            df.to_csv(args.outpath, index=False)
            print(f'Intermediate results saved to {args.outpath}')
        else:
            print(f'{prop} already calculated, skipping...')
    # print(df.head())
