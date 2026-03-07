import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scipy.sparse as sp
import numpy as np
import scanpy as sc
import anndata as an
import multiprocessing as mp
from pathlib import Path



_CONN = None


def _init_worker(connectivities_path: str):
    """Runs once per worker process."""
    global _CONN
    _CONN = sp.load_npz(connectivities_path)



def get_cluster_path(output_folder: Path | str, seed) -> Path:
    return Path(output_folder) / f"louvain_cluster_seed{seed}.npy"


def _one_job(args):
    """One (resolution, seed) job using global _CONN."""
    global _CONN
    out_path, resolution, seed = args

    out_path = Path(out_path)
    if out_path.exists():
        return

    n = _CONN.shape[0]
    a = an.AnnData(X=np.zeros((n, 0), dtype=np.float32))
    a.obsp["connectivities"] = _CONN
    a.uns["neighbors"] = {"connectivities_key": "connectivities"}
    print(f"Running Louvain clustering with seed {seed} and resolution {resolution}...")
    sc.tl.louvain(a, resolution=resolution, random_state=seed, key_added="tmp")
    labels = a.obs["tmp"].astype(int).to_numpy()
    np.save(out_path, labels)
    print(f"Finished Louvain clustering with seed {seed} and resolution {resolution}. Saved to {out_path}.")


def parallel_cluster_for_graph(output_folder: str | Path, conn_path: str | Path, num_cluster_seeds: int, resolution:float, n_jobs: int = 8):       

    output_folder = Path(output_folder)
    jobs = []
    for seed in range(num_cluster_seeds):
        out_path = get_cluster_path(output_folder, seed)
        jobs.append((str(out_path), float(resolution), int(seed)))

    # Use spawn on Windows for stability
    print(f"Running {len(jobs)} clusterings in parallel with {n_jobs} jobs...")
    ctx = mp.get_context("spawn")
    print(f"Using multiprocessing with {n_jobs} jobs...")
    with ctx.Pool(processes=n_jobs, initializer=_init_worker, initargs=(str(conn_path),)) as pool:
        list(pool.imap_unordered(_one_job, jobs, chunksize=10))

def main():
    # We will now cluster the data with many different random seeds.
    # To speed up the clustering, we will run multiple clusterings in parallel with different random seeds.
    resolution = 1.25           # Resolution for community detection
    num_cluster_seeds = 100     # Number of trials (repeated clusterings with different randomized initializations)

    # Create a directory where we will save clustering results for different seeds.
    output_dir = Path('Preprocessed')
    cluster_dir  = output_dir / 'clusters'
    cluster_dir.mkdir(exist_ok=True)

    # Load our preprocessed anndata
    adata_preprocessed_path = Path('Preprocessed/adata.h5ad')
    ad = sc.read_h5ad(adata_preprocessed_path)

    conn_path = cluster_dir / "connectivities.npz"
    sp.save_npz(conn_path, ad.obsp["connectivities"])

    del ad
    # Run the clustering
    parallel_cluster_for_graph(cluster_dir, conn_path, num_cluster_seeds, resolution, n_jobs=8)



if __name__ == "__main__":
    mp.freeze_support()  # optional but recommended on Windows
    main()