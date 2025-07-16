import os
import numpy as np
import multiprocessing as mp
import pandas as pd
import shutil
from tqdm import tqdm
from Bio import PDB
from sklearn.cluster import AgglomerativeClustering

def calculate_rmsd(structure1, structure2):
    super_imposer = PDB.Superimposer()
    atoms1 = list(structure1.get_atoms())
    atoms2 = list(structure2.get_atoms())
    if len(atoms1) != len(atoms2) or not atoms1 or not atoms2:
        return 100.0  # Assign a large value for non-matching structures
    super_imposer.set_atoms(atoms1, atoms2)
    super_imposer.apply(structure2.get_atoms())
    return super_imposer.rms

def worker(i, j, pdb_files):
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure(f'struct{i}', pdb_files[i])
    structure2 = parser.get_structure(f'struct{j}', pdb_files[j])
    return (i, j, calculate_rmsd(structure1, structure2))

def calculate_pairwise_rmsd(pdb_files, num_cpus):
    num_files = len(pdb_files)
    rmsd_matrix = np.full((num_files, num_files), 100.0)  # Initialize with large values
    
    pool = mp.Pool(num_cpus)
    tasks = [(i, j, pdb_files) for i in range(num_files) for j in range(i+1, num_files)]
    results = list(tqdm(pool.starmap(worker, tasks), 
                        total=len(tasks), 
                        desc="Calculating RMSD"))
    pool.close()
    pool.join()
    
    for i, j, rmsd in results:
        rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd
    
    return rmsd_matrix

def perform_clustering(rmsd_matrix, threshold=2.0):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="precomputed", linkage="complete")
    labels = clustering.fit_predict(rmsd_matrix)
    print(f"Clusters at RMSD {threshold}: {len(set(labels))}")
    return labels

def align_and_save_pdbs(reference_pdb, pdbs, output_folder):
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    ref_structure = parser.get_structure("ref", reference_pdb)
    ref_atoms = list(ref_structure.get_atoms())
    
    for pdb in pdbs:
        structure = parser.get_structure("struct", pdb)
        atoms = list(structure.get_atoms())
        
        if len(atoms) == len(ref_atoms):
            super_imposer = PDB.Superimposer()
            super_imposer.set_atoms(ref_atoms, atoms)
            super_imposer.apply(structure.get_atoms())
        
        output_path = os.path.join(output_folder, os.path.basename(pdb))
        io.set_structure(structure)
        io.save(output_path)

def save_cluster_report(pdb_files, labels, output_folder, threshold):
    threshold_folder = os.path.join(output_folder, f"RMSD_{threshold}")
    if os.path.exists(threshold_folder):
        shutil.rmtree(threshold_folder)
    os.makedirs(threshold_folder, exist_ok=True)
    
    cluster_dict = {}
    for pdb, cluster in zip(pdb_files, labels):
        cluster_dict.setdefault(cluster, []).append(pdb)
    
    sorted_clusters = sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)
    df_data = {"Cluster Number": [], "Cluster Size": [], "Structures": []}
    
    common_folder = os.path.join(output_folder, "Common_Structures")
    os.makedirs(common_folder, exist_ok=True)
    common_rmsd_folder = os.path.join(common_folder, f"RMSD_{threshold}")
    os.makedirs(common_rmsd_folder, exist_ok=True)
    
    for cluster_num, (cluster, pdbs) in enumerate(sorted_clusters, start=1):
        df_data["Cluster Number"].append(f"Cluster {cluster_num}")
        df_data["Cluster Size"].append(len(pdbs))
        df_data["Structures"].append(", ".join(os.path.basename(pdb) for pdb in pdbs))
        
        cluster_folder = os.path.join(threshold_folder, f"Cluster_{cluster_num}")
        os.makedirs(cluster_folder, exist_ok=True)
        
        # Align and save structures
        align_and_save_pdbs(pdbs[0], pdbs, cluster_folder)
        
        # Save one representative structure in common folder for each RMSD
        shutil.copy(pdbs[0], os.path.join(common_rmsd_folder, os.path.basename(pdbs[0])))
    
    df = pd.DataFrame(df_data)
    df.to_excel(os.path.join(threshold_folder, "cluster_report.xlsx"), index=False)

def main():
    pdb_folder = ""
    output_folder = ""
    
    pdb_files = sorted([os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith(".pdb")])
    num_cpus = min(48, mp.cpu_count())
    
    print(f"Processing {len(pdb_files)} PDB files using {num_cpus} CPUs")
    rmsd_matrix = calculate_pairwise_rmsd(pdb_files, num_cpus)
    
    for threshold in range(1, 11):
        labels = perform_clustering(rmsd_matrix, threshold=threshold)
        save_cluster_report(pdb_files, labels, output_folder, threshold)
    
    print(f"Clustering completed. Results saved in {output_folder}.")

if __name__ == "__main__":
    main()
