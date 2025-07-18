import os
import numpy as np
import pandas as pd
import time
import tempfile
from Bio import PDB
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

def check_clashes(structure, chain_y_id, chain_c_id):
    """Check for steric clashes between chains Y and G, ignoring hydrogen atoms."""
    try:
        chain_y = structure[0][chain_y_id]
        chain_c = structure[0][chain_c_id]
    except KeyError:
        return False  

    y_atoms = np.array([atom.coord for res in chain_y for atom in res if atom.element != "H"])
    c_atoms = np.array([atom.coord for res in chain_c for atom in res if atom.element != "H"])

    if y_atoms.size == 0 or c_atoms.size == 0:
        return False  

    tree = cKDTree(c_atoms)
    clashes = tree.query(y_atoms, distance_upper_bound=2.0)[0] < 2.0

    return np.any(clashes)

def align_and_save_combined(pdb1, pdb2, chain1_id, chain2_id):
    """Aligns two PDB structures and combines them if no clashes occur."""
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("complex1", pdb1)
    structure2 = parser.get_structure("complex2", pdb2)

    model1, model2 = structure1[0], structure2[0]

    if chain2_id not in model2:
        return "missing_chain"

    chain1 = model1[chain1_id]
    chain2 = model2[chain2_id]

    atoms1 = [atom for atom in chain1.get_atoms() if atom.element != "H"]
    atoms2 = [atom for atom in chain2.get_atoms() if atom.element != "H"]

    if len(atoms1) == 0 or len(atoms2) == 0:
        return "missing_atoms"

    min_len = min(len(atoms1), len(atoms2))
    atoms1, atoms2 = atoms1[:min_len], atoms2[:min_len]

    superimposer = PDB.Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    superimposer.apply(model2.get_atoms())

    combined_structure = PDB.Structure.Structure("aligned_complex")
    combined_model = PDB.Model.Model(0)

    for chain in model1:
        combined_model.add(chain.copy())
    for chain in model2:
        if chain.id not in combined_model:
            combined_model.add(chain.copy())

    combined_structure.add(combined_model)

    if check_clashes(combined_structure, "Y", "G"):
        return "clashes"

    temp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
    io = PDB.PDBIO()
    io.set_structure(combined_structure)
    io.save(temp_pdb)

    return temp_pdb

def calculate_lys_cys_distances(pdb_file, lys_chain="Y", cys_chain="G", cys_resnum=93):
    """Calculates distances from terminal Lysine nitrogen (NZ) to Cysteine sulfur (SG)."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("aligned_complex", pdb_file)

    try:
        cys_residue = next(res for res in structure[0][cys_chain] if res.get_resname() == "CYS" and res.get_id()[1] == cys_resnum)
        cys_atom = cys_residue["SG"]
    except (KeyError, StopIteration):
        return None  # CYS 93 not found

    distances = {}
    for res in structure[0][lys_chain]:
        if res.get_resname() == "LYS" and "NZ" in res:
            lys_atom = res["NZ"]  # Use terminal amine nitrogen
            distances[f"Lys_{res.get_id()[1]}"] = cys_atom - lys_atom

    return distances

def process_pdb(pdb_filename, reference_pdb, pdb_directory):
    """Processes a PDB file: aligns, checks clashes, and computes distances."""
    pdb2_path = os.path.join(pdb_directory, pdb_filename)

    output_pdb = align_and_save_combined(reference_pdb, pdb2_path, "B", "X")

    if output_pdb in {"clashes", "missing_chain", "missing_atoms"}:
        return {"PDB_File": pdb_filename, "Status": output_pdb}

    distances = calculate_lys_cys_distances(output_pdb)

    if distances is None:
        return {"PDB_File": pdb_filename, "Status": "CYS Not Found"}

    distances["PDB_File"] = pdb_filename
    return distances

def process_pdb_directory(reference_pdb, pdb_directory, output_file):
    """Processes all PDB files in a directory using parallel processing."""
    pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith(".pdb")]
    total_files = len(pdb_files)

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_pdb = {executor.submit(process_pdb, pdb, reference_pdb, pdb_directory): pdb for pdb in pdb_files}
        
        for i, future in enumerate(as_completed(future_to_pdb), start=1):
            results.append(future.result())

            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / i) * (total_files - i)
            print(f"Processed {i}/{total_files} PDBs. Estimated time left: {remaining_time:.2f} seconds", end="\r")

    df = pd.DataFrame(results)

    # Ensure "PDB_File" is the first column
    cols = ["PDB_File"] + [col for col in df.columns if col != "PDB_File"]
    df = df[cols]

    df.to_excel(output_file, index=False)
    print(f"\nProcessed {total_files} PDBs. Output saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    reference_pdb = "8RX0_prepared.pdb"
    pdb_directory = "modified_pdbs/"
    output_file = "modified_pdbs/Lys_distance.xlsx"
    
    process_pdb_directory(reference_pdb, pdb_directory, output_file)
