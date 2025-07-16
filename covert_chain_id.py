import os
from Bio import PDB

def modify_pdb(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    chain_x, chain_y = None, None

    # Identify the chain containing Arg 69
    for chain in structure[0]:
        for residue in chain:
            if residue.get_resname() == "ARG" and residue.get_id()[1] == 69:
                chain_x = chain
                break

    # Assign chain names
    for chain in structure[0]:
        if chain == chain_x:
            chain.id = "X"
        else:
            chain.id = "Y"

    # Remove ACE and NME residues
    for chain in structure[0]:
        residues_to_remove = [residue for residue in chain if residue.get_resname() in ("ACE", "NME")]
        for residue in residues_to_remove:
            chain.detach_child(residue.id)

    # Save the modified PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

def process_pdb_folder(input_folder):
    output_folder = os.path.join(os.path.dirname(input_folder), "modified_pdbs")
    os.makedirs(output_folder, exist_ok=True)

    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]

    for pdb_file in pdb_files:
        input_pdb = os.path.join(input_folder, pdb_file)
        output_pdb = os.path.join(output_folder, pdb_file)  # Save with same name in output folder

        modify_pdb(input_pdb, output_pdb)
        print(f"Processed: {pdb_file} → {output_pdb}")

# Example usage
input_folder = "" # Change this to your folder location
process_pdb_folder(input_folder)

