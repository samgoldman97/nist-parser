""" reformat_nist_sdf.py

Reformat the NIST sdf file

"""

from pathlib import Path
import re
import argparse
import pandas as pd
import numpy as np
from typing import Iterator, List, Tuple
from itertools import groupby
from functools import partial
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm
from pathos import multiprocessing as mp

NAME_STRING = r"<(.*)>"
COLLISION_REGEX = "([0-9]+)"
VALID_ELS = set(["C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B",
                 "Se", "Fe", "Co", "As", "Na", "K"])
ION_MAP = {'[M+H-H2O]+': '[M-H2O+H]+',
           '[M+NH4]+': '[M+H3N+H]+',
           '[M+H-2H2O]+': '[M-H4O2+H]+',}


def get_els(form):
    return {i[0] for i in re.findall("([A-Z][a-z]*)([0-9]*)", form)}


def chunked_parallel(input_list, function, chunks=100, max_cpu=16):
    """chunked_parallel

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
    """

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    list_outputs = list(tqdm(pool.imap(batch_func, chunked_list), total=num_chunks))
    full_output = [j for i in list_outputs for j in i]
    return full_output


def build_mgf_str(
    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=("PEPMASS", "parentmass", "PRECURSOR_MZ"),
    precision=4,
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec in tqdm(meta_spec_list):
        str_rows = ["BEGIN IONS"]

        # Try to add precusor mass
        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            mz_to_inten = {}
            for i, j in spec_ar:
                i = np.round(i, precision)
                mz_to_inten[i] = mz_to_inten.get(i, 0) + j

            spec_ar = [[i, j] for i,j in mz_to_inten.items()]
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])


        else:
            raise NotImplementedError()
        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def process_sdf(line: Iterator):
    """process_sdf.

    Args:
        line (Iterator): line
    """
    entry_iterator = groupby(line, key=lambda x: x.startswith(">"))
    mol_block = "".join(next(entry_iterator)[1])


    output_dict = {}
    for new_field, field in entry_iterator:
        name = "".join(field).strip()
        data = "".join(next(entry_iterator)[1]).strip()

        name = re.findall(NAME_STRING, name)[0]

        # Process
        if name == "MASS SPECTRAL PEAKS":
            peaks = []
            for i in data.split("\n"):
                # Only include first two entries
                split_entry = i.split('"')[0]
                peak_tuple = [float(j) for j in split_entry.split()[:2]]
                peaks.append(peak_tuple)
            output_dict["Peaks"] = peaks
        elif name == "INCHIKEY":
            output_dict[name] = data.strip()
        elif name == "FORMULA":
            # Should line up, but computing our way to be sure
            output_dict[name] = data.strip()
        elif name == "SYNONYMS":
            output_dict[name] = data.split("\n")[0].strip()
        elif name == "NISTNO":
            output_dict[name] = data
            output_dict["spec_id"] = f"nist_{data}"
        else:
            output_dict[name] = data

    # Apply filter before converting
    if fails_filter(output_dict):
        return {}

    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None or mol.GetNumAtoms() == 0:
        return {}

    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smi)
    inchikey = Chem.MolToInchiKey(mol)
    formula = uncharged_formula(mol, mol_type="mol")
    output_dict['FORMULA'] = formula
    output_dict['INCHIKEY'] = inchikey
    output_dict['smiles'] = smi
    return output_dict


def merge_data(collision_dict: dict):
    base_dict = None
    out_peaks = {}
    num_peaks = 0
    energies = []
    for energy, sub_dict in collision_dict.items():
        if base_dict is None:
            base_dict = sub_dict
        if energy in out_peaks:
            print("Unexpected to see {energy} in {json.dumps(sub_dict, indent=2)}")
            raise ValueError()
        out_peaks[energy] = np.array(sub_dict["Peaks"])
        energies.append(energy)
        num_peaks += len(out_peaks[energy])

    base_dict["Peaks"] = out_peaks
    base_dict["COLLISION ENERGY"] = energies
    base_dict["NUM PEAKS"] = num_peaks

    peak_list = list(base_dict.pop("Peaks").items())
    info_dict = base_dict
    return (info_dict, peak_list)


def dump_to_file(entry: tuple, out_folder) -> dict:
    # Create output entry
    entry, peaks = entry
    output_name = entry["spec_id"]
    common_name = entry.get("SYNONYMS", "")
    formula = entry["FORMULA"]
    ionization = entry["PRECURSOR TYPE"]
    parent_mass = entry["PRECURSOR M/Z"]
    out_entry = {
        "dataset": "nist2020",
        "spec": output_name,
        "name": common_name,
        "formula": formula,
        "ionization": ionization,
        "smiles": entry["smiles"],
        "inchikey": entry["INCHIKEY"],
    }

    # create_output_file
    # All keys to exclude from the comments
    exclude_comments = {"Peaks"}

    output_name = Path(out_folder) / f"{output_name}.ms"
    header_str = [
        f">compound {common_name}",
        f">formula {formula}",
        f">ionization {ionization}",
        f">parentmass {parent_mass}",
    ]
    header_str = "\n".join(header_str)
    comment_str = "\n".join(
        [f"#{k} {v}" for k, v in entry.items() if k not in exclude_comments]
    )

    # Maps collision energy to peak set
    peak_list = []
    for k, v in peaks:
        peak_entry = []
        peak_entry.append(f">collision {k}")
        peak_entry.extend([f"{row[0]} {row[1]}" for row in v])
        peak_list.append("\n".join(peak_entry))

    peak_str = "\n\n".join(peak_list)
    with open(output_name, "w") as fp:
        fp.write(header_str + "\n")
        fp.write(comment_str + "\n\n")
        fp.write(peak_str)
    return out_entry


def read_sdf(input_file, debug=False):
    key_func = lambda x: "$$" in x

    lines_to_process = []
    print("Reading in file")
    debug_entries = 100000
    with open(input_file, "r") as fp:
        for index, (is_true, line) in tqdm(enumerate(groupby(fp, key=key_func))):
            if is_true:
                pass
                # print("skipping: " +  "\n".join(list(line)) + "\n\n")
            else:
                lines_to_process.append(list(line))
            if debug and index > debug_entries:
                break
    return lines_to_process


def fails_filter(entry, valid_adduct=("[M+H]+", "[M+Na]+", "[M+K]+",
                                      "[M+H-H2O]+", "[M+NH4]+",
                                      "[M+H-2H2O]+",),
                 max_mass=1500,
                 ):
    """ fails_filter. """
    if entry['PRECURSOR TYPE'] not in valid_adduct:
        return True

    if float(entry['EXACT MASS']) > max_mass:
        return True

    # QTOF, HCD,
    if entry['INSTRUMENT TYPE'].upper() != "HCD":
        return True

    form_els = get_els(entry['FORMULA'])
    if len(form_els.intersection(VALID_ELS)) != len(form_els):
        return True


    return False
        

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--input-file", action="store",
                        default="../hr_msms_nist.SDF")
    parser.add_argument("--workers", action="store",
                        default=32)
    parser.add_argument("--targ-dir", action="store",
                        default="../processed_data/")
    args = parser.parse_args()
    debug = args.debug
    workers = args.workers

    target_directory = args.targ_dir
    input_file = args.input_file

    if debug:
        target_directory = Path(target_directory) / "debug"

    target_directory = Path(target_directory)

    target_directory.mkdir(exist_ok=True, parents=True)
    target_ms = target_directory / "spec_files"
    target_mgf = target_directory / "mgf_files"
    target_labels = target_directory / "labels.tsv"

    target_ms.mkdir(exist_ok=True, parents=True)
    target_mgf.mkdir(exist_ok=True, parents=True)

    lines_to_process = read_sdf(input_file, debug=debug)

    print("Parallelizing smiles processing")
    process_sdf_temp = partial(process_sdf)
    if debug:
        output_dicts = [process_sdf_temp(i) for i in tqdm(lines_to_process)]
    else:
        output_dicts = chunked_parallel(lines_to_process, process_sdf_temp,
                                        1000, max_cpu=workers) 

    # Reformat output dicts
    # {inchikey: {adduct : {instrumnet : {collision energy : spectra} }  }}
    parsed_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    )


    print("Shuffling dict before merge")
    spec_types = []
    for output_dict in tqdm(output_dicts):
        if len(output_dict) == 0:
            continue
        inchikey = output_dict["INCHIKEY"]
        precusor_type = output_dict["PRECURSOR TYPE"]
        instrument_type = output_dict["INSTRUMENT TYPE"]
        collision_energy = output_dict["COLLISION ENERGY"]
        spec_type = output_dict["SPECTRUM TYPE"]
        spec_types.append(spec_type)
        col_energies = re.findall(COLLISION_REGEX, collision_energy)
        if len(col_energies) == 0:
            print(f"Skipping entry {output_dict} due to no col energy")
            continue
        collision_energy = col_energies[-1]
        parsed_data[inchikey][precusor_type][instrument_type][
            collision_energy
        ] = output_dict

    # merge entries
    merged_entries = []
    print("Merging dicts")
    for inchikey, adduct_dict in tqdm(parsed_data.items()):
        for adduct, instrument_dict in adduct_dict.items():
            for instrument, collision_dict in instrument_dict.items():
                output_dict = merge_data(collision_dict)
                merged_entries.append(output_dict)

    print(f"Parallelizing export to file")
    dump_fn = partial(dump_to_file, out_folder=target_ms)
    if debug:
        output_entries = [dump_fn(i) for i in merged_entries]
    else:
        output_entries = chunked_parallel(merged_entries, dump_fn, 10000,
                                          max_cpu=workers)

    mgf_out = build_mgf_str(merged_entries)
    open(target_mgf / "nist_all.mgf", "w").write(mgf_out)

    df = pd.DataFrame(output_entries)

    # Transform ions
    df['ionization'] = [ION_MAP.get(i, i) for i in df['ionization'].values]
    df.to_csv(target_labels, sep="\t", index=False)
