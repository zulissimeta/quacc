"""Common workflows for phonons."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from monty.dev import requires

from quacc import job, subflow
from quacc.atoms.phonons import (
    get_atoms_supercell_by_phonopy,
    get_phonopy,
    phonopy_atoms_to_ase_atoms,
)
from quacc.runners.phonons import PhonopyRunner
from quacc.schemas.phonons import summarize_phonopy
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.job_patterns import (
    map_partition,
    partition,
    unpartition,
    map_partitioned_lists,
)

has_phonopy = bool(find_spec("phonopy"))
has_seekpath = bool(find_spec("seekpath"))

if TYPE_CHECKING:
    from typing import Any

    from quacc import Job
    from quacc.types import PhononSchema

    if has_phonopy:
        from phonopy import Phonopy


@requires(has_phonopy, "Phonopy must be installed. Run `pip install quacc[phonons]`")
@requires(has_seekpath, "Seekpath must be installed. Run `pip install quacc[phonons]`")
def phonon_subflow(
    atoms: Atoms,
    force_job: Job,
    fixed_atom_indices: list[int] | None = None,
    symprec: float = 1e-4,
    min_lengths: float | tuple[float, float, float] | None = 20.0,
    supercell_matrix: (
        tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None
    ) = None,
    displacement: float = 0.01,
    t_step: float = 10,
    t_min: float = 0,
    t_max: float = 1000,
    phonopy_kwargs: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
    thermo_job_decorator_kwargs: dict[str, Any] | None = None,
    num_partitions: int = 8,
) -> PhononSchema:
    """
    Calculate phonon properties using the Phonopy package.

    Parameters
    ----------
    atoms
        Atoms object with calculator attached.
    force_job
        The static job to calculate the forces.
    fixed_atom_indices
        Indices of fixed atoms. These atoms will not be displaced
        during the phonon calculation. Useful for adsorbates on
        surfaces with weak coupling etc. Important approximation,
        use with caution.
    symprec
        Precision for symmetry detection.
    min_lengths
        Minimum length of each lattice dimension (A).
    supercell_matrix
        The supercell matrix to use. If specified, it will override any
        value specified by `min_lengths`.
    displacement
        Atomic displacement (A).
    t_step
        Temperature step (K).
    t_min
        Min temperature (K).
    t_max
        Max temperature (K).
    phonopy_kwargs
        Additional kwargs to pass to the Phonopy class.
    additional_fields
        Additional fields to add to the output schema.

    Returns
    -------
    PhononSchema
        Dictionary of results from [quacc.schemas.phonons.summarize_phonopy][]
    """

    @job(**(thermo_job_decorator_kwargs or {}))
    def get_phonopy_and_supercell(
        atoms,
        fixed_atom_indices,
        symprec,
        min_lengths,
        supercell_matrix,
        displacement,
        phonopy_kwargs,
        additional_fields,
    ):

        mask_to_fix = np.zeros(len(atoms), dtype=bool)

        if fixed_atom_indices:
            mask_to_fix[fixed_atom_indices] = True

        displaced_atoms, non_displaced_atoms = atoms[~mask_to_fix], atoms[mask_to_fix]

        phonopy = get_phonopy(
            displaced_atoms,
            min_lengths=min_lengths,
            supercell_matrix=supercell_matrix,
            symprec=symprec,
            displacement=displacement,
            phonopy_kwargs=phonopy_kwargs,
        )

        if non_displaced_atoms:
            non_displaced_atoms_supercell = get_atoms_supercell_by_phonopy(
                non_displaced_atoms, phonopy.supercell_matrix
            )
        else:
            non_displaced_atoms_supercell = Atoms()

        supercells = [
            phonopy_atoms_to_ase_atoms(s) + non_displaced_atoms_supercell
            for s in phonopy.supercells_with_displacements
        ]

        if non_displaced_atoms:
            additional_fields = recursive_dict_merge(
                additional_fields,
                {
                    "displaced_atoms": displaced_atoms,
                    "non_displaced_atoms": non_displaced_atoms,
                },
            )

        return {
            "phonopy": phonopy,
            "supercells": supercells,
            "non_displaced_atoms": non_displaced_atoms,
            "additional_fields": additional_fields,
        }

    @job(**thermo_job_decorator_kwargs)
    def _thermo_job(
        atoms: Atoms,
        phonopy: Phonopy,
        force_job_results: list[dict],
        t_step: float,
        t_min: float,
        t_max: float,
        additional_fields: dict[str, Any] | None,
        non_displaced_atoms: Atoms | None,
    ) -> PhononSchema:
        parameters = force_job_results[-1].get("parameters")
        forces = [
            output["results"]["forces"][: len(phonopy.supercell)]
            for output in force_job_results
        ]
        phonopy_results = PhonopyRunner().run_phonopy(
            phonopy,
            forces,
            symmetrize=bool(non_displaced_atoms),
            t_step=t_step,
            t_min=t_min,
            t_max=t_max,
        )

        return summarize_phonopy(
            phonopy,
            atoms,
            phonopy_results.directory,
            parameters=parameters,
            additional_fields=additional_fields,
        )

    phonopy_and_supercell = get_phonopy_and_supercell(
        atoms,
        fixed_atom_indices=fixed_atom_indices,
        min_lengths=min_lengths,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
        displacement=displacement,
        phonopy_kwargs=phonopy_kwargs,
        additional_fields=additional_fields,
    )

    force_job_results = unpartition(
        map_partitioned_lists(
            force_job,
            atoms=partition(phonopy_and_supercell["supercells"], num_partitions),
            num_partitions=num_partitions,
        )
    )

    return _thermo_job(
        atoms=atoms,
        phonopy=phonopy_and_supercell["phonopy"],
        force_job_results=force_job_results,
        t_step=t_step,
        t_min=t_min,
        t_max=t_max,
        additional_fields=phonopy_and_supercell["additional_fields"],
        non_displaced_atoms=phonopy_and_supercell["non_displaced_atoms"],
    )
