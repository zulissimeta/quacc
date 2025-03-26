from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule
from fairchem.data.oc.core.adsorbate import Adsorbate
from fairchem.data.oc.core.bulk import Bulk
from fairchem.data.oc.core.multi_adsorbate_slab_config import (
    MultipleAdsorbateSlabConfig,
)
from fairchem.data.oc.core.slab import Slab, tile_and_tag_atoms
from fairchem.data.oc.utils import DetectTrajAnomaly

from quacc import Job, flow, job
from quacc.utils.dicts import recursive_dict_merge
from quacc.wflow_tools.customizers import strip_decorator

if TYPE_CHECKING:
    from quacc.types import (
        AdsorbMLSchema,
        AtomicReferenceEnergies,
        MoleculeReferenceResults,
        OptSchema,
        RunSchema,
    )

logger = logging.getLogger(__name__)





@job
def ocp_surface_generator(bulk_atoms: Atoms, max_miller: int = 1) -> list[Slab]:
    """
    Generate surface slabs from bulk atoms.

    Parameters
    ----------
    bulk_atoms : Atoms
        The bulk atomic structure.
    max_miller : int, optional
        Maximum Miller index, by default 1.

    Returns
    -------
    list[Slab]
        List of generated surface slabs.
    """
    return Slab.from_bulk_get_all_slabs(Bulk(bulk_atoms), max_miller)


class CustomSlab(Slab):
    # Custom slab object to ignore the composition of the bulk in the initialization
    # and make sure that the user did the right things to have the surface tagged for
    # adsorbate placement

    def __init__(
        self,
        atoms: Atoms,
        bulk: Atoms | None = None,
        millers: tuple | None = None,
        shift: float | None = None,
        top: bool | None = None,
        min_ab: float = 0.8,
    ):
        """
        Initialize a CustomSlab object.

        Parameters
        ----------
        atoms : Atoms
            The slab atomic structure.
        bulk : Atoms, optional
            The bulk atomic structure, by default None.
        millers : tuple, optional
            Miller indices, by default None.
        shift : float, optional
            Shift value, by default None.
        top : bool, optional
            Top value, by default None.
        min_ab : float, optional
            Minimum a and b lattice parameters, by default 0.8.
        """
        self.bulk = bulk
        self.atoms = atoms
        self.millers = millers
        self.shift = shift
        self.top = top

        assert np.linalg.norm(self.atoms.cell[0]) >= min_ab, f"Slab not tiled, you need to repeat it to at least {min_ab}"
        assert np.linalg.norm(self.atoms.cell[1]) >= min_ab, f"Slab not tiled, you need to repeat it to at least {min_ab}"
        assert self.has_surface_tagged(), "Slab not tagged"
        assert len(self.atoms.constraints) > 0, "Sub-surface atoms not constrained"


@job
def ocp_adslab_generator(
    slab: Slab | Atoms,
    adsorbates_kwargs: list[dict[str,Any]] | None = None,
    multiple_adsorbate_slab_config_kwargs: dict[str,Any] | None = None,
) -> list[Atoms]:
    """
    Generate adsorbate-slab configurations.

    Parameters
    ----------
    slab : Slab | Atoms
        The slab structure.
    adsorbates_kwargs : list[dict[str,Any]], optional
        List of keyword arguments for generating adsorbates, by default None.
    multiple_adsorbate_slab_config_kwargs : dict[str,Any], optional
        Keyword arguments for generating multiple adsorbate-slab configurations, by default None.

    Returns
    -------
    list[Atoms]
        List of generated adsorbate-slab configurations.
    """
    adsorbates = [
        Adsorbate(**adsorbate_kwargs) for adsorbate_kwargs in adsorbates_kwargs
    ]

    if isinstance(slab, Atoms):
        try:
            slab = CustomSlab(atoms=slab)
        except AssertionError:
            slab = CustomSlab(atoms=tile_and_tag_atoms(slab))
            logger.warning(
                "The slab was not tagged and/or tiled. "
                "We did the best we could, but you should be careful and check the results!"
            )

    if multiple_adsorbate_slab_config_kwargs is None:
        multiple_adsorbate_slab_config_kwargs = {}

    adslabs = MultipleAdsorbateSlabConfig(
        copy.deepcopy(slab), adsorbates, **multiple_adsorbate_slab_config_kwargs
    )

    atoms_list = adslabs.atoms_list
    for atoms in atoms_list:
        atoms.pbc = True

    return adslabs.atoms_list


@flow
def find_adslabs_each_slab(
    slabs: list[Slab],
    adsorbates_kwargs: dict[str,Any],
    multiple_adsorbate_slab_config_kwargs: dict[str,Any] | None = None,
) -> list[dict[str, Slab | list[Atoms]]]:
    """
    Find adsorbate-slab configurations for each slab.

    Parameters
    ----------
    slabs : list[Slab]
        List of slabs.
    adsorbates_kwargs : AdsorbatesKwargs
        Keyword arguments for generating adsorbates.
    multiple_adsorbate_slab_config_kwargs : dict[str,Any], optional
        Keyword arguments for generating multiple adsorbate-slab configurations, by default None.

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries containing slabs and their corresponding adsorbate-slab configurations.
    """
    return [
        {
            "slab": slab,
            "adslabs": ocp_adslab_generator(
                slab, adsorbates_kwargs, multiple_adsorbate_slab_config_kwargs
            ),
        }
        for slab in slabs
    ]


def detect_anomaly(
    initial_atoms: Atoms, final_atoms: Atoms
) -> list[
    Literal[
        "adsorbate_dissociated",
        "adsorbate_desorbed",
        "surface_changed",
        "adsorbate_intercalated",
    ]
]:
    """
    Detect anomalies between initial and final atomic structures.

    Parameters
    ----------
    initial_atoms : Atoms
        Initial atomic structure.
    final_atoms : Atoms
        Final atomic structure.

    Returns
    -------
    list[Literal["adsorbate_dissociated", "adsorbate_desorbed", "surface_changed", "adsorbate_intercalated"]]
        List of detected anomalies.
    """
    atom_tags = initial_atoms.get_tags()

    detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
    anomalies = []
    if detector.is_adsorbate_dissociated():
        anomalies.append("adsorbate_dissociated")
    if detector.is_adsorbate_desorbed():
        anomalies.append("adsorbate_desorbed")
    if detector.has_surface_changed():
        anomalies.append("surface_changed")
    if detector.is_adsorbate_intercalated():
        anomalies.append("adsorbate_intercalated")
    return anomalies


@job
def filter_sort_select_adslabs(
    adslab_results: list[OptSchema], adslab_anomalies_list: list[list[str]]
) -> list[OptSchema]:
    """
    Filter, sort, and select adsorbate-slab configurations based on anomalies and energy.

    Parameters
    ----------
    adslab_results : list[OptSchema]
        List of adsorbate-slab results.
    adslab_anomalies_list : list[list[str]]
        List of detected anomalies for each adsorbate-slab configuration.

    Returns
    -------
    list[OptSchema]
        Sorted list of adsorbate-slab configurations without anomalies.
    """
    for adslab_result, adslab_anomalies in zip(
        adslab_results, adslab_anomalies_list, strict=True
    ):
        adslab_result["results"]["adslab_anomalies"] = adslab_anomalies

    adslabs_no_anomalies = [
        adslab_result
        for adslab_result in adslab_results
        if len(adslab_result["results"]["adslab_anomalies"]) == 0
    ]

    return sorted(adslabs_no_anomalies, key=lambda x: x["results"]["energy"])


@flow
def adsorb_ml_pipeline(
    slab: Slab,
    adsorbates_kwargs: dict[str,Any],
    multiple_adsorbate_slab_config_kwargs: dict[str, Any],
    ml_slab_adslab_relax_job: Job,
    slab_validate_job: Job,
    adslab_validate_job: Job,
    gas_validate_job: Job,
    num_to_validate_with_DFT: int = 0,
    reference_ml_energies_to_gas_phase: bool = False,
    molecule_results: MoleculeReferenceResults | None = None,
    atomic_reference_energies: AtomicReferenceEnergies | None = None,
) -> AdsorbMLSchema:
    """
    Run a machine learning-based pipeline for adsorbate-slab systems.

    1. Relax slab using ML
    2. Generate trial adsorbate-slab configurations for the relaxed slab
    3. Relax adsorbate-slab configurations using ML
    4. Validate slab and adsorbate-slab configurations (check for anomalies like dissociations))
    5. Reference the energies to gas phase if needed (eg using a total energy ML model)
    6. Optionally validate top K configurations with DFT single-points or relaxations

    Parameters
    ----------
    slab : Slab
        The slab structure to which adsorbates will be added.
    adsorbates_kwargs : dict[str,Any]
        Keyword arguments for generating adsorbate configurations.
    multiple_adsorbate_slab_config_kwargs : dict[str, Any]
        Keyword arguments for generating multiple adsorbate-slab configurations.
    ml_slab_adslab_relax_job : Job
        Job for relaxing slab and adsorbate-slab configurations using ML.
    slab_validate_job : Job
        Job for validating the slab structure.
    adslab_validate_job : Job
        Job for validating the adsorbate-slab structures.
    gas_validate_job : Job
        Job for validating gas phase structures.
    num_to_validate_with_DFT : int, optional
        Number of top configurations to validate with DFT, by default 0.
    reference_ml_energies_to_gas_phase : bool, optional
        Whether to reference ML energies to gas phase, by default False.
    molecule_results : MoleculeReferenceResults, optional
        Precomputed molecule results for referencing, by default None.
    atomic_reference_energies : AtomicReferenceEnergies, optional
        Atomic reference energies for referencing, by default None.

    Returns
    -------
    dict
        Dictionary containing the slab, ML-relaxed adsorbate-slab configurations,
        detected anomalies, and optionally DFT-validated structures.
    """

    slab.atoms.pbc=True
    ml_relaxed_slab_result = ml_slab_adslab_relax_job(slab.atoms)

    unrelaxed_adslab_configurations = ocp_adslab_generator(
        ml_relaxed_slab_result["atoms"], adsorbates_kwargs, multiple_adsorbate_slab_config_kwargs
    )

    ml_relaxed_configurations = [
        ml_slab_adslab_relax_job(adslab_configuration)
        for adslab_configuration in unrelaxed_adslab_configurations
    ]

    if reference_ml_energies_to_gas_phase:
        if atomic_reference_energies is None and molecule_results is None:
            molecule_results = generate_molecule_reference_results(
                ml_slab_adslab_relax_job
            )

        ml_relaxed_configurations = reference_adslab_energies(
            ml_relaxed_configurations,
            ml_relaxed_slab_result,
            atomic_energies=atomic_reference_energies,
            molecule_results=molecule_results,
        )

    adslab_anomalies_list = [
        job(detect_anomaly)(
            relaxed_result["input_atoms"]["atoms"], relaxed_result["atoms"]
        )
        for relaxed_result in ml_relaxed_configurations
    ]

    top_candidates = filter_sort_select_adslabs(
        adslab_results=ml_relaxed_configurations,
        adslab_anomalies_list=adslab_anomalies_list,
    )

    if num_to_validate_with_DFT == 0:
        return {
            "slab": slab.get_metadata_dict(),
            "adslabs": top_candidates,
            "adslab_anomalies": adslab_anomalies_list,
        }
    else:
        dft_validated_adslabs = [
                    adslab_validate_job(top_candidates[i]["atoms"], relax_cell=False)
                    for i in range(num_to_validate_with_DFT)
                ]

        dft_validated_slab = slab_validate_job(slab.atoms, relax_cell=False)

        if reference_ml_energies_to_gas_phase:
            if atomic_reference_energies is None and molecule_results is None:
                molecule_results = generate_molecule_reference_results(
                    gas_validate_job
                )

            dft_validated_adslabs = reference_adslab_energies(
                dft_validated_adslabs,
                dft_validated_slab,
                atomic_energies=atomic_reference_energies,
                molecule_results=molecule_results,
            )

        return {
            "slab": slab.get_metadata_dict(),
            "adslabs": top_candidates,
            "adslab_anomalies": adslab_anomalies_list,
            "validated_structures": {"slab": dft_validated_slab, "adslabs": dft_validated_adslabs}}



@job
def reference_adslab_energies(
    adslab_results: list[OptSchema],
    slab_result: RunSchema,
    atomic_energies: AtomicReferenceEnergies | None,
    molecule_results: MoleculeReferenceResults | None,
) -> list[OptSchema]:
    """
    Reference adsorbate-slab energies to atomic and slab energies.

    Parameters
    ----------
    adslab_results : list[dict[str, Any]]
        List of adsorbate-slab results.
    slab_result : RunSchema
        Result of the slab calculation.
    atomic_energies : AtomicReferenceEnergies | None
        Dictionary of atomic energies.
    molecule_results : MoleculeReferenceResults | None
        Dictionary of molecule results.

    Returns
    -------
    list[dict[str, Any]]
        List of adsorbate-slab results with referenced energies.
    """
    adslab_results = copy.deepcopy(adslab_results)
    if atomic_energies is None:
        if molecule_results is not None:
            atomic_energies = {
            "H": molecule_results["H2"]["results"]["energy"] / 2,
            "N": molecule_results["N2"]["results"]["energy"] / 2,
            "O": (
                molecule_results["H2O"]["results"]["energy"]
                - molecule_results["H2"]["results"]["energy"]
            ),
            "C": molecule_results["CO"]["results"]["energy"]
            - (
                molecule_results["H2O"]["results"]["energy"]
                - molecule_results["H2"]["results"]["energy"]
            ),
        }
        else:
            raise Exception(
            "Missing atomic energies and gas phase energies; unable to continue!"
        )


    slab_energy = slab_result["results"]["energy"]

    return [
        recursive_dict_merge(
            adslab_result,
            {
                "results": {
                    "referenced_adsorption_energy": {
                        "atomic_energies": atomic_energies,
                        "slab_energy": slab_energy,
                        "adslab_energy": adslab_result["results"]["energy"],
                        "gas_reactant_energy": sum(
                            [
                                atomic_energies[atom.symbol]
                                for atom in adslab_result["atoms"][
                                    adslab_result["atoms"].get_tags() == 2
                                ]  # all adsorbate tagged with tag=2!
                            ]
                        ),
                        "adsorption_energy": adslab_result["results"]["energy"]
                        - slab_energy
                        - sum(
                            [
                                atomic_energies[atom.symbol]
                                for atom in adslab_result["atoms"][
                                    adslab_result["atoms"].get_tags() == 2
                                ]  # all adsorbate tagged with tag=2!
                            ]
                        ),
                    }
                }
            },
        )
        for adslab_result in adslab_results
    ]


def molecule_pbc(*args: Any, **molecule_kwargs: Any) -> Atoms:
    """
    Create a molecule with periodic boundary conditions.

    Parameters
    ----------
    *args : Any
        Positional arguments for the molecule function.
    **molecule_kwargs : Any
        Keyword arguments for the molecule function.

    Returns
    -------
    Atoms
        Atomic structure with periodic boundary conditions.
    """
    atoms = molecule(*args, **molecule_kwargs)
    atoms.pbc = True
    return atoms


def generate_molecule_reference_results(relax_job: Job) -> MoleculeReferenceResults:
    """
    Generate reference results for molecules.

    Parameters
    ----------
    relax_job : Job
        Job for relaxing molecular structures.

    Returns
    -------
    MoleculeReferenceResults
        Dictionary of reference results for molecules.
    """
    return {
        "N2": relax_job(molecule_pbc("N2", vacuum=10), relax_cell=False),
        "CO": relax_job(molecule_pbc("CO", vacuum=10), relax_cell=False),
        "H2": relax_job(molecule_pbc("H2", vacuum=10), relax_cell=False),
        "H2O": relax_job(molecule_pbc("H2O", vacuum=10), relax_cell=False),
    }


@flow
def bulk_to_surfaces_to_adsorbml(
    bulk_atoms: Atoms,
    adsorbates_kwargs: dict[str,Any],
    multiple_adsorbate_slab_config_kwargs: dict[str, Any],
    ml_relax_job: Job,
    slab_validate_job: Job,
    adslab_validate_job: Job,
    gas_validate_job: Job,
    max_miller: int = 1,
    bulk_relax_job: Job | None = None,

    num_to_validate_with_DFT: int = 0,
    reference_ml_energies_to_gas_phase: bool = True,
    relax_bulk: bool = True,
    atomic_reference_energies: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Run a pipeline from bulk atoms to adsorbate-slab configurations using machine learning!
    For full details, see the AdsorbML paper (https://arxiv.org/abs/2211.16486,
                                     https://www.nature.com/articles/s41524-023-01121-5).

    1. Relax bulk structure if desired
    2. Generate surface slabs from bulk atoms
    3. Generate gas phase reference energies if needed

    For each slab generated in (3):
        1. Relax slab using ML
        2. Generate trial adsorbate-slab configurations for the relaxed slab
        3. Relax adsorbate-slab configurations using ML
        4. Validate slab and adsorbate-slab configurations (check for anomalies like dissociations))
        5. Reference the energies to gas phase if needed (eg using a total energy ML model)
        6. Optionally validate top K configurations with DFT single-points or relaxations

    Parameters
    ----------
    bulk_atoms : Atoms
        The bulk atomic structure.
    adsorbates_kwargs : AdsorbatesKwargs
        Keyword arguments for generating adsorbate configurations.
    multiple_adsorbate_slab_config_kwargs : dict[str, Any]
        Keyword arguments for generating multiple adsorbate-slab configurations.
    ml_relax_job : Job
        Job for relaxing slab and adsorbate-slab configurations using ML.
    slab_validate_job : Job
        Job for validating the slab structure.
    adslab_validate_job : Job
        Job for validating the adsorbate-slab structures.
    gas_validate_job : Job
        Job for validating gas phase structures.
    max_miller : int, optional
        Maximum Miller index, by default 1.
    bulk_relax_job : Job | None, optional
        Job for relaxing the bulk structure, by default None.
    num_to_validate_with_DFT : int, optional
        Number of top configurations to validate with DFT, by default 0.
    reference_ml_energies_to_gas_phase : bool, optional
        Whether to reference ML energies to gas phase, by default True.
    relax_bulk : bool, optional
        Whether to relax the bulk structure, by default True.
    atomic_reference_energies : dict[str, float] | None, optional
        Atomic reference energies for referencing if known ahead of time, by default None.

    Returns
    -------
    list[AdsorbMLSchema]
        List of AdsorbML results for each slab
    """

    if relax_bulk:
        bulk_atoms = bulk_relax_job(bulk_atoms, relax_cell=True)["atoms"]

    slabs = ocp_surface_generator(bulk_atoms=bulk_atoms, max_miller=max_miller)

    if reference_ml_energies_to_gas_phase and atomic_reference_energies is not None:
        molecule_results = generate_molecule_reference_results(
            ml_relax_job
        )
    else:
        molecule_results = None

    @flow
    def adsorbML_each_surface(slabs: list[Slab], **kwargs: Any) -> list[dict[str, Any]]:
        return [
            # We strip the decorator here so it's a bunch of jobs, not subflows. Helpful for prefect!
            strip_decorator(adsorb_ml_pipeline)(slab=slab, **kwargs)
            for slab in slabs
        ]

    return adsorbML_each_surface(
        slabs=slabs,
        adsorbates_kwargs=adsorbates_kwargs,
        multiple_adsorbate_slab_config_kwargs=multiple_adsorbate_slab_config_kwargs,
        ml_slab_adslab_relax_job=ml_relax_job,
        slab_validate_job=slab_validate_job,
        adslab_validate_job=adslab_validate_job,
        gas_validate_job=gas_validate_job,
        num_to_validate_with_DFT=num_to_validate_with_DFT,
        molecule_results=molecule_results,
        reference_ml_energies_to_gas_phase=reference_ml_energies_to_gas_phase,
        atomic_reference_energies=atomic_reference_energies
    )
