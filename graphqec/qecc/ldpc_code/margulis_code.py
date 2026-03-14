import numpy as np
from dataclasses import dataclass
from graphqec.qecc.code import QuantumCode, TannerGraph, TemporalTannerGraph
from graphqec.qecc.ldpc_code.bbcode import ETHBBCode, build_memory_circuit
from graphqec.qecc.utils import (
    get_bipartite_indices,
    get_data_to_logical_from_pcm,
    map_bipartite_edge_indices,
)


@dataclass
class MargulisCodeBlueprint:
    """
    Blueprint for the Margulis-240 code.
    Drop-in replacement for BBCodeBlueprint / ETHBBCode.
    """
    A1: np.ndarray;  A2: np.ndarray;  A3: np.ndarray
    B1: np.ndarray;  B2: np.ndarray;  B3: np.ndarray
    A1_T: np.ndarray; A2_T: np.ndarray; A3_T: np.ndarray
    B1_T: np.ndarray; B2_T: np.ndarray; B3_T: np.ndarray
    Hz: np.ndarray;  Hx: np.ndarray
    Lz: np.ndarray;  Lx: np.ndarray
    n_half: int

    @classmethod
    def from_file(cls, path: str) -> "MargulisCodeBlueprint":
        d = np.load(path)
        return cls(
            A1=d['A1'],   A2=d['A2'],   A3=d['A3'],
            B1=d['B1'],   B2=d['B2'],   B3=d['B3'],
            A1_T=d['A1_T'], A2_T=d['A2_T'], A3_T=d['A3_T'],
            B1_T=d['B1_T'], B2_T=d['B2_T'], B3_T=d['B3_T'],
            Hz=d['Hz'],   Hx=d['Hx'],
            Lz=d['Lz'],   Lx=d['Lx'],
            n_half=int(d['n_half']),
        )


class MargulisCode(QuantumCode):
    """
    Margulis-240 code as a full QuantumCode subclass.
    Reuses ETHBBCode's tanner graph and circuit logic via the blueprint.
    """

    def __init__(self, blueprint: MargulisCodeBlueprint,
                 logical_basis='Z', check_basis='ZX'):
        self.blue_print = blueprint
        self.n_half = blueprint.n_half
        self.logical_basis = logical_basis
        self.check_basis = check_basis
        self.qX = {i: i for i in range(self.n_half)}
        self.qL = {i: self.n_half + i for i in range(self.n_half)}
        self.qR = {i: 2 * self.n_half + i for i in range(self.n_half)}
        self.qZ = {i: 3 * self.n_half + i for i in range(self.n_half)}

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "MargulisCode":
        blueprint = MargulisCodeBlueprint.from_file(path)
        return cls(blueprint, **kwargs)

    def get_tanner_graph(self) -> TemporalTannerGraph:
        return ETHBBCode.get_tanner_graph(self)

    def get_syndrome_circuit(self, num_cycle: int, *,
                             physical_error_rate: float = 0, **kwargs):
        circuit = build_memory_circuit(
            self.blue_print, physical_error_rate,
            num_cycle + 1, z_basis=True, use_both=True
        )
        return circuit.without_noise() if physical_error_rate == 0 else circuit

    def get_dem(self, num_cycle, *, physical_error_rate, **kwargs):
        assert physical_error_rate > 0
        return self.get_syndrome_circuit(
            num_cycle, physical_error_rate=physical_error_rate
        ).detector_error_model()
