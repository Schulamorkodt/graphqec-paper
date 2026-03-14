import numpy as np
import stim
from dataclasses import dataclass
from graphqec.qecc.code import QuantumCode, TannerGraph, TemporalTannerGraph
from graphqec.qecc.ldpc_code.bbcode import ETHBBCode
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
    Uses a direct CSS circuit builder based on Hx/Hz parity check matrices.
    Code parameters: [[240, 8, ~20]]
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
                             physical_error_rate: float = 0, **kwargs) -> stim.Circuit:
        """
        Build a Z-basis CSS memory circuit directly from Hx/Hz.
        Works correctly for arbitrary CSS codes including Margulis codes.
        """
        Hx = self.blue_print.Hx
        Hz = self.blue_print.Hz
        Lz = self.blue_print.Lz
        p = physical_error_rate

        n = Hx.shape[1]
        n_cx = Hx.shape[0]
        n_cz = Hz.shape[0]
        x_anc = list(range(n, n + n_cx))
        z_anc = list(range(n + n_cx, n + n_cx + n_cz))

        circuit = stim.Circuit()

        # Initialize data qubits in Z basis
        for i in range(n):
            circuit.append('R', [i])
            if p > 0:
                circuit.append('X_ERROR', [i], p)
        circuit.append('TICK')

        for rnd in range(num_cycle):
            for q in x_anc + z_anc:
                circuit.append('R', [q])
                if p > 0:
                    circuit.append('X_ERROR', [q], p)
            circuit.append('TICK')

            circuit.append('H', x_anc)
            circuit.append('TICK')

            for i in range(n_cx):
                for d in np.where(Hx[i])[0]:
                    circuit.append('CNOT', [x_anc[i], int(d)])
                    if p > 0:
                        circuit.append('DEPOLARIZE2', [x_anc[i], int(d)], p)
            circuit.append('TICK')

            for i in range(n_cz):
                for d in np.where(Hz[i])[0]:
                    circuit.append('CNOT', [int(d), z_anc[i]])
                    if p > 0:
                        circuit.append('DEPOLARIZE2', [int(d), z_anc[i]], p)
            circuit.append('TICK')

            circuit.append('H', x_anc)
            circuit.append('TICK')

            if p > 0:
                circuit.append('X_ERROR', x_anc + z_anc, p)
            circuit.append('M', x_anc + z_anc)

            total = n_cx + n_cz

            # Z check detectors (deterministic from round 0)
            for i in range(n_cz):
                rec = stim.target_rec(-total + n_cx + i)
                if rnd == 0:
                    circuit.append('DETECTOR', [rec])
                else:
                    prev_rec = stim.target_rec(-total - total + n_cx + i)
                    circuit.append('DETECTOR', [rec, prev_rec])

            # X check detectors (only from round 1)
            if rnd > 0:
                for i in range(n_cx):
                    rec = stim.target_rec(-total + i)
                    prev_rec = stim.target_rec(-total - total + i)
                    circuit.append('DETECTOR', [rec, prev_rec])

            circuit.append('TICK')

        # Final data measurement
        if p > 0:
            circuit.append('X_ERROR', list(range(n)), p)
        circuit.append('M', list(range(n)))

        total = n_cx + n_cz

        # Final Z stabilizer detectors
        for i, row in enumerate(Hz):
            targets = [stim.target_rec(-n + int(d)) for d in np.where(row)[0]]
            targets.append(stim.target_rec(-n - total + n_cx + i))
            circuit.append('DETECTOR', targets)

        # Logical observables
        for i, row in enumerate(Lz):
            targets = [stim.target_rec(-n + int(d)) for d in np.where(row)[0]]
            circuit.append('OBSERVABLE_INCLUDE', targets, i)

        return circuit if p > 0 else circuit.without_noise()

    def get_dem(self, num_cycle, *, physical_error_rate, **kwargs):
        assert physical_error_rate > 0, "physical_error_rate must be > 0"
        return self.get_syndrome_circuit(
            num_cycle, physical_error_rate=physical_error_rate
        ).detector_error_model()
