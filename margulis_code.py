import numpy as np
from dataclasses import dataclass


@dataclass
class MargulisCodeBlueprint:
    """
    Blueprint for the Margulis-240 code.
    Drop-in replacement for BBCodeBlueprint / ETHBBCode.

    Fields:
        A1, A2, A3       -- permutation index arrays for X-check rounds
        B1, B2, B3       -- permutation index arrays for Z-check rounds
        A1_T ... B3_T    -- transposes (used in the reverse rounds)
        Hx, Hz           -- parity check matrices (120 x 240)
        Lx, Lz           -- logical operators (8 x 240)
        n_half           -- half the number of physical qubits (= 120)
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
        """
        Load blueprint from a pre-computed .npz file.

        Usage:
            blueprint = MargulisCodeBlueprint.from_file('margulis_240.npz')
        """
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
