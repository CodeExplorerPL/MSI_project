from enum import IntEnum

import numpy as np
from .ANFIS.params import FuzzyInputVariable_3Trapezoids
from .ANFIS.ANFIS import ANFIS
from .genetic import ANFIS_Specimen

class StrategyType(IntEnum):
    SEARCH = 0     # Patrolowanie, szukanie wrogów
    ATTACK = 1     # Agresywne dążenie do kontaktu i strzału
    FLEE = 2       # Ucieczka od najbliższego wroga
    SAVE = 3       # Defensywa, szukanie osłon
    POWERUP = 4    # Ignorowanie walki na rzecz wzmocnień
#    RELOAD = 5     # Wycofanie się w celu przeładowania

INPUTS_DEFINITION = [
    FuzzyInputVariable_3Trapezoids(0.55, 0.24, 0.24, 0.24, "my_hp", ["low", "mid", "high"]),
    FuzzyInputVariable_3Trapezoids(0.35, 0.20, 0.20, 0.35, "enemy_dist", ["near", "mid", "far"]),
    FuzzyInputVariable_3Trapezoids(0.30, 0.18, 0.12, 0.42, "reload_status", ["ready", "reloading", "locked"]),
    FuzzyInputVariable_3Trapezoids(0.25, 0.14, 0.20, 0.45, "terrain_risk", ["safe", "risky", "deadly"]),
    FuzzyInputVariable_3Trapezoids(0.40, 0.18, 0.22, 0.40, "enemies_left", ["few", "some", "many"]),
]

class StrategyModel:
    def __init__(self, fuzzy_inputs, output_range: tuple[float, float] = (0.0, 6.0)):
        """
        :param fuzzy_inputs: Lista gotowych obiektów FuzzyInputVariable_List_Trapezoids
        :param output_range: Zakres wartości wyjściowych (domyślnie 0-6)
        """

        dummy_x = np.zeros((len(fuzzy_inputs), 1))
        dummy_y = np.zeros(1)
        
        self.model = ANFIS(fuzzy_inputs, dummy_x, dummy_y)
        self.inputs = fuzzy_inputs
        self.output_range = output_range

        # Zapamiętujemy rozmiary struktur, aby wiedzieć jak ciąć tablicę flatten()
        self._premises_shape = np.shape(self.model.premises)
        self._op_shape = np.shape(self.model.op)
        self._tsk_shape = np.shape(self.model.tsk)
        self._split_indices = [
            np.prod(self._premises_shape), 
            np.prod(self._premises_shape) + np.prod(self._op_shape)
        ]

    def set_params_from_genes(self, specimen: ANFIS_Specimen):
        """Wstrzykuje geny osobnika do struktur modelu ANFIS"""
        p_flat, o_flat, t_flat = np.split(specimen.flatten(), self._split_indices)
        self.model.set_premises_parameters(p_flat.reshape(self._premises_shape))
        self.model.op = o_flat.reshape(self._op_shape)
        self.model.tsk = t_flat.reshape(self._tsk_shape)

    def get_result(self, inputs_vector: np.ndarray) -> float:
        """Pobiera wynik dla pojedynczego stanu gry [0, 1]"""
        self.model.training_data = inputs_vector.reshape(-1, 1)
        res = self.model.get_results()
        return float(res[0])