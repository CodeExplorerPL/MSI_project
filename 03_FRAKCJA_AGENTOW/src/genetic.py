import json
import numpy as np

class GeneLine:
    def __init__(self, value_range: tuple[float, float], n_values: int, name: str, values: list[float] = None):
        self.name = name
        self.range = value_range
        if values is None:
            self.line = np.random.uniform(*value_range, n_values).tolist()
        else:
            self.line = values
    def mutate(self) -> None:
        if len(self) == 0:
            return

        # Keep mutations local: perturb a small subset instead of random full resets.
        max_fraction = 0.02 if len(self) < 500 else 0.005
        max_mutations = max(1, int(len(self) * max_fraction))
        n_mutations = np.random.randint(1, max_mutations + 1)
        span = float(self.range[1] - self.range[0])
        sigma = 0.10 * span

        for _ in range(n_mutations):
            idx = np.random.randint(0, len(self))
            mutated = float(self.line[idx]) + float(np.random.normal(0.0, sigma))
            self.line[idx] = float(np.clip(mutated, self.range[0], self.range[1]))

    def crossover(self, parent: 'GeneLine') -> tuple['GeneLine', 'GeneLine']:
        cut_point = np.random.randint(1, len(self)-1)

        line_a = GeneLine(self.range, len(self), self.name, values=[*self.line[:cut_point], *parent.line[cut_point:]])
        line_b = GeneLine(self.range, len(self), self.name, values=[*parent.line[:cut_point], *self.line[cut_point:]])

        return line_a, line_b

    def __len__(self) -> int:
        return len(self.line)
    
    def to_dict(self):
        return {
            "name": self.name,
            "range": self.range,
            "line": self.line
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            value_range=tuple(data["range"]),
            n_values=len(data["line"]),
            name=data["name"],
            values=data["line"]
        )

class ANFIS_Specimen:
    def __init__(self, genes: list['GeneLine']):
        super().__init__()
        self.genes = genes
        self.score = 0
        self.path = "temp.json"

    def mutate(self) -> None:
        for line in self.genes:
            line.mutate()

    def create_descendants(self, partner: 'ANFIS_Specimen') -> tuple['ANFIS_Specimen', 'ANFIS_Specimen']:
        child_genes_a = []
        child_genes_b = []

        for parent_line_a, parent_line_b in zip(self.genes, partner.genes):
            child_line_a, child_line_b = parent_line_a.crossover(parent_line_b)
            child_genes_a.append(child_line_a)
            child_genes_b.append(child_line_b)

        return ANFIS_Specimen(child_genes_a), ANFIS_Specimen(child_genes_b)
    
    def flatten(self) -> np.ndarray:
        genes_map = {line.name: line for line in self.genes}
    
        # Sklejamy parametry przesłanek (muszą być 4 na każdą funkcję)
        # W Twoim systemie to mid, top i side
        premises_parts = []
        mid = genes_map['mid'].line
        top = genes_map['top'].line
        side = genes_map['side'].line
        
        for i in range(len(mid)):
            premises_parts.extend([mid[i], top[i], side[2*i], side[2*i+1]])

        # Sklejamy resztę
        op_part = genes_map['op'].line
        tsk_part = genes_map['tsk'].line
        
        return np.array(premises_parts + op_part + tsk_part)
    
    def save_to_file(self, filename: str = None):
        """Zapisuje genom osobnika do pliku JSON."""
        if filename is None:
            filename = self.path

        data = {
            "score": self.score,
            "genes": [gene.to_dict() for gene in self.genes]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_file(cls, filename: str) -> 'ANFIS_Specimen':
        """Wczytuje genom z pliku i zwraca nowy obiekt ANFIS_Specimen."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        genes = [GeneLine.from_dict(g) for g in data["genes"]]
        specimen = cls(genes)
        specimen.score = data.get("score", 0)
        specimen.path = filename
        return specimen
    
    @classmethod
    def generate_random(cls, inputs_definition: list) -> 'ANFIS_Specimen':
        n_inputs = len(inputs_definition)
        total_mfs = sum(var.n_functions for var in inputs_definition)
        n_rules = 1
        for var in inputs_definition:
            n_rules *= var.n_functions
        n_tsk = n_rules * (n_inputs + 1)

        # Seed premise genes around configured fuzzy inputs with jitter.
        base_mid = []
        base_top = []
        base_side = []
        for variable in inputs_definition:
            center, kernel, fuzzy_left, fuzzy_right = variable.get()
            base_mid.append(float(np.clip(center, 0.0, 1.0)))
            base_top.append(float(np.clip(kernel, 0.05, 1.0)))
            base_side.extend(
                [
                    float(np.clip(fuzzy_left, 0.02, 1.0)),
                    float(np.clip(fuzzy_right, 0.02, 1.0)),
                ]
            )

        def jitter(values: list[float], sigma: float, low: float, high: float) -> list[float]:
            arr = np.array(values, dtype=float)
            arr = arr + np.random.normal(0.0, sigma, len(arr))
            return np.clip(arr, low, high).tolist()

        return cls([
            GeneLine((0.0, 1.0), n_inputs, "mid", values=jitter(base_mid, sigma=0.10, low=0.0, high=1.0)),
            GeneLine((0.0, 1.0), n_inputs, "top", values=jitter(base_top, sigma=0.08, low=0.05, high=1.0)),
            GeneLine((0.0, 1.0), n_inputs * 2, "side", values=jitter(base_side, sigma=0.08, low=0.02, high=1.0)),
            GeneLine((0.0, 1.0), n_rules, "op"),
            GeneLine((-1.0, 1.0), n_tsk, "tsk")
        ])
