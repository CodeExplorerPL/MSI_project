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
        for _ in range( np.random.randint(0, len(self)//3) ):
            idx = np.random.randint( 0, len(self)-1 )
            self.line[idx] = np.random.uniform(*self.range)

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
        return cls([
            GeneLine((0.0, 1.0), n_inputs, "mid"),
            GeneLine((0.0, 1.0), n_inputs, "top"),
            GeneLine((0.0, 1.0), n_inputs * 2, "side"),
            GeneLine((0.0, 1.0), n_rules, "op"),
            GeneLine((-1.0, 1.0), n_tsk, "tsk")
        ])