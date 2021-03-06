from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from .agent import CellAgent, ProliferativeCellAgent, QuiescentCellAgent, tumor_cells_count
from math import sqrt

MAX_TUMOR_SIZE_MM = 100

def compute_MTD(m):
    return sqrt(tumor_cells_count(m)) * MAX_TUMOR_SIZE_MM / m.grid.width

class TumorModel(Model):
    def __init__(self, width, height,
            initial_tumor_size=2,
            first_cycle_offset=80,
            treatment_cycles=30,
            treatment_cycle_interval=4,
            param_scale=1.0,
            kde=0.02, 
            proliferative_growth_rate=0.121,
            proliferative_to_quiescent_rate=0.03,
            proliferative_elimination_rate=0.7,
            quiescent_to_damaged_rate=0.7,
            damaged_to_proliferative_rate=0.003,
            damaged_elimination_rate=0.008):
        self.initial_tumor_size = initial_tumor_size
        self.first_cycle_offset = first_cycle_offset
        self.treatment_cycles = treatment_cycles
        self.treatment_cycle_interval = treatment_cycle_interval
        self.dissolve_rate = (1 - kde)
        self.proliferative_growth_rate = proliferative_growth_rate * param_scale
        self.proliferative_elimination_rate = proliferative_elimination_rate * param_scale
        self.proliferative_to_quiescent_rate = proliferative_to_quiescent_rate * param_scale
        self.quiescent_to_damaged_rate = quiescent_to_damaged_rate * param_scale
        self.damaged_to_proliferative_rate = damaged_to_proliferative_rate * param_scale
        self.damaged_elimination_rate = damaged_elimination_rate * param_scale
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.step_number = 0
        self.tumor_cells_count = initial_tumor_size ** 2

        # Calculate tumor area coordinates
        x1 = self.grid.width // 2 - self.initial_tumor_size // 2
        x2 = x1 + self.initial_tumor_size - 1
        y1 = self.grid.height // 2 - self.initial_tumor_size // 2
        y2 = y1 + self.initial_tumor_size - 1

        # Create agents
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                # uncomment to place a proliferative cell on the center
                if x in range(x1, x2 + 1) and y in range(y1, y2 + 1):
                    if x == x1 or x == x2 or y == y1 or y == y2:
                        a = ProliferativeCellAgent(y * width + x, self, 0)
                    else:
                        a = QuiescentCellAgent(y * width + x, self, 0)
                else:
                    a = CellAgent(y * width + x, self, 0)
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"C": "C"},
            model_reporters={"MTD": compute_MTD})

    def step(self):
        if self.step_number >= self.first_cycle_offset and \
                self.treatment_cycles > 0 and \
                (self.step_number - self.first_cycle_offset) % \
                self.treatment_cycle_interval == 0:
            self.treatment_cycles -= 1
            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    cell = self.grid.get_cell_list_contents([(x,y)])[0]
                    if cell.pos[0] == 0 or \
                       cell.pos[1] == 0 or \
                       cell.pos[0] == self.grid.width - 1 or \
                       cell.pos[1] == self.grid.height - 1:
                            cell.C = 1
        self.tumor_cells_count = tumor_cells_count(self)
        self.datacollector.collect(self)
        self.schedule.step()
        self.step_number += 1

