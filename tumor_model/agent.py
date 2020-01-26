from mesa import Agent
import numpy as np
from math import sqrt

diffusion_kernel = np.array([[0.02, 0.04, 0.02],
                             [0.04, 0.76, 0.04],
                             [0.02, 0.04, 0.02]])

# transfer a random cell from possible_cells to passed cell_type
def transfer(agent, cell_type, possible_cells):
    other_agent = agent.random.choice(possible_cells)
    a = cell_type(other_agent.unique_id, other_agent.model, other_agent.C)
    pos = other_agent.pos
    agent.model.grid.remove_agent(other_agent)
    agent.model.schedule.remove(other_agent)
    agent.model.schedule.add(a)
    agent.model.grid.place_agent(a, pos)

def tumor_cells_count(model):
    count = 0
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            cell = model.grid.get_cell_list_contents([(x,y)])[0]
            if is_tumor_cell(cell):
                count += 1

    return count

def is_tumor_cell(cell):
    return isinstance(cell, ProliferativeCellAgent) or \
           isinstance(cell, QuiescentCellAgent) or \
           isinstance(cell, DamagedQuiescentCellAgent)

class CellAgent(Agent):
    def __init__(self, unique_id, model, C):
        super().__init__(unique_id, model)
        self.C = C

    def diffusion_step(self):
        kernel = np.copy(diffusion_kernel) * self.model.kde

        if self.pos[0] == 0:
            kernel[:, 1] += kernel[:, 0]
            kernel[:, 0] = 0
        if self.pos[1] == 0:
            kernel[1] += kernel[0]
            kernel[0] = 0
        if self.pos[0] == self.model.grid.height - 1:
            kernel[:, 1] += kernel[:, 2]
            kernel[:, 2] = 0
        if self.pos[1] == self.model.grid.width - 1:
            kernel[1] += kernel[2]
            kernel[2] = 0

        x_range = int(len(diffusion_kernel[0])/2)
        y_range = int(len(diffusion_kernel)/2)
        C = self.C
        for i, muls in enumerate(kernel, start=self.pos[1] - x_range):
            for j, mul in enumerate(muls, start=self.pos[0] - y_range):
                if self.model.grid.width > i >= 0 and self.model.grid.height > j >= 0:
                    if self.pos == (j, i):
                        self.C *= mul
                    else:
                        neighbor = self.model.grid.get_cell_list_contents([(j, i)])[0]
                        neighbor.C = neighbor.C + C * mul


    def step(self):
        self.diffusion_step()


class ProliferativeCellAgent(CellAgent):
    def step(self):
        super().step()

        # tumor growth
        growth_prob = self.model.proliferative_growth_rate * (1 - sqrt(tumor_cells_count(self.model)) /
                                sqrt(self.model.grid.width * self.model.grid.height))
#        growth_prob = self.model.proliferative_growth_rate

        if growth_prob > self.random.random():
            neighbors = self.model.grid.get_neighborhood(self.pos, True, include_center=False)
            non_tumor_cells = [cell for cell in self.model.grid.get_cell_list_contents(neighbors) if
                               not (is_tumor_cell(cell) or isinstance(cell,DeadCell)) ]
            if len(non_tumor_cells) > 0:
                transfer(self, ProliferativeCellAgent, non_tumor_cells)

        # become quiescent
        if self.model.proliferative_to_quiescent_rate > self.random.random():
            transfer(self, QuiescentCellAgent, [self])

        # become eliminated - divide by (1 - proliferative_to_quiescent_rate) because conditional probability
        elif self.model.proliferative_elimination_rate * self.C / (1 - self.model.proliferative_to_quiescent_rate) > self.random.random():
            transfer(self, CellAgent, [self])


class QuiescentCellAgent(CellAgent):
    def step(self):
        super().step()

        # become damaged
        if self.model.quiescent_to_damaged_rate * self.C > self.random.random():
            transfer(self, DamagedQuiescentCellAgent, [self])


class DamagedQuiescentCellAgent(CellAgent):
    def step(self):
        super().step()

        # become proliferative again
        if self.model.damaged_to_proliferative_rate > self.random.random():
            transfer(self, ProliferativeCellAgent, [self])

        # become eliminated - divide by (1 - damaged_to_proliferative_rate) because conditional probability
        elif self.model.damaged_elimination_rate / (1 - self.model.damaged_to_proliferative_rate) > self.random.random():
            transfer(self, CellAgent, [self])

class DeadCell(CellAgent):
    def step(self):
        super().step()

