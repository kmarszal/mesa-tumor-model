from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector
import numpy as np


# transfer a random cell from possible_cells to passed cell_type
def transfer(agent, cell_type, possible_cells):
    other_agent = agent.random.choice(possible_cells)
    a = cell_type(other_agent.unique_id, other_agent.model, other_agent.C)
    pos = other_agent.pos
    agent.model.grid.remove_agent(other_agent)
    agent.model.schedule.remove(other_agent)
    agent.model.schedule.add(a)
    agent.model.grid.place_agent(a, pos)


class TumorModel(Model):
    def __init__(self, width, height, kde):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.step_number = 0

        # Create agents
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                # uncomment to place a proliferative cell on the center
                # if i == j == self.grid.width/2:
                #     a = ProliferativeCellAgent(j * width + i, self, 0)
                # else:
                a = CellAgent(j * width + i, self, 0)
                self.schedule.add(a)
                self.grid.place_agent(a, (i, j))

        self.datacollector = DataCollector(
            agent_reporters={"C": "C"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.step_number += 1


class CellAgent(Agent):
    def __init__(self, unique_id, model, C):
        super().__init__(unique_id, model)
        self.C = C

    def diffusion_step(self):
        if self.pos == (0, 0) or self.pos == (self.model.grid.width - 1, self.model.grid.height - 1):
            if self.model.step_number < 200:
                self.C = 1

        kernel = np.copy(diffusion_kernel)

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
        if proliferative_growth_rate > self.random.random():
            neighbors = self.model.grid.get_neighborhood(self.pos, True, include_center=False)
            non_tumor_cells = [cell for cell in self.model.grid.get_cell_list_contents(neighbors) if
                               not (isinstance(cell, ProliferativeCellAgent) or isinstance(cell, QuiescentCellAgent) or isinstance(cell, DamagedQuiescentCellAgent))]
            if len(non_tumor_cells) > 0:
                transfer(self, ProliferativeCellAgent, non_tumor_cells)

        # become quiescent
        if proliferative_to_quiescent_rate > self.random.random():
            transfer(self, QuiescentCellAgent, [self])

        # become damaged
        elif (any_to_damaged_rate * self.C) / (1 - proliferative_to_quiescent_rate) > self.random.random():
            transfer(self, DamagedQuiescentCellAgent, [self])


class QuiescentCellAgent(CellAgent):
    def step(self):
        super().step()

        # become damaged
        if any_to_damaged_rate * self.C > self.random.random():
            transfer(self, DamagedQuiescentCellAgent, [self])


class DamagedQuiescentCellAgent(CellAgent):
    def step(self):
        super().step()

        # become proliferative again
        if any_to_damaged_rate > self.random.random():
            transfer(self, ProliferativeCellAgent, [self])

        # become eliminated
        elif damaged_elimination_rate / (1 - any_to_damaged_rate) > self.random.random():
            transfer(self, CellAgent, [self])


def agent_portrayal(agent):
    portrayal = dict(Shape="circle", Filled="true", Layer=0, r=1)
    if isinstance(agent, ProliferativeCellAgent):
        portrayal["Color"] = "#00aaaa"
    elif isinstance(agent, QuiescentCellAgent):
        portrayal["Color"] = "#007777"
    elif isinstance(agent, DamagedQuiescentCellAgent):
        portrayal["Color"] = "#003333"
    else:
        portrayal["Color"] = "#0000aa" + hex(int(min(agent.C, 1) * 255 / 16))[-1] + hex(int(min(agent.C, 1) * 255 % 16))[-1]
    return portrayal


diffusion_kernel = np.array([[0.02, 0.04, 0.02],
                             [0.04, 0.76, 0.04],
                             [0.02, 0.04, 0.02]])


proliferative_growth_rate = 0.5
proliferative_to_quiescent_rate = 0.2
any_to_damaged_rate = 0.2
damaged_to_proliferative_rate = 0.1
damaged_elimination_rate = 0.1

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
server = ModularServer(TumorModel,
                       [grid],
                       "Tumor Model",
                       {"width": 10, "height": 10, "kde": 0})
server.port = 8521  # The default
server.launch()
