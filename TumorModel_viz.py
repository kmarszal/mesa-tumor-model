from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector


class TumorModel(Model):
    def __init__(self, width, height, diffusion_kernel, kde):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                if i == j == 0 or i == j == 9:
                    a = CellAgent(j * width + i, self, 1)
                elif i == j == 5:
                    a = ProliferativeCellAgent(j * width + i, self, 0)
                else:
                    a = CellAgent(j * width + i, self, 0)
                self.schedule.add(a)
                self.grid.place_agent(a, (i, j))

        self.datacollector = DataCollector(
            agent_reporters={"C": "C"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


class CellAgent(Agent):
    def __init__(self, unique_id, model, C):
        super().__init__(unique_id, model)
        self.C = C

    def diffusion_step(self):
        x_range = int(len(diffusion_kernel[0])/2)
        y_range = int(len(diffusion_kernel)/2)
        for i, muls in enumerate(diffusion_kernel, start=self.pos[0] - x_range):
            for j, mul in enumerate(muls, start=self.pos[1] - y_range):
                if self.model.grid.width > i >= 0 and self.model.grid.height > j >= 0:
                    if self.pos == (i, j):
                        self.C *= mul
                    else:
                        neighbor = self.model.grid.get_cell_list_contents([(i, j)])[0]
                        neighbor.C = min(neighbor.C + self.C * mul, 1.0)

    def step(self):
        self.diffusion_step()


class ProliferativeCellAgent(CellAgent):
    def step(self):
        super().step()
        neighbors = self.model.grid.get_neighborhood(self.pos, True, include_center=False)
        non_tumor_cells = [cell for cell in self.model.grid.get_cell_list_contents(neighbors) if isinstance(cell, CellAgent)]
        if len(non_tumor_cells) > 0:
            other_agent = self.random.choice(non_tumor_cells)
            a = ProliferativeCellAgent(other_agent.unique_id, other_agent.model, other_agent.C)
            pos = other_agent.pos
            self.model.grid.remove_agent(other_agent)
            self.model.schedule.remove(other_agent)
            self.model.schedule.add(a)
            self.model.grid.place_agent(a, pos)


def agent_portrayal(agent):
    portrayal = dict(Shape="circle", Filled="true", Layer=0, r=1)
    if isinstance(agent, ProliferativeCellAgent):
        portrayal["Color"] = "#00aaaa"
    else:
        portrayal["Color"] = "#0000aa" + hex(int(agent.C * 255 / 16))[-1] + hex(int(agent.C * 255 % 16))[-1]
    return portrayal


diffusion_kernel = [[0.01, 0.02, 0.01],
                    [0.02, 1.0, 0.02],  # 0.88
                    [0.01, 0.02, 0.01]]

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
server = ModularServer(TumorModel,
                       [grid],
                       "Tumor Model",
                       {"width": 10, "height": 10, "kde": 0, "diffusion_kernel": diffusion_kernel})
server.port = 8521  # The default
server.launch()
