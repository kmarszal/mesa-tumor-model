from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from .model import TumorModel
from .agent import CellAgent, ProliferativeCellAgent, QuiescentCellAgent, DamagedQuiescentCellAgent, DeadCell

def agent_portrayal(agent):
    portrayal = dict(Shape="circle", Filled="true", Layer=0, r=1)
    if isinstance(agent, ProliferativeCellAgent):
        portrayal["Color"] = "#00aaaa"
    elif isinstance(agent, QuiescentCellAgent):
        portrayal["Color"] = "#007777"
    elif isinstance(agent, DamagedQuiescentCellAgent):
        portrayal["Color"] = "#003333"
    elif isinstance(agent, DeadCell):
        portrayal["Color"] = "#000000" 
    else:
        portrayal["Color"] = "#0000aa" + hex(int(min(agent.C, 1) * 255 / 16))[-1] + hex(int(min(agent.C, 1) * 255 % 16))[-1]
    return portrayal



height = 20
width = 20

grid = CanvasGrid(agent_portrayal, height, width, 500, 500)
chart = ChartModule([{"Label": "MTD",
                      "Color": "Black"}])

model_params = { 
    "height": height,
    "width": width,
    "kde": UserSettableParameter("slider", "KDE", 0.27 * 3 / 4, 0, 1.0, 0.01),
    "initial_tumor_size": UserSettableParameter("slider", "initial tumor size", 2, 1, width // 2, 1),
    "first_cycle_offset": UserSettableParameter("slider", "first cycle offset", 80, 1, 200, 1),
    "treatment_cycles": UserSettableParameter("slider", "treatment cycles", 30, 0,50, 1),
    "treatment_cycle_interval": UserSettableParameter("slider", "treatment cycle interval", 4, 1, 200, 1),
#    "proliferative_growth_rate": UserSettableParameter("slider",
#        "proliferative growth rate", 0.121, 0, 1.0, 0.01),
#    "proliferative_to_quiescent_rate": UserSettableParameter("slider",
#        "proliferative to quiescent_rate", 0.03, 0, 1.0, 0.01),
#    "proliferative_elimination_rate": UserSettableParameter("slider",
#        "proliferative elimination rate", 0.7, 0, 1.0, 0.01),
#    "quiescent_to_damaged_rate": UserSettableParameter("slider",
#        "quiescent to damaged rate", 0.7, 0, 1.0, 0.01),
#    "damaged_to_proliferative_rate": UserSettableParameter("slider",
#        "damaged to proliferative rate", 0.003, 0, 0.1, 0.001),
#    "damaged_elimination_rate": UserSettableParameter("slider",
#        "damaged elimination rate", 0.008, 0, 0.1, 0.001)
}


server = ModularServer(TumorModel,
                       [grid, chart],
                       "Tumor Model",
                       model_params)
server.port = 8521  # The default
