from abc import ABC, abstractmethod

class Gene(ABC):
    pass

class Agent(ABC):
    pop : np.array[int] # human, horse, livestock, boat, cart
    gene : Gene
    money : float
    inventory : np.array[float] # food, cloth, tool, wood, labor, service, power

    internal_state : np.array[float] # health, hunger ... emotions ... 
    min_needs : np.array[float] # same as inventory, derived from building + pop needs
    needs : np.array[float] # same as inventory, derived from building + pop needs, but can be reduced by internal_state (e.g. hunger reduces need for labor)
    avg_cost : np.array[float] # Average cost multiplier of Agent's inventory depending on agent's location


class Market(ABC):
    base_prices : np.array[float] # food, cloth, tool, wood, labor, service, power
    
class Land(ABC):
    needs : np.array[float] # same as inventory, derived from building + pop needs