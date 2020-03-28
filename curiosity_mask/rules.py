from experta import *
from experta.fact import *
import schema

class Occupancy(Fact):
   level = Field(schema.Or("low", "normal", "high"), mandatory=True)

class CO2(Fact):
    level = Field(schema.Or("low", "high", "non-compliant"), mandatory=True)

class Outdoor_Temperature(Fact):
    level = Field(schema.Or("normal", "extreme"), mandatory=True)

class HVAC_Machine_Teaching(KnowledgeEngine):
    @DefFacts()
    def enivronment_definitions(self, people, co2, weather):
        """Declare threshold rules."""
        if people > 5:
            yield Occupancy(level='high')
        elif people < 1:
            yield Occupancy(level='low')
        else:
            yield Occupancy(level='normal')

        if weather > 15 or weather < 2:
            yield Outdoor_Temperature(level='extreme')
        else:
            yield Outdoor_Temperature(level='normal')

        if co2 > 900:
            yield CO2(level='high')
        elif co2 < 300:
            yield CO2(level='low')
        else:
            yield CO2(level='non-compliant')

    # Heuristic Machine Teaching Strategies
    @Rule(Occupancy(level=L('low')) & CO2(level=L('low')) | Outdoor_Temperature(level='extreme'))
    def recycle_air(self):
        
        # Close the Damper
        valid_actions1 = [1] * 21
        valid_actions2 = []

        #for action in valid_actions1:
        for index, value in enumerate(valid_actions1):
            if index > 1:
                valid_actions2.append([0] * 6)
            if index <= 1:
                valid_actions2.append([1] * 6)

        valid_actions3 = []

        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            valid_actions3.append(tmp)

        valid_actions = [valid_actions1, valid_actions2, valid_actions3]
        print('found another rule')
        self.test = valid_actions
        return valid_actions

    @Rule(Occupancy(level=L('high')) | CO2(level=L('high') | CO2(level=L('non-compliant')) | Outdoor_Temperature(level='normal')))
    def freshen_air(self):
        
        # Open the Damper
        valid_actions1 = [1] * 21
        valid_actions2 = []

        #for action in valid_actions1:
        for index, value in enumerate(valid_actions1):
            if (index < 2 or index == 5):
                valid_actions2.append([0] * 6)
            else:
                valid_actions2.append([1] * 6)

        valid_actions3 = []

        for i in range(21):
            tmp = [] 
            for j in range(6):
                tmp.append([1] * 4)
            valid_actions3.append(tmp)

        valid_actions = [valid_actions1, valid_actions2, valid_actions3]
        #print('found a rule', valid_actions)
        self.test = valid_actions
        return valid_actions

mt = HVAC_Machine_Teaching()
mt.reset(people=6, co2=400, weather=7)
mt.run()
print(mt.test)

