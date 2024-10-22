import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import Model, GRB


# Load wind and PV data for 8760 hours
wind_solar_data = pd.read_csv('wind&solar.csv')
pv_output = wind_solar_data.iloc[:, 0].values  # Assuming the column name is 'pv'
wind_output = wind_solar_data.iloc[:, 1].values  # Assuming the column name is 'wind'

# Define parameters for wind, PV, storage, and electrolyzer
wind_cost = 1000  # Wind installation cost ($/KW)
pv_cost = 600  # PV installation cost ($/KW)
battery_cost = 300  # Storage installation cost ($/KW)
electrolyzer_cost = 600  # Electrolyzer installation cost ($/KW)

electrolyzer_eff = 0.7  # Electrolyzer efficiency
battery_efficiency = 0.9  # Battery round-trip efficiency
battery_charge_penalty = 0.001  # Small penalty cost for charging ($/kWh)
battery_discharge_penalty = 0.001  # Small penalty cost for discharging ($/kWh)

wind_om = 20  # Wind fixed O&M cost ($/KW)
pv_om = 15  # PV fixed O&M cost ($/KW)
battery_om = 10  # Storage fixed O&M cost ($/KW)
electrolyzer_om = 25  # Electrolyzer fixed O&M cost ($/KW)

# Discount rate and lifetime
discount_rate = 0.07  # Discount rate
wind_lifetime = 20  # Wind lifetime (years)
pv_lifetime = 25  # PV lifetime (years)
battery_lifetime = 10  # Storage lifetime (years)
electrolyzer_lifetime = 15  # Electrolyzer lifetime (years)

electrolyzer_capacity = 1.0  # Electrolyzer capacity fixed at 1 KW
electrolyzer_demand_curve = np.array([1.0] * 8760)  # Electrolyzer 8760-hour demand curve (KW), fixed at 1 KW
unmet_demand_penalty = 0.07  # Penalty cost for unmet demand ($/kWh)
surplus_penalty = 0.0001  # Penalty cost for surplus power ($/kWh)

# Create Gurobi model
model = Model("HydrogenProductionOptimization")

# Set model parameters to handle large models
model.setParam('TimeLimit', 600)  # Set a time limit of 600 seconds to avoid long runtimes
model.setParam('MIPFocus', 1)  # Focus on finding a feasible solution
model.setParam('Threads', 2)  # Limit the number of threads to 2 to reduce resource usage

# Define decision variables
wind_capacity = model.addVar(lb=0, name="wind_capacity")
pv_capacity = model.addVar(lb=0, name="pv_capacity")
battery_capacity = model.addVar(lb=0, name="battery_capacity")

# Add energy balance variables for storage (aggregated to reduce model size)
time_steps = 8760
energy_balance = model.addVars(time_steps, lb=0, name="energy_balance")

# Add variables for battery charging and discharging (aggregated to reduce model size)
battery_charge = model.addVars(time_steps, lb=0, name="battery_charge")
battery_discharge = model.addVars(time_steps, lb=0, name="battery_discharge")

# Add variables for surplus and unmet demand
surplus = model.addVars(time_steps, lb=0, name="surplus")
unmet_demand = model.addVars(time_steps, lb=0, name="unmet_demand")

# Set objective function: minimize levelized hydrogen production cost
wind_annual_cost = wind_cost * discount_rate / (1 - (1 + discount_rate) ** -wind_lifetime) + wind_om
pv_annual_cost = pv_cost * discount_rate / (1 - (1 + discount_rate) ** -pv_lifetime) + pv_om
battery_annual_cost = battery_cost * discount_rate / (1 - (1 + discount_rate) ** -battery_lifetime) + battery_om
electrolyzer_annual_cost = electrolyzer_cost * discount_rate / (1 - (1 + discount_rate) ** -electrolyzer_lifetime) + electrolyzer_om

Hydrogen_production = electrolyzer_capacity * 8760 * electrolyzer_eff * 3.6 / 120

model.setObjective(
    ((wind_annual_cost * wind_capacity + pv_annual_cost * pv_capacity + battery_annual_cost * battery_capacity + electrolyzer_annual_cost * electrolyzer_capacity) +
    + unmet_demand_penalty * sum(unmet_demand[t] for t in range(time_steps))
    + surplus_penalty * sum(surplus[t] for t in range(time_steps)) / 8760
    + battery_charge_penalty * sum(battery_charge[t] for t in range(time_steps)) 
    + battery_discharge_penalty * sum(battery_discharge[t] for t in range(time_steps)))/ Hydrogen_production,
    GRB.MINIMIZE
)

# Add constraints
for t in range(time_steps):
    wind_power = wind_output[t] * wind_capacity
    pv_power = pv_output[t] * pv_capacity

    # Energy storage constraint, battery energy cannot exceed battery capacity
    model.addConstr(energy_balance[t] <= battery_capacity, name=f"energy_balance_limit_{t}")

    # Battery charging and discharging combined should be less than total installed capacity
    model.addConstr(battery_charge[t] + battery_discharge[t] <= battery_capacity, name=f"battery_charge_discharge_limit_{t}")

    # Power balance constraint (allow unmet demand or surplus)
    model.addConstr(
        wind_power + pv_power + battery_discharge[t] + unmet_demand[t] == electrolyzer_demand_curve[t] + battery_charge[t] + surplus[t],
        name=f"power_balance_{t}"
    )

    # Energy storage constraint, battery energy cannot exceed battery capacity
    #model.addConstr(unmet_demand[t] <= electrolyzer_demand_curve[t]*0.4, name=f"unmet_demand_limit_{t}")

    # Add constraint for annual unmet demand and surplus equality
    #model.addConstr(sum(unmet_demand[t] for t in range(time_steps)) <= sum(surplus[t] for t in range(time_steps)), name="annual_unmet_demand_surplus_equality")
    
    # Energy balance constraint with battery efficiency
    if t == 0:
        model.addConstr(
            energy_balance[t] == battery_charge[t] * battery_efficiency - battery_discharge[t] * battery_efficiency,
            name=f"energy_balance_{t}"
        )
    else:
        model.addConstr(
            energy_balance[t] == energy_balance[t-1] + battery_charge[t] * battery_efficiency - battery_discharge[t] * battery_efficiency,
            name=f"energy_balance_{t}"
        )

    # Add constraint for unmet demand change rate limit (cannot exceed 50%)
    if t > 0:
        model.addConstr(
            unmet_demand[t] <= 1.1 * unmet_demand[t-1],
            name=f"unmet_demand_rate_increase_{t}"
        )
        model.addConstr(
            unmet_demand[t] >= 0.9 * unmet_demand[t-1],
            name=f"unmet_demand_rate_decrease_{t}"
        )
# Solve model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal system configuration:")
    print(f"Wind capacity: {wind_capacity.x:.2f} kW")
    print(f"PV capacity: {pv_capacity.x:.2f} KW")
    print(f"Battery capacity: {battery_capacity.x:.2f} KW")
    print(f"Electrolyzer capacity: 1.00 KW")
    print(f"Levelized hydrogen production cost: ${model.objVal:.2f} / kg")

    # Calculate power output profiles
    wind_power_profile = wind_output * wind_capacity.x
    pv_power_profile = pv_output * pv_capacity.x
    electrolyzer_demand = electrolyzer_demand_curve  # Electrolyzer demand (1 MW fixed)

    # Plot results for each month
    hours_in_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    for i in range(12):
        start = hours_in_month[i]
        end = hours_in_month[i + 1]

        battery_charge_profile = [battery_charge[t].x for t in range(start, end)]
        battery_discharge_profile = [battery_discharge[t].x for t in range(start, end)]
        energy_balance_profile = [energy_balance[t].x for t in range(start, end)]
        unmet_demand_profile = [unmet_demand[t].x for t in range(start, end)]

        plt.figure(figsize=(15, 8))
        plt.plot(wind_power_profile[start:end], label='Wind Output (KW)', linestyle='-', color='blue')
        plt.plot(pv_power_profile[start:end], label='PV Output (KW)', linestyle='-', color='orange')
        plt.plot(battery_charge_profile, label='Battery Charge (KW)', linestyle='--', color='green')
        plt.plot(battery_discharge_profile, label='Battery Discharge (KW)', linestyle='--', color='red')
        plt.plot(electrolyzer_demand[start:end], label='Electrolyzer Demand (KW)', linestyle='-', color='purple')
        plt.plot(unmet_demand_profile, label='Battery Energy Balance (kWh)', linestyle='-', color='brown')

        plt.xlabel('Hour')
        plt.ylabel('Power (KW)')
        plt.title(f'Power Output and Demand Profiles ({month_names[i]})')
        plt.legend()
        plt.grid(True)
        plt.show()

else:
    print("No feasible solution found.")
