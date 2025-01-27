# Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import Model, GRB

# Load wind and PV data for 8760 hours
wind_solar_data = pd.read_csv('wind&solar_NM.csv')
pv_output = wind_solar_data.iloc[:, 0].values  # Assuming the column name is 'pv'
wind_output = wind_solar_data.iloc[:, 1].values  # Assuming the column name is 'wind'

# Define parameters for wind, PV, storage, electrolyzer, and methanol synthesis
wind_cost = 800  # Wind installation cost ($/KW)
pv_cost = 300  # PV installation cost ($/KW)
battery_cost = 300  # Storage installation cost ($/KW)
electrolyzer_cost = 300  # Electrolyzer installation cost ($/KW)
methanol_synthesis_cost = 5000  # Methanol synthesis installation cost ($/kg*h)

# Hydrogen storage parameters
hydrogen_storage_cost = 120 / 3.6 * 120  # Storage installation cost ($/kg)
hydrogen_storage_om = 120 * 0.03 / 3.6 * 120  # O&M cost ($/kg)
hydrogen_storage_efficiency = 0.90  # Hydrogen storage round-trip efficiency
hydrogen_storage_elec = 1.707 # kwh/kgH2
hydrogen_storage_lifetime = 25

# Operational efficiencies
electrolyzer_eff = 0.7  # Electrolyzer efficiency
battery_efficiency = 0.98  # Battery round-trip efficiency
battery_charge_penalty = 0.002  # Small penalty cost for charging ($/kWh)
battery_discharge_penalty = 0.002  # Small penalty cost for discharging ($/kWh)

electric_boiler_cost = 100  # Electric boiler installation cost ($/kW)
electric_boiler_om = electric_boiler_cost * 0.03  # Fixed O&M cost ($/KW)
electric_boiler_variable_om = 1 / 1000  # Variable O&M cost ($/kWh)
electric_boiler_efficiency = 0.99  # Electric boiler efficiency

# Operational and maintenance costs
wind_om = wind_cost * 0.03  # Wind fixed O&M cost ($/KW)
pv_om = pv_cost * 0.03  # PV fixed O&M cost ($/KW)
battery_om = battery_cost * 0.03  # Storage fixed O&M cost ($/KW)
electrolyzer_om = electrolyzer_cost * 0.03  # Electrolyzer fixed O&M cost ($/KW)
methanol_synthesis_om = methanol_synthesis_cost * 0.03   # Methanol synthesis fixed O&M cost ($/kg*h)

# Discount rate and lifetime
discount_rate = 0.07  # Discount rate
wind_lifetime = 20  # Wind lifetime (years)
pv_lifetime = 25  # PV lifetime (years)
battery_lifetime = 10  # Storage lifetime (years)
electrolyzer_lifetime = 15  # Electrolyzer lifetime (years)
methanol_synthesis_lifetime = 30  # Methanol synthesis lifetime (years)
electric_boiler_lifetime = 25  # Electric boiler lifetime (years)

# Target production requirements
target_methanol_production = 10 * 1000  # Target annual methanol production (kg)
methanol_hydrogen_ratio = 0.19  # Hydrogen requirement for 1 ton of methanol (kg H2/kg CH3OH)
methanol_co2_ratio = 1.4  # CO2 requirement for 1 ton of methanol (kg CO2/kg CH3OH)
methanol_electricity_ratio = 0.5  # Electricity requirement for 1 ton of methanol (kWh/kg CH3OH)
methanol_heat_ratio = 0.58  # Net stream requirement for 1 ton of methanol (kWh/kg CH3OH)
co2_cost = 45  # Cost of CO2 ($/ton)
target_hydrogen_for_methanol = target_methanol_production * methanol_hydrogen_ratio  # Required hydrogen for methanol synthesis (kg)

Flex = 0.2  # Electrolyzer hourly output flexibility
surplus_penalty = 0.000  # Penalty cost for surplus power ($/kWh)

# Create Gurobi model
model = Model("HydrogenMethanolOptimization")

# Set model parameters
model.setParam('TimeLimit', 1000)
#model.setParam('MIPFocus', 1)
#model.setParam('Threads', 1)
#model.setParam('MIPGap', 0.1)

# Define decision variables
wind_capacity = model.addVar(lb=0, name="wind_capacity")
pv_capacity = model.addVar(lb=0, name="pv_capacity")
battery_capacity = model.addVar(lb=0, name="battery_capacity")
electrolyzer_capacity = model.addVar(lb=0, name="electrolyzer_capacity")
methanol_synthesis_capacity = model.addVar(lb=0, name="methanol_capacity")
electric_boiler_capacity = model.addVar(lb=0, name="electric_boiler_capacity")

# Add energy balance variables for storage (aggregated)
time_steps = 8760
energy_balance = model.addVars(time_steps, lb=0, name="energy_balance")

# Add variables for battery charging and discharging (aggregated)
battery_charge = model.addVars(time_steps, lb=0, name="battery_charge")
battery_discharge = model.addVars(time_steps, lb=0, name="battery_discharge")

# Add variables for electrolyzer and methanol synthesis operation, surplus, and unmet demand
electrolyzer_power = model.addVars(time_steps, lb=0, name="electrolyzer_power")
methanol_synthesis_power = model.addVars(time_steps, lb=0, name="methanol_synthesis_power")
electric_boiler_power = model.addVars(time_steps, lb=0, name="electric_boiler_power")
surplus = model.addVars(time_steps, lb=0, name="surplus")

# Hydrogen storage variables
hydrogen_storage_capacity = model.addVar(lb=0, name="hydrogen_storage_capacity")  # Max storage capacity
hydrogen_storage_balance = model.addVars(time_steps, lb=0, name="hydrogen_storage_balance")  # Stored hydrogen at each time step
hydrogen_charge = model.addVars(time_steps, lb=0, name="hydrogen_charge")  # Hydrogen charging
hydrogen_discharge = model.addVars(time_steps, lb=0, name="hydrogen_discharge")

# Set objective function: minimize levelized methanol production cost
wind_annual_cost = wind_cost * discount_rate / (1 - (1 + discount_rate) ** -wind_lifetime) + wind_om
pv_annual_cost = pv_cost * discount_rate / (1 - (1 + discount_rate) ** -pv_lifetime) + pv_om
battery_annual_cost = battery_cost * discount_rate / (1 - (1 + discount_rate) ** -battery_lifetime) + battery_om
electrolyzer_annual_cost = electrolyzer_cost * discount_rate / (1 - (1 + discount_rate) ** -electrolyzer_lifetime) + electrolyzer_om
methanol_synthesis_annual_cost = methanol_synthesis_cost * discount_rate / (1 - (1 + discount_rate) ** -methanol_synthesis_lifetime) + methanol_synthesis_om
electric_boiler_annual_cost = electric_boiler_cost * discount_rate / (1 - (1 + discount_rate) ** -electric_boiler_lifetime) + electric_boiler_om
hydrogen_storage_annual_cost = hydrogen_storage_cost * discount_rate / (1 - (1 + discount_rate) ** -hydrogen_storage_lifetime) + hydrogen_storage_om

model.setObjective(
    (wind_annual_cost * wind_capacity 
    + pv_annual_cost * pv_capacity 
    + battery_annual_cost * battery_capacity 
    + electrolyzer_annual_cost * electrolyzer_capacity 
    + methanol_synthesis_annual_cost * methanol_synthesis_capacity
    + electric_boiler_annual_cost * electric_boiler_capacity
    + hydrogen_storage_annual_cost * hydrogen_storage_capacity
    + surplus_penalty * sum(surplus[t] for t in range(time_steps))
    + battery_charge_penalty * sum(battery_charge[t] for t in range(time_steps))
    + battery_discharge_penalty * sum(battery_discharge[t] for t in range(time_steps))
    + electric_boiler_variable_om * sum(electric_boiler_power[t] for t in range(time_steps))
    + co2_cost * target_methanol_production * methanol_co2_ratio / 1000),
    GRB.MINIMIZE
)

# Add constraints

# Add constraint for electrolyzer power change rate
for t in range(1, time_steps):
    model.addConstr(electrolyzer_power[t] <= (1 + Flex) * electrolyzer_power[t-1], name=f"electrolyzer_power_increase_limit_{t}")
    model.addConstr(electrolyzer_power[t] >= (1 - Flex) * electrolyzer_power[t-1], name=f"electrolyzer_power_decrease_limit_{t}")

    model.addConstr(methanol_synthesis_power[t]+1 <= (1+Flex) * (methanol_synthesis_power[t-1]+1), name=f"methanol_power_increase_limit_{t}")
    model.addConstr(methanol_synthesis_power[t]+1 >= (1-Flex) * (methanol_synthesis_power[t-1]+1), name=f"methanol_power_decrease_limit_{t}")
    
for t in range(time_steps):
    wind_power = wind_output[t] * wind_capacity
    pv_power = pv_output[t] * pv_capacity

    # Energy storage constraint
    model.addConstr(energy_balance[t] <= battery_capacity, name=f"energy_balance_limit_{t}")
    model.addConstr(hydrogen_storage_balance[t] <= hydrogen_storage_capacity, name=f"hydrogen_storage_capacity_limit_{t}")

    # Battery charging and discharging combined should be less than total installed capacity
    model.addConstr(battery_charge[t] + battery_discharge[t] <= battery_capacity, name=f"battery_charge_discharge_limit_{t}")

    # Power balance constraint
    model.addConstr(
        wind_power + pv_power + battery_discharge[t] == electrolyzer_power[t] + 
        methanol_synthesis_power[t] * methanol_electricity_ratio + battery_charge[t] + hydrogen_charge[t] * hydrogen_storage_elec
        + electric_boiler_power[t] + surplus[t],
        name=f"power_balance_{t}"
    )

    # Energy storage constraint
    if t == 0:
        model.addConstr(
            energy_balance[t] == battery_charge[t] * battery_efficiency - battery_discharge[t] / battery_efficiency,
            name=f"energy_balance_{t}"
        )
    else:
        model.addConstr(
            energy_balance[t] == energy_balance[t-1] + battery_charge[t] * battery_efficiency - battery_discharge[t] / battery_efficiency,
            name=f"energy_balance_{t}"
        )

    if t == 0:
        model.addConstr(
            hydrogen_storage_balance[t] == hydrogen_charge[t] * hydrogen_storage_efficiency - hydrogen_discharge[t],
            name=f"hydrogen_balance_{t}"
        )
    else:
        model.addConstr(
            hydrogen_storage_balance[t] == hydrogen_storage_balance[t-1] + hydrogen_charge[t] * hydrogen_storage_efficiency - hydrogen_discharge[t],
            name=f"hydrogen_balance_{t}"
        )

    # Methanol synthesis output constraints
    model.addConstr(methanol_synthesis_power[t] <= methanol_synthesis_capacity, name=f"methanol_synthesis_max_output_{t}")
    model.addConstr(methanol_synthesis_power[t] >= 0.2 * methanol_synthesis_capacity, name=f"methanol_synthesis_min_output_{t}")
    
    # Hydrogen supply-demand balance
    model.addConstr(
        electrolyzer_power[t] * electrolyzer_eff * 3.6 / 120 + hydrogen_discharge[t] == methanol_synthesis_power[t] * methanol_hydrogen_ratio + hydrogen_charge[t],
        name=f"hydrogen_PD_balance_{t}"
    )

    # Heat supply-demand balance
    model.addConstr(
        electric_boiler_power[t] * electric_boiler_efficiency == methanol_synthesis_power[t] * methanol_heat_ratio,
        name=f"heat_PD_balance_{t}"
    )

model.addConstrs((electrolyzer_capacity >= electrolyzer_power[t] for t in range(time_steps)), name="electrolyzer_capacity_max_constraint") 
model.addConstrs((electric_boiler_capacity >= electric_boiler_power[t] for t in range(time_steps)), name="electric_boiler_capacity_max_constraint") 
    
# Methanol production constraint
annual_methanol_production = sum(methanol_synthesis_power[t] for t in range(time_steps))
model.addConstr(annual_methanol_production >= target_methanol_production, name="annual_methanol_production_constraint")

# Solve model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    methanol_production_value = annual_methanol_production.getValue()
    hydrogen_production_kwh = target_hydrogen_for_methanol * 33.33  # Convert kg to kWh
    electrolyzer_utilization_hours = hydrogen_production_kwh / (electrolyzer_capacity.x * electrolyzer_eff * 8760)*100
    methanol_production_kg = target_methanol_production
    methanol_utilization_hours = methanol_production_kg / (methanol_synthesis_capacity.x * 8760)*100

    print("Optimal system configuration:")
    print(f"Wind capacity: {wind_capacity.x:.2f} kW")
    print(f"PV capacity: {pv_capacity.x:.2f} kW")
    print(f"Battery capacity: {battery_capacity.x:.2f} kW")
    print(f"Electrolyzer capacity: {electrolyzer_capacity.x:.2f} kW")
    print(f"Electric boiler capacity: {electric_boiler_capacity.x:.2f} kW")
    print(f"Methanol synthesis capacity: {methanol_synthesis_capacity.x * 8760 / 1000:.2f} t*y")
    print(f"Methanol production: {methanol_production_value:.2f} kg")
    print(f"Hydrogen storage capacity: {hydrogen_storage_capacity.x / 3.6 * 120:.2f} kW")
    print(f"Electrolyzer utilization hours: {electrolyzer_utilization_hours:.2f} %")
    print(f"Methanol utilization hours: {methanol_utilization_hours:.2f} %")
    print(f"Levelized methanol production cost: ${model.objVal / methanol_production_value * 1000:.2f} / ton")

else:
    print("No feasible solution found.")

# Calculate power output profiles
wind_power_profile = wind_output * wind_capacity.x
pv_power_profile = pv_output * pv_capacity.x
electrolyzer_demand = [electrolyzer_power[t].x for t in range(time_steps)]
electric_boiler_demand = [electric_boiler_power[t].x for t in range(time_steps)]
methanol_demand = [methanol_synthesis_power[t].x for t in range(time_steps)]
hydrogen_storage = [hydrogen_storage_balance[t].x for t in range(time_steps)]

# Plot a sample week (168 hours)
week_start = 0 + 168 * 10  # Example: start from hour 0 (first week of the year)
week_end = 168 + 168 * 10  # 168 hours for a week

battery_charge_profile_week = [battery_charge[t].x for t in range(week_start, week_end)]
battery_discharge_profile_week = [battery_discharge[t].x for t in range(week_start, week_end)]
energy_balance_profile_week = [energy_balance[t].x for t in range(week_start, week_end)]
wind_power_profile_week = wind_power_profile[week_start:week_end]
pv_power_profile_week = pv_power_profile[week_start:week_end]
electrolyzer_demand_week = [electrolyzer_demand[t] for t in range(week_start, week_end)]
electric_boiler_demand_week = [electric_boiler_demand[t] for t in range(week_start, week_end)]
methanol_demand_week = [methanol_demand[t] for t in range(week_start, week_end)]
hydrogen_storage_week = [hydrogen_storage[t] for t in range(week_start, week_end)]

plt.figure(figsize=(15, 8))
plt.plot(wind_power_profile_week, label='Wind Output (KW)', linestyle='-', color='blue')
plt.plot(pv_power_profile_week, label='PV Output (KW)', linestyle='-', color='orange')
plt.plot(battery_charge_profile_week, label='Battery Charge (KW)', linestyle='--', color='green')
plt.plot(battery_discharge_profile_week, label='Battery Discharge (KW)', linestyle='--', color='red')
plt.plot(electrolyzer_demand_week, label='Electrolyzer Demand (KW)', linestyle='-', color='purple')
plt.plot(electric_boiler_demand_week, label='Electric Boiler Demand (KW)', linestyle='-', color='brown')
plt.plot(methanol_demand_week, label='Methanol Demand (kg)', linestyle='-', color='yellow')
plt.plot(hydrogen_storage_week, label='Hydrogen Storage Balance (kg)', linestyle='-', color='cyan')

plt.xlabel('Hour')
plt.ylabel('Power (KW)')
plt.title('Power Output and Demand Profiles (Sample Week)')
plt.legend()
plt.grid(True)
plt.show()

# Plot results for each month
hours_in_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

for i in range(12):
    start = hours_in_month[i]
    end = hours_in_month[i + 1]

    battery_charge_profile = [battery_charge[t].x for t in range(start, end)]
    battery_discharge_profile = [battery_discharge[t].x for t in range(start, end)]
    energy_balance_profile = [energy_balance[t].x for t in range(start, end)]

    plt.figure(figsize=(15, 8))
    plt.plot(wind_power_profile[start:end], label='Wind Output (KW)', linestyle='-', color='blue')
    plt.plot(pv_power_profile[start:end], label='PV Output (KW)', linestyle='-', color='orange')
    plt.plot(battery_charge_profile, label='Battery Charge (KW)', linestyle='--', color='green')
    plt.plot(battery_discharge_profile, label='Battery Discharge (KW)', linestyle='--', color='red')
    plt.plot(electrolyzer_demand[start:end], label='Electrolyzer Demand (KW)', linestyle='-', color='purple')
    plt.plot(electric_boiler_demand[start:end], label='Electric boiler Demand (KW)', linestyle='-', color='brown')
    plt.plot(methanol_demand[start:end], label='Methanol Demand (kg)', linestyle='-', color='yellow')
    plt.plot(hydrogen_storage[start:end], label='Hydrogen Storage Balance (kg)', linestyle='-', color='cyan')

    plt.xlabel('Hour')
    plt.ylabel('Power (KW)')
    plt.title(f'Power Output and Demand Profiles ({month_names[i]})')
    plt.legend()
    plt.grid(True)
    plt.show()
