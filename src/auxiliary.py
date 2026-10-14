import numpy as np
from scipy.integrate import trapezoid, solve_ivp
from scipy.optimize import curve_fit
import yaml
from src.model import Environment
import matplotlib.pyplot as plt

class HeatPump:
    def __init__(self, name, Uc, T_threshold_on, T_threshold_off, Ac, Tc, COP_file, max_power=5000) -> None:
        self.name = name
        self.pump_on = False
        self.Uc = Uc  # Overall heat transfer coefficient (W/m²K)
        self.Ac = Ac  # Heat transfer area (m²)
        self.Tc = Tc  # Fixed condenser temperature (K)
        self.T_threshold_on = T_threshold_on
        self.T_threshold_off = T_threshold_off
        self.COP_file_name = COP_file
        self.max_power = max_power  # Maximum power output in Watts
        self.find_COP_coefficients()
        self.qtransfer_values = []

    def find_COP_coefficients(self):
        def cop_optimisation_model(temp, a, b):
            condenser_temp_C = self.Tc - 273
            delta_t = abs(condenser_temp_C - temp)
            return a + (b / delta_t)

        with open(self.COP_file_name, 'r') as file:
            manufacturer_data = yaml.safe_load(file)
        outdoor_temp = np.array([entry['outdoor_temp_C'] for entry in manufacturer_data['heat_pump_cop_data']])
        cop_values = np.array([entry['COP_noisy'] for entry in manufacturer_data['heat_pump_cop_data']])
        popt, _ = curve_fit(cop_optimisation_model, outdoor_temp, cop_values)
        self.a, self.b = popt

    def calculate_cop(self, T_outdoor, pump_mode=1):
        if pump_mode == 1:
            cop = self.a + self.b / (abs(self.Tc - (T_outdoor + 273)))
        else:
            cop = np.NaN
        return cop

    def update_pump_status(self, T_tank):
        if T_tank < self.T_threshold_on:
            self.pump_on = True
        elif T_tank > self.T_threshold_off:
            self.pump_on = False

    def calculate_heat_transfer_rate(self, T_tank, pump_mode=3):
        if pump_mode == 3:
            self.update_pump_status(T_tank)
        if self.pump_on or pump_mode == 1:
            Qtransfer = self.Uc * self.Ac * (self.Tc - T_tank)
            Qtransfer = min(Qtransfer, self.max_power)
        else:
            Qtransfer = 0
        return Qtransfer


class WaterTank:
    def __init__(self, name, T_initial, Utank, m, Cp) -> None:
        self.name = name
        self.T_tank = T_initial
        self.Utank = Utank  # Heat loss coefficient of the tank (W/K)
        self.m = m
        self.Ct = m * Cp  # Total thermal capacity (J/K)
        self.calculate_area()

    def calculate_heat_loss(self, T_ambient_C, T_tank):
        temp_ambient_K = T_ambient_C + 273
        Qloss = self.Utank * self.Atank * (T_tank - temp_ambient_K)
        return Qloss

    def calculate_area(self):
        density_water = 1000
        volume = self.m / density_water
        height_radius_ratio = 3
        r = (volume / (height_radius_ratio * np.pi)) ** (1 / 3)
        h = height_radius_ratio * r
        self.Atank = 2 * np.pi * r * h + 2 * np.pi * r ** 2

    def __str__(self) -> str:
        return self.name


class Building:
    def __init__(self, name, Aw, Ar, Uw, Ur, Tsp_K) -> None:
        self.name = name
        self.Aw = Aw  # Wall area (m²)
        self.Ar = Ar  # Roof area (m²)
        self.Uw = Uw  # Wall heat transfer coefficient (W/m²K)
        self.Ur = Ur  # Roof heat transfer coefficient (W/m²K)
        self.Tsp_K = Tsp_K  # Setpoint temperature (K)

    def calculate_heat_load(self, T_outdoor_C):
        T_outdoor_K = T_outdoor_C + 273.15
        Q = -(self.Aw * self.Uw * (T_outdoor_K - self.Tsp_K) + 
             self.Ar * self.Ur * (T_outdoor_K - self.Tsp_K))
        return Q


class Simulation:
    def __init__(self, building, heat_pump, tank, coords) -> None:
        self.env = Environment(coords, (2023, 1, 1, 0), (2023, 1, 2, 0))
        self.building = building
        self.heat_pump = heat_pump
        self.water_tank = tank
        
        # Auxiliary heater configuration
        self.aux_heater_power = 3000  # 3kW auxiliary heater
        self.aux_heater_on = False
        self.Taux_on_threshold = 273.15 + 38  # 38°C (311.15K)
        self.Taux_off_threshold = 273.15 + 45  # 45°C (318.15K)
        
        # Initialize tracking
        self.aux_heater_power_vals = []
        self.heat_pump_power_vals = []
        self.pump_log = []

    def update_aux_heater_status(self, T_tank):
        if T_tank < self.Taux_on_threshold:
            self.aux_heater_on = True
        elif T_tank > self.Taux_off_threshold:
            self.aux_heater_on = False

    def heatpump_model(self, t, T_tank):
        time_index = min(int(t // 3600), len(self.env.temperature_data) - 1)
        T_outdoor = self.env.temperature_data[time_index]

        # Calculate loads and transfers
        Qload = self.building.calculate_heat_load(T_outdoor)
        Qloss = self.water_tank.calculate_heat_loss(T_outdoor, T_tank)
        Qtransfer = self.heat_pump.calculate_heat_transfer_rate(T_tank)
        
        # Update auxiliary heater
        self.update_aux_heater_status(T_tank)
        Qaux = self.aux_heater_power if self.aux_heater_on else 0

        # Total heat transfer
        total_heat = Qtransfer + Qaux
        
        # Store pump status
        self.pump_log.append(1 if self.heat_pump.pump_on else 0)
        
        return (total_heat - Qload - Qloss) / self.water_tank.Ct

    def simulate_system(self):
        total_time = 86400  # 24 hours in seconds
        num_points = 1000
        self.time_points = np.linspace(0, total_time, num_points)
        self.pump_log = []

        # Run simulation
        solution = solve_ivp(
            self.heatpump_model,
            [0, total_time],
            [self.water_tank.T_tank],
            t_eval=self.time_points,
            method='BDF'
        )

        # Process temperatures
        self.Ttank_K = solution.y[0]
        self.Ttank_C = self.Ttank_K - 273.15

        # Initialize arrays
        self.COP_vals = []
        self.heat_pump_power_vals = []
        self.aux_heater_power_vals = []
        self.heat_pump.qtransfer_values = []

        # Calculate values at each time step
        for i, t in enumerate(self.time_points):
            time_index = min(int(t // 3600), len(self.env.temperature_data) - 1)
            T_outdoor = self.env.temperature_data[time_index]
            
            # Calculate COP
            cop = self.heat_pump.calculate_cop(T_outdoor)
            self.COP_vals.append(cop)
            
            # Calculate heat transfer
            qtransfer = self.heat_pump.calculate_heat_transfer_rate(self.Ttank_K[i])
            self.heat_pump.qtransfer_values.append(qtransfer)
            
            # Calculate powers
            if cop > 0 and not np.isnan(cop):
                hp_power = qtransfer / cop
            else:
                hp_power = 0
            self.heat_pump_power_vals.append(hp_power)
            
            # Calculate auxiliary heater power
            if self.Ttank_K[i] < self.Taux_on_threshold:
                self.aux_heater_power_vals.append(self.aux_heater_power)
            else:
                self.aux_heater_power_vals.append(0)

        self.calculate_metrics()

    def calculate_metrics(self):
        time_hours = self.time_points / 3600  # Convert time to hours
        
        # Calculate heat pump energy using the trapezoidal rule
        hp_energy = trapezoid(self.heat_pump_power_vals, time_hours) / 1000  # kWh
        self.heat_pump_energy_consumption = hp_energy
        
        # Calculate auxiliary heater energy using the trapezoidal rule
        aux_energy = trapezoid(self.aux_heater_power_vals, time_hours) / 1000  # kWh
        self.aux_energy_consumption = aux_energy
        
        # Total energy consumption
        self.total_energy_consumption = hp_energy + aux_energy
        
        # Calculate average COP
        valid_cops = [cop for cop in self.COP_vals if cop > 0 and not np.isnan(cop)]
        self.average_cop = np.mean(valid_cops) if valid_cops else 0
        
        # Calculate maximum thermal output
        self.max_thermal_output = max(self.heat_pump.qtransfer_values) / 1000  # Convert to kW
       
def main():
    # Create components (heat pump, building, and tank)
    heat_pump = HeatPump(
        name="Limited Heat Pump",
        Uc=300.0,
        T_threshold_on=313.15,  # 40°C
        T_threshold_off=333.15,  # 60°C
        Ac=1.11,
        Tc=338.15,
        COP_file='data/heat_pump_cop_synthetic_full.yaml',
        max_power=2500  # Limtied Heat Pump Power (2.5KW)
    )

    building = Building("Test Building", 150.0, 140.0, 1.0, 0.5, 293.15)
    tank = WaterTank("Test Tank", 308.15, 20.0, 200, 4186)

    # Run the simulation
    sim = Simulation(building, heat_pump, tank, coords=(55.9533, -3.1883))
    sim.simulate_system()

    # Create visualization
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot tank temperature
    time_hours = sim.time_points / 3600  # Convert seconds to hours
    ax1.plot(time_hours, sim.Ttank_C, 'b-', label='Tank Temperature')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    
    # Add threshold lines
    ax1.axhline(y=40, color='g', linestyle='--', label='Heat Pump ON')
    ax1.axhline(y=60, color='r', linestyle='--', label='Heat Pump OFF')
    ax1.axhline(y=38, color='y', linestyle='--', label='Aux Heater ON')
    ax1.axhline(y=45, color='orange', linestyle='--', label='Aux Heater OFF')

    # Create a secondary y-axis for power
    ax2 = ax1.twinx()
    
    # Plot auxiliary heater operation
    aux_power_kw = [p/1000 for p in sim.aux_heater_power_vals]  # Convert to kW
    ax2.plot(time_hours, aux_power_kw, 'r-', alpha=0.3, label='Auxiliary Heater Power')
    ax2.set_ylabel('Power (kW)')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Tank Temperature and Auxiliary Heater Operation Over Time')
    plt.grid(True)
    plt.show()

    # Print results after simulation completes
    print("\nSimulation Results:")
    print("-" * 40)
    print(f"Total Energy Consumption: {sim.total_energy_consumption:.2f} kWh")
    print(f"Auxiliary Heater Energy: {sim.aux_energy_consumption:.2f} kWh")
    print(f"Heat Pump Energy: {sim.heat_pump_energy_consumption:.2f} kWh")
    print(f"Average COP: {sim.average_cop:.2f}")
    print(f"Maximum Thermal Output: {sim.max_thermal_output:.2f} kW")
    print(f"Temperature Range: {min(sim.Ttank_C):.1f}°C to {max(sim.Ttank_C):.1f}°C")
    
    # Add debugging info
    print("\nAuxiliary Heater Analysis:")
    print("-" * 40)
    aux_activations = sum(1 for p in sim.aux_heater_power_vals if p > 0)
    print(f"Number of times auxiliary heater activated: {aux_activations}")
    print(f"Percentage of time auxiliary heater was on: {(aux_activations/len(sim.aux_heater_power_vals))*100:.1f}%")

if __name__ == "__main__":
    main()