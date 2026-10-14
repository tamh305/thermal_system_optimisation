from meteostat import Point, Hourly
from datetime import datetime
import yaml
import scipy as sci
from scipy.integrate import solve_ivp, trapezoid
import numpy as np
from scipy.interpolate import interp1d



class Environment:
    """
    Represents the environmental data used in the simulation.

    Attributes:
        temperature_data (list[float]): Hourly temperature data in Celsius.
    """
    def __init__(self, position, time_start, time_end) -> None:
        location = Point(*position)  # Define the location
        # Set the time period for data retrieval
        start = datetime(*time_start)  # Start time
        end = datetime(*time_end)  # End time (24-hour period)
        # Fetch hourly temperature data
        data = Hourly(location, start, end)
        data = data.fetch()
        self.temperature_data = data['temp'].tolist()  # Extract temperature values from the data



class HeatPump:
    """
    Models a heat pump system.

    Attributes:
        name (str): Name of the heat pump.
        pump_on (bool): Current status of the heat pump (on/off).
        Uc (float): Overall heat transfer coefficient (W/m²K).
        Ac (float): Heat transfer area (m²).
        Tc (float): Fixed condenser temperature in Kelvin.
        T_threshold_off (float): Temperature to turn off the pump (K).
        T_threshold_on (float): Temperature to turn on the pump (K).
        total_energy_consumption (float): Total energy consumed by the pump.
        running_time (float): Total running time of the pump.
        cop_values (list[float]): Coefficients of performance recorded.
        heat_output_values (list[float]): Heat output values recorded.
    """
    def __init__(self, name, Uc, T_threshold_off, T_threshold_on, Ac, Tc, COP_file) -> None:
        """
        Initializes the HeatPump class.

        Args:
            name (str): Name of the heat pump.
            Uc (float): Heat transfer coefficient.
            T_threshold_off (float): Off threshold temperature (K).
            T_threshold_on (float): On threshold temperature (K).
            Ac (float): Heat transfer area.
            Tc (float): Fixed condenser temperature.
            COP_file (str): File containing COP data.

        Returns:
        None
        """
        # Initialize parameters
        self.name = name
        self.pump_on = False
        self.Uc = Uc  # Overall heat transfer coefficient (W/m²K)
        self.Ac = Ac  # Heat transfer area (m²)
        self.Tc = Tc  # Fixed condenser temperature in K
        self.T_threshold_off = T_threshold_off  # Temperature to turn off the pump (K)
        self.T_threshold_on = T_threshold_on  # Temperature to turn on the pump (K)
        self.COP_file_name = COP_file
        self.find_COP_coefficients()
        self.total_energy_consumption = 0
        self.running_time = 0
        self.cop_values = []
        self.heat_output_values = []
        self.qtransfer_values = []
 
    def find_COP_coefficients(self):
        """
        Reads the COP data file and computes coefficients for the optimization model.
        
        Returns:
            None
        """
        def cop_optimisation_model(temp, a, b):
            condenser_temp_C = self.Tc -273
            delta_t = abs(condenser_temp_C - temp)
            return a + (b / delta_t)

        with open(self.COP_file_name, 'r') as file:
            manufacturer_data = yaml.safe_load(file)
        outdoor_temp = np.array([entry['outdoor_temp_C'] for entry in manufacturer_data['heat_pump_cop_data']])
        cop_values = np.array([entry['COP_noisy'] for entry in manufacturer_data['heat_pump_cop_data']])
        popt, _ = sci.optimize.curve_fit(cop_optimisation_model, outdoor_temp, cop_values)
        self.a, self.b = popt
 
    def calculate_cop(self, T_outdoor,pump_mode = 1):
        """
        Calculates the coefficient of performance (COP) based on outdoor temperature.

        Args:
            T_outdoor (float): Outdoor temperature in Celsius.
            pump_mode (int): Pump mode (1 for on, 0 for off).

        Returns:
            float: Calculated COP.
        """
        if pump_mode == 1:
            cop = self.a + self.b / (abs(self.Tc - (T_outdoor + 273)))
        else :
            cop = np.NaN
        return cop

    def calculate_power_compressor(self,cop,T_tank,pump_mode = 3):
        """
        Calculates the power consumption of the compressor.

        Args:
            cop (float): Coefficient of performance.
            T_tank (float): Tank temperature in Kelvin.
            pump_mode (int): Mode of the pump:0-off,1-on,3-auto

        Returns:
            float: Power consumption in watts.
        """
        qHP = self.calculate_heat_transfer_rate(T_tank,pump_mode=pump_mode)
        if cop == 0 or np.isnan(cop):
            return 0
        else :
            return qHP/cop
        
    def update_pump_status(self, T_tank):
        """
        Updates the status of the heat pump based on tank temperature.

        Args:
            T_tank (float): Current tank temperature in Kelvin.
        """
        if  T_tank < self.T_threshold_on:
            self.pump_on = True
        elif T_tank > self.T_threshold_off:
            self.pump_on = False

    def calculate_heat_transfer_rate(self, T_tank, pump_mode = 3):
        """
        Calculates the heat transfer rate.

        Args:
            T_tank (float): Tank temperature in Kelvin.
            pump_mode (int): Pump operation mode.

        Returns:
            float: Heat transfer rate in watts.
        """
        if pump_mode == 3:
            self.update_pump_status(T_tank)
        if (self.pump_on and pump_mode == 3) or (pump_mode == 1):
            Qtransfer = self.Uc * self.Ac * (self.Tc - T_tank)
        else:
            Qtransfer = 0
        return Qtransfer



class Building:
    """
    Represents a building with thermal properties.

    Attributes:
        Aw (float): Wall area (m²).
        Ar (float): Roof area (m²).
        Uw (float): Wall U-value (W/m²K).
        Ur (float): Roof U-value (W/m²K).
        Tsp_K (float): Indoor setpoint temperature in Kelvin.
        name (str): Name of the building.
    """
    def __init__(self, name, Aw, Ar, Uw, Ur, Tsp_K) -> None:
        """
        Initializes the Building class.

        Args:
            name (str): Name of the building.
            Aw (float): Wall area in m².
            Ar (float): Roof area in m².
            Uw (float): Wall U-value in W/m²K.
            Ur (float): Roof U-value in W/m²K.
            Tsp_K (float): Setpoint temperature in Kelvin.

        Returns:
        None
        """
        self.Aw = Aw # Wall area in m²
        self.Uw = Uw  # Wall U-value in W/m²K
        self.Ar = Ar  # Roof area in m²
        self.Ur = Ur  # Roof U-value in W/m²K
        self.Tsp_K = Tsp_K  # Indoor setpoint temperature (K)
        self.name = name
   
    def __str__(self) -> str:
        return self.name
   
    def calculate_heat_load(self, outdoor_temp_C):
        """
        Calculates the heat load based on outdoor temperature.

        Args:
            outdoor_temp_C (float): Outdoor temperature in Celsius.

        Returns:
            float: Heat load in watts.
        """
        # Convert inputs to Kelvin
        outdoor_temp_K = outdoor_temp_C + 273.15
        
        # Calculate heat load using the Qload formula
        Qload = -(self.Aw * self.Uw * (outdoor_temp_K - self.Tsp_K) +
                  self.Ar * self.Ur * (outdoor_temp_K - self.Tsp_K))
        return Qload



class WaterTank:
    """
    Represents a hot water tank.

    Attributes:
        name (str): Name of the water tank.
        T_tank (float): Initial tank temperature in Kelvin.
        Utank (float): Heat loss coefficient (W/K).
        m (float): Mass of water in the tank (kg).
        Ct (float): Total thermal capacity (J/K).
        Atank (float): Surface area of the tank (m²).
    """
    def __init__(self, name, T_initial, Utank, m, Cp) -> None:
        """
        Initializes a WaterTank object.

        Args:
            name (str): Name of the water tank.
            T_initial (float): Initial temperature of the tank (K).
            Utank (float): Heat loss coefficient (W/K).
            m (float): Mass of water in the tank (kg).
            Cp (float): Specific heat capacity (J/kg·K).

        Returns:
        None
        """
        self.name = name
        self.T_tank = T_initial  # Initial tank temperature (K)
        self.Utank = Utank  # Heat loss coefficient of the tank (W/m²K)
        self.m = m  # Mass of water in tank (kg)
        self.Ct = m * Cp  # Total thermal capacity (J/K)
        self.calculate_area()

    def calculate_heat_loss(self, T_ambient_C, T_tank):
        """
        Calculates heat loss from the tank to the surroundings.

        Args:
            T_ambient_C (float): Ambient temperature in Celsius.
            T_tank (float): Current tank temperature in Kelvin.

        Returns:
            float: Heat loss in watts.
        """
        temp_ambient_K = T_ambient_C + 273
        Qloss = self.Utank * self.Atank * (T_tank - temp_ambient_K)
        return Qloss

    def calculate_area(self):
        """
        Calculates the surface area of the tank
        based on a ratio of radius to height

        Returns:
            None
        """
        # Density of water (kg/m³)
        density_water = 1000  # kg/m³
        # Calculate volume of water (in m³)
        volume = self.m / density_water
        # Assume height to radius ratio (h = 3r)
        height_radius_ratio = 3
        r = (volume / (height_radius_ratio * np.pi)) ** (1 / 3)
        h = height_radius_ratio * r
        # Calculate the surface area of the cylindrical tank (m²)
        self.Atank = 2 * np.pi * r * h + 2 * np.pi * r ** 2
   
    def __str__(self) -> str:
        return self.name



class Simulation :
    """
    Simulates the operation of a heat pump system.

    Attributes:
        env (Environment): Environmental data.
        building (Building): Building configuration.
        heat_pump (HeatPump): Heat pump configuration.
        water_tank (WaterTank): Water tank configuration.
    """
    def __init__(self, building, heat_pump, tank, coords) -> None:
        """
        Initializes the Simulation with system components and environment.

        Args:
            building (Building): Building object.
            heat_pump (HeatPump): Heat pump object.
            tank (WaterTank): Water tank object.
            coords (tuple): Geographic coordinates for the simulation.

        Returns:
        None
        """
        self.env = Environment(coords, (2023, 1, 1, 0), (2023, 1, 2, 0))
        self.building = building
        self.heat_pump = heat_pump
        self.water_tank = tank

        # Initialize performance metric tracking
        self.total_heat_output = 0
        self.total_energy_input = 0
        self.max_thermal_output = 0
        self.system_efficiency = 0
        self.total_energy_consumption = 0
        self.average_cop = 0

    def heatpump_model(self,t,T_tank):
        """
        Calculates the rate of change of tank temperature and logs pump state.
        
        Args:
            t (float): Current time value to solve at.
            T_tank (float): Current tank temperature in kelvin.

        Returns:
            float: Rate of change of tank temperature.
        """
        time_index = min(int(t // 3600), len(self.env.temperature_data) - 1)  # Get the hour
        T_outdoor = self.env.temperature_data[time_index]                     # Outdoor temp in C

        # Calculate heat load from the building
        Qload = self.building.calculate_heat_load(T_outdoor)

        # Calculate heat loss from the water tank
        Qloss = self.water_tank.calculate_heat_loss(T_outdoor,T_tank)
        
        # Calculate heat transfer rate from the heat pump based on current tank temperature
        # This determines how much heat energy is being added to the tank
        # Returns heat transfer rate in Watts (W)
        Qtransfer = self.heat_pump.calculate_heat_transfer_rate(T_tank)
        
        # Log the pump's operational state at current time step
        # Appends a tuple of (time, pump_state) where:
        #   - time (t) is in seconds
        #   - pump_state is 1 if pump is running, 0 if pump is off
        # This log is used later for visualization and performance analysis
        self.pump_log.append((t,1 if self.heat_pump.pump_on else 0))

        # Calculate the rate of change of tank temperature using thermal capacity (Ct), tank dynamics ODE
        dTtank_dt = (Qtransfer - Qload - Qloss) / self.water_tank.Ct
        return dTtank_dt   

    def calculate_metrics(self):
        """
        Calculates key performance metrics for the system.
        
        Returns:
            None
        """
        # Calculate maximum thermal output (in kW)
        self.max_thermal_output = max(self.heat_pump.qtransfer_values) / 1000
        
        # Calculate total energy consumption (in kWh) using trapezoid for integration of power values
        total_energy_joules = trapezoid(self.power_vals, self.time_points)
        self.total_energy_consumption = total_energy_joules / 3600000  # Convert J to kWh

        # Calculate total heat output using trapezoid for integration of Qtransfer values
        total_heat_output = trapezoid(self.heat_pump.qtransfer_values, self.time_points)

        # Calculate system efficiency as percentage
        self.system_efficiency = (total_heat_output / (total_energy_joules + 1e-10)) * 100

        # Calculate average COP for positive COP values
        self.average_cop = np.mean([cop for cop in self.COP_vals if cop > 0 and not np.isnan(cop)])

    def simulate_system(self):
        """
        Simulates the heat pump system over a 24-hour period.
    
        This function:
        1. Sets up the simulation timeframe and discretization
        2. Solves the differential equations for tank temperature
        3. Processes temperature data and pump states
        4. Calculates system performance metrics (COP, power, heat transfer)
        5. Stores results for analysis and visualization
        
        Returns:
            None
        """
        total_time = 86400  # 24 hours in seconds
        num_points = 1000   # Number of time steps for numerical solution
        self.time_points = np.linspace(0, total_time, num_points)  # Create evenly spaced time points for the simulation
        self.pump_log = []  # Initialize empty log to track pump operation states
       
        # Clear previous simulation results
        self.heat_pump.qtransfer_values = []  # Reset heat transfer values list
       
        # Run simulation and solve the differential equations for tank temperature
        # Uses BDF (Backward Differentiation Formula) method 
        # BDF is more suitable than RK45 for this system due to rapid temperature changes
        solution = solve_ivp(
            self.heatpump_model,       # Function containing system equations
            [0, total_time],           # Time span for simulation
            [self.water_tank.T_tank],  # Initial tank temperature
            t_eval=self.time_points,   # Time points to evaluate solution
            method='BDF'               # Numerical solving method
        )
       
        # Process pump operational states from raw log data
        self.extract_pump_states()
   
        # Temperature processing
        self.Ttank_K = solution.y[0]          # Tank temperature in Kelvin
        self.Ttank_C = self.Ttank_K - 273.15  # Convert to Celsius for analysis
        
        # Initialize arrays for performance metrics
        self.COP_vals = []
        self.power_vals = []
        
        # Calculate system performance at each time step
        for i, t in enumerate(self.time_points):
            time_index = min(int(t // 3600), len(self.env.temperature_data) - 1)    # Convert simulation time to hour index for temperature data lookup
            # Calculate COP based on outdoor temperature and pump state
            cop = self.heat_pump.calculate_cop(self.env.temperature_data[time_index],pump_mode = self.pump_log[i])
            self.COP_vals.append(cop)
           
            # Calculate heat transfer rate at current time step and store it
            qtransfer = self.heat_pump.calculate_heat_transfer_rate(
                self.Ttank_K[i],
                pump_mode=self.pump_log[i]
            )
            self.heat_pump.qtransfer_values.append(qtransfer)
           
            # Calculate power consumption
            # Power is heat transfer divided by COP when pump is running
            # Set to 0 if COP is invalid (pump off or error condition)
            power = qtransfer / cop if cop > 0 and not np.isnan(cop) else 0
            power = power if power>0 else 0
            self.power_vals.append(power)

        # Calculate final performance metrics
        self.calculate_metrics()



    def extract_pump_states(self):
        """
        Processes raw pump operation log into uniformly sampled states.
    
        This function:
        1. Extracts time points and operational states from pump log
        2. Creates an interpolation function to map between recorded states
        3. Generates pump states for all simulation time points
        
        The interpolation ensures we have pump states aligned with our
        simulation time points, using nearest-neighbor interpolation to
        maintain binary (on/off) states.
        
        Returns:
            None
        """
        # Separate time points and pump states from log entries
        log_times, log_values = zip(*self.pump_log)
        
        # Create interpolation function for pump states
        # Uses 'nearest' interpolation to maintain binary states
        # assume_sorted=False allows for potentially non-monotonic time logs
        interp_func = interp1d(log_times, log_values, kind="nearest", fill_value="extrapolate", assume_sorted=False)
        
        # Generate pump states for all simulation time points
        pump_values = interp_func(self.time_points)
        self.pump_log = pump_values