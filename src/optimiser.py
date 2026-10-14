from src.model import HeatPump, Building, WaterTank, Simulation
import numpy as np
from tabulate import tabulate
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class BuildingOptimizer:
    def __init__(self):
        """Initialize the BuildingOptimizer with default heat pump and location settings"""
        # Create default heat pump with standard parameters
        self.heat_pump = HeatPump(
            name="Default Heat Pump",
            Uc=300.0,  # Heat transfer coefficient
            T_threshold_off=333.15,  # Temperature to turn off pump (K)
            T_threshold_on=313.15,  # Temperature to turn on pump (K)
            Ac=1.11,  # Heat transfer area (m²)
            Tc=339,  # Condenser temperature (K)
            COP_file="data/heat_pump_cop_synthetic_full.yaml"  # File containing COP data
        )
        self.current_building = None  # Stores current building being optimized
        self.current_param_names = None  # Stores parameter names for optimization
        self.coords = (55.9533, -3.1883)  # Edinburgh coordinates for weather data

    def setup_parameter_ranges(self, building: Building) -> dict:
        """
        Define optimization ranges for building and tank parameters.
        
        Args:
            building (Building): Current building being optimized
            
        Returns:
            dict: Dictionary of parameter ranges with (min, max) tuples
        """
        return {
            'Uw': (0.1, 0.99),  # Wall U-value range (W/m²K)
            'Ur': (0.1, 0.99),  # Roof U-value range (W/m²K)
            'Tsp_K': (288, 303.15),  # Set point temperature range (K) [15°C to 30°C]
            'T_initial': (313.15, 323.15),  # Initial tank temperature range (K) [40°C to 50°C]
            'Utank': (3.0, 7.0),  # Tank heat loss coefficient range (W/K)
            'm': (150.0, 250.0)  # Tank mass range (kg)
        }

    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization that calculates total energy consumption.
        
        Args:
            x (np.ndarray): Array of parameter values to evaluate
            
        Returns:
            float: Total energy consumption or penalty value if simulation fails
        """
        params = {name: value for name, value in zip(self.current_param_names, x)}
        
        try:
            test_building = Building(
                self.current_building.name,
                self.current_building.Aw,
                self.current_building.Ar,
                params['Uw'],
                params['Ur'],
                params['Tsp_K']
            )
            
            test_tank = WaterTank(
                "Optimized Tank",
                params['T_initial'],
                params['Utank'],
                params['m'],
                4186
            )
            
            sim = Simulation(test_building, self.heat_pump, test_tank, self.coords)
            sim.simulate_system()
            
            return sim.total_energy_consumption
            
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            return 1e6

    def plot_performance_metrics(self, building_name, original_sim, optimized_sim):
        """
        Create subplot figure comparing original vs optimized performance metrics.
        
        Args:
            building_name (str): Name of the building being analyzed
            original_sim (Simulation): Original simulation results
            optimized_sim (Simulation): Optimized simulation results
        """
        # Create main figure with specified size
        fig = plt.figure(figsize=(12, 8))
        
        # Set up subplot grid layout
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])  # Top left plot
        ax2 = fig.add_subplot(gs[0, 1])  # Top right plot
        ax3 = fig.add_subplot(gs[1, :])  # Bottom full-width plot
        
        # Define metrics to compare between original and optimized systems
        metrics = {
            'Maximum Thermal Output (kW)': [original_sim.max_thermal_output, 
                                          optimized_sim.max_thermal_output],
            'Energy Consumption (kWh)': [original_sim.total_energy_consumption, 
                                       optimized_sim.total_energy_consumption],
            'System Efficiency (%)': [original_sim.system_efficiency, 
                                    optimized_sim.system_efficiency]
        }
        
        # Define colors for original and optimized bars
        colors = ['#1f77b4', '#2ca02c']  # Blue for original, green for optimized
        
        # Helper function to create consistent bar plots
        def create_bar_plot(ax, metric_name, values):
            # Create bars with specified colors
            bars = ax.bar(['Original', 'Optimized'], values, color=colors)
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name.split('(')[1].strip(')'))
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom')
        
        # Create individual plots for each metric
        create_bar_plot(ax1, 'Maximum Thermal Output (kW)', metrics['Maximum Thermal Output (kW)'])
        create_bar_plot(ax2, 'Energy Consumption (kWh)', metrics['Energy Consumption (kWh)'])
        create_bar_plot(ax3, 'System Efficiency (%)', metrics['System Efficiency (%)'])
        
        # Add main title to the figure
        fig.suptitle(f'Performance Metrics Comparison for {building_name}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def optimize_building(self, building: Building) -> tuple:
        """
        Optimize parameters for a single building.
        
        Args:
            building (Building): Building to optimize
            
        Returns:
            tuple: Optimal parameters and optimized energy consumption
        """
        print(f"\nOptimizing parameters for {building.name}...")
        
        self.current_building = building
        param_ranges = self.setup_parameter_ranges(building)
        bounds = [param_ranges[param] for param in param_ranges.keys()]
        self.current_param_names = list(param_ranges.keys())
        
        default_tank = WaterTank("Default Tank", 318.15, 5.0, 200, 4186)
        original_sim = Simulation(building, self.heat_pump, default_tank, self.coords)
        original_sim.simulate_system()
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            popsize=15,
            maxiter=30,
            workers=-1,
            updating='deferred',
            disp=True
        )
        
        optimal_params = {name: value for name, value 
                         in zip(self.current_param_names, result.x)}
        
        optimized_building = Building(
            building.name,
            building.Aw,
            building.Ar,
            optimal_params['Uw'],
            optimal_params['Ur'],
            optimal_params['Tsp_K']
        )
        
        optimized_tank = WaterTank(
            "Optimized Tank",
            optimal_params['T_initial'],
            optimal_params['Utank'],
            optimal_params['m'],
            4186
        )
        
        optimized_sim = Simulation(optimized_building, self.heat_pump, optimized_tank, self.coords)
        optimized_sim.simulate_system()
        
        self.plot_performance_metrics(building.name, original_sim, optimized_sim)
        
        return optimal_params, optimized_sim.total_energy_consumption    

    def create_comparison_table(self, building: Building, 
                              optimal_params: dict, 
                              original_energy: float,
                              optimized_energy: float) -> pd.DataFrame:
        """
        Create comparison table for original vs optimized parameters.
        
        Args:
            building (Building): Original building
            optimal_params (dict): Optimized parameters
            original_energy (float): Original energy consumption
            optimized_energy (float): Optimized energy consumption
            
        Returns:
            pd.DataFrame: Comparison table of parameters
        """
        # Create default tank for comparison
        original_tank = WaterTank("Default Tank", 318.15, 5.0, 200, 4186)
        
        # Define data structure for comparison table
        data = {
            'Parameter': [
                'Wall U-value (W/m²K)',
                'Roof U-value (W/m²K)',
                'Set Point Temperature (°C)',
                'Initial Tank Temperature (°C)',
                'Tank Heat Loss Coefficient (W/K)',
                'Tank Mass (kg)',
                'Energy Consumption (kWh)'
            ],
            'Original Value': [
                building.Uw,
                building.Ur,
                building.Tsp_K - 273.15,  # Convert K to °C
                original_tank.T_tank - 273.15,  # Convert K to °C
                original_tank.Utank,
                original_tank.m,
                original_energy
            ],
            'Optimized Value': [
                optimal_params['Uw'],
                optimal_params['Ur'],
                optimal_params['Tsp_K'] - 273.15,  # Convert K to °C
                optimal_params['T_initial'] - 273.15,  # Convert K to °C
                optimal_params['Utank'],
                optimal_params['m'],
                optimized_energy
            ]
        }
        
        # Calculate percentage changes
        data['Change (%)'] = [
            ((opt - orig) / orig * 100) if orig != 0 else float('inf')
            for opt, orig in zip(data['Optimized Value'], data['Original Value'])
        ]
        
        return pd.DataFrame(data)

    def plot_table_results(self, comparison_df, building_name, original_energy, optimized_energy):
        """
        Plot the comparison table results.
        
        Args:
            comparison_df (pd.DataFrame): Comparison data
            building_name (str): Name of the building
            original_energy (float): Original energy consumption
            optimized_energy (float): Optimized energy consumption
        """
        # Create figure and remove axes
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        # Add title and descriptive text
        plt.title(f'Results for {building_name}\n', 
                 pad=20, 
                 fontsize=16, 
                 color='blue', 
                 fontweight='bold')
        
        plt.figtext(0.5, 0.85, 
                   "Please refer to the table below to see the tailored recommendations for this building type",
                   ha='center',
                   fontsize=12,
                   style='italic')
        
        # Prepare table data and format
        table_data = [
            comparison_df['Parameter'],
            comparison_df['Original Value'].round(2),
            comparison_df['Optimized Value'].round(2),
            comparison_df['Change (%)'].round(2)
        ]
        
        # Transform data for table display
        table_data = list(map(list, zip(*table_data)))
        column_labels = ['Parameter', 'Original Value', 'Optimized Value', 'Change (%)']
        
        # Create and format table
        table = ax.table(cellText=table_data,
                        colLabels=column_labels,
                        loc='center',
                        cellLoc='center')
        
        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)
        
        # Style header row
        for i in range(len(column_labels)):
            table.get_celld()[(0, i)].set_facecolor('#E6E6E6')
            table.get_celld()[(0, i)].set_text_props(weight='bold', size=12)
        
        # Add energy reduction text if applicable
        if original_energy != 0:
            energy_reduction = ((original_energy - optimized_energy) / original_energy * 100)
            plt.figtext(0.1, 0.02, 
                       f'Energy Reduction: {energy_reduction:.1f}%',
                       fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def optimize_all_buildings(self, buildings: list):
        """
        Optimize parameters for all building types and display results.
        
        Args:
            buildings (list): List of buildings to optimize
            
        Returns:
            list: List of tuples containing building names and comparison tables
        """
        results = []
        print("")
        print("_"*10, "Beginning Energy Consumption Optimisation for all building types", "_"*10)
        print("")
        print("A custom set of recommendations for the optimised building, tank and heat pump parameters\n"
              "will be displayed in a table and saved to an excel file for you ")
        
        for building in buildings:
            default_tank = WaterTank("Default Tank", 318.15, 5.0, 200, 4186)
            print("")
            print("Please wait while the optimisation process takes place\n"
                  "This may take 2 to 3 minutes...")
            print(f"\nOptimizing parameters for {building.name}...")
            
            original_sim = Simulation(building, self.heat_pump, default_tank, self.coords)
            original_sim.simulate_system()
            original_energy = original_sim.total_energy_consumption
            
            optimal_params, optimized_energy = self.optimize_building(building)
            
            comparison = self.create_comparison_table(
                building, optimal_params, original_energy, optimized_energy
            )
            
            results.append((building.name, comparison))
            
            self.plot_table_results(comparison, building.name, original_energy, optimized_energy)
        
        return results


# Main script execution
if __name__ == "__main__":
    # Define test buildings with different characteristics
    test_buildings = [
        Building("Default Building", 132.0, 120.0, 0.51, 0.18, 293.15),
        Building("Poorly Insulated Building", 150.0, 140.0, 1.0, 0.5, 293.15),
        Building("Well Insulated Building", 132.0, 120.0, 0.2, 0.1, 293.15),
        Building("Large Modern Building", 200.0, 180.0, 0.15, 0.1, 293.15)
    ]

    # Create optimizer instance and run optimization
    optimizer = BuildingOptimizer()
    results = optimizer.optimize_all_buildings(test_buildings)

    # Print optimization summary header
    print("\n" + "="*80 + "\n" + " "*30 + "OPTIMIZATION SUMMARY\n" + "="*80)
    
    # Process and display results for each building
    for building_name, comparison_df in results:
        # Calculate key metrics
        energy_reduction = abs(comparison_df.loc[6, 'Change (%)'])
        avg_parameter_change = np.mean([abs(x) for x in comparison_df['Change (%)'].iloc[:-1]])
        
        # Print building analysis header
        print(f"\n{'-'*80}")
        print(f"{' '*30}Analysis for {building_name}")
        print(f"{'-'*80}")
        
        # Display summary metrics
        summary_data = [
            ["Energy Reduction", f"{energy_reduction:.1f}%"],
            ["Average Parameter Change", f"{avg_parameter_change:.1f}%"]
        ]
        print(tabulate(summary_data, headers=["Metric", "Value"], 
                      tablefmt="fancy_grid", numalign="right"))
    
        # Analyze and display significant parameter changes
        changes = comparison_df['Change (%)'].iloc[:-1]
        significant_changes = changes[abs(changes) > 10].sort_values(key=abs, ascending=False)
        
        if not significant_changes.empty:
            print(f"\nSignificant Parameter Changes (>10%):")
            
            significant_data = []
            for param, change in significant_changes.items():
                param_name = comparison_df.loc[changes.index[changes.index == param], 'Parameter'].iloc[0]
                significant_data.append([param_name, f"{change:+.1f}%"])
            
            print(tabulate(significant_data, 
                          headers=["Parameter", "Change"],
                          tablefmt="fancy_grid",
                          numalign="right"))
        print("\n")
                    
    # Save results to Excel file
    with pd.ExcelWriter('building_optimization_results.xlsx') as writer:
         for building_name, comparison_df in results:
             comparison_df.to_excel(writer, sheet_name=building_name[:31], index=False)
             
    print("\nOptimization results have been saved to 'building_optimization_results.xlsx'")
