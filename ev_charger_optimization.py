import pandas as pd
import numpy as np
from pyomo.environ import *
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class EVChargerOptimization:
    """
    Optimization model for EV charger placement using Pyomo
    """
    def __init__(self, 
                 base_dir = 'data/network_analysis',
                 network_df_path='network_df.parquet',
                 grid_constraint_df_path='example_grid_constraint_df.parquet',
                 feeder_block_matrix_path='feeder_block_matrix.parquet'):
        """
        Initialize the optimization model with data files.
        
        Parameters:
        -----------
        network_df_path : str
            Path to network dataframe with edges and demand data
        grid_constraint_df_path : str
            Path to grid constraint dataframe with feeder capacities
        feeder_block_matrix_path : str
            Path to feeder-block mapping matrix
        """
        print("Loading data files...")
        self.network_df = pd.read_parquet(Path(base_dir) / network_df_path)
        self.grid_df = pd.read_parquet(Path(base_dir) / grid_constraint_df_path)
        self.feeder_block_df = pd.read_parquet(Path(base_dir) / feeder_block_matrix_path)
        
        print(f"Network edges: {len(self.network_df)}")
        print(f"Grid feeders: {len(self.grid_df)}")
        print(f"Feeder-block matrix: {self.feeder_block_df.shape}")
        
        # Initialize model components
        self.model = None
        self.nodes = None
        self.edges = None
        self.feeders = None
        
    def prepare_data(self):
        """
        Prepare and validate data for optimization.
        """
        print("\nPreparing data...")
        
        origin_nodes = self.network_df['geoid'].unique()
        dest_nodes = self.network_df['neighbor_geoid'].unique()
        self.nodes = list(set(origin_nodes) | set(dest_nodes))
        print(f"Total nodes (census block groups): {len(self.nodes)}")
        
        self.node_demand = {node: self.network_df[self.network_df['geoid'] == node]['origin_demand_(kW)'].iloc[0] 
                             if len(self.network_df[self.network_df['geoid'] == node]) > 0 else 0.0 
                             for node in self.nodes}
        
        self.edges = [(row['geoid'], row['neighbor_geoid'], row['distance_km']) for idx, row in self.network_df.iterrows()]
        print(f"Total edges: {len(self.edges)}")

        self.feeders = list(self.grid_df['feeder_id'].values)
        self.feeder_capacity = dict(zip(self.grid_df['feeder_id'], self.grid_df['available_capacity']))
        
        print(f"Total feeders: {len(self.feeders)}")
        
        # Modify negative capacities to zero and apply adjustments as needed
        for feeder_id in self.feeder_capacity:
            self.feeder_capacity[feeder_id] = max(0, self.feeder_capacity[feeder_id])

        # Create feeder-node proportion mapping
        self.feeder_node_proportion = {}
        nodes_represented = set()
        for feeder_id in self.feeder_block_df.index:
            for geoid in self.feeder_block_df.columns:
                proportion = self.feeder_block_df.loc[feeder_id, geoid]
                if proportion > 0:
                    if feeder_id not in self.feeder_node_proportion:
                        self.feeder_node_proportion[feeder_id] = {}
                    self.feeder_node_proportion[feeder_id][geoid] = proportion
                    nodes_represented.add(geoid)

        # Remove unrepresented nodes and edges
        self.nodes = [geoid for geoid in self.nodes if geoid in nodes_represented]
        self.node_demand = {k: v for k, v in self.node_demand.items() if k in self.nodes}
        self.edges = [(i, j, c) for (i, j, c) in self.edges if i in self.nodes and j in self.nodes]
        
        print(f"Nodes not mapped to feeders: {len(set(self.nodes) - nodes_represented)} of {len(self.nodes)}")
        print(f"Feeder-node mappings created: {len(self.feeder_node_proportion)}")

    def build_model(self, 
                   max_upgrades=10000,
                   feeder_upgrade_cost=10000,
                   charger_capacity_cost=0,
                   displacement_cost_multiplier=1.0,
                   auto_upgrade_overloaded_feeders=True,
                   maximum_node_capacity=10000,
                   maximum_upgrade_capacity=1e6):
        """
        Build the optimization model using Pyomo.
        """
        print("\nBuilding optimization model...")
        
        # Initialize Pyomo model
        self.model = ConcreteModel()
        
        # Set indices for nodes, edges, and feeders
        self.model.N = Set(initialize=self.nodes)
        self.model.E = Set(initialize=[(i, j) for (i, j, _) in self.edges])
        self.model.F = Set(initialize=self.feeders)

        # Decision Variables
        self.model.charger_capacity = Var(self.model.N, within=NonNegativeReals)
        self.model.demand_flow = Var(self.model.E, within=NonNegativeReals)
        self.model.feeder_upgraded = Var(self.model.F, within=Binary)

        # Objective Function
        self.model.cost = Objective(expr=sum(feeder_upgrade_cost * self.model.feeder_upgraded[f] for f in self.model.F) +
                                     sum(charger_capacity_cost * self.model.charger_capacity[n] for n in self.model.N) +
                                     sum(displacement_cost_multiplier * cost * self.model.demand_flow[i, j] for (i, j, cost) in self.edges), 
                                     sense=minimize)

        # Constraints
        # Demand balance for each node
        self.model.demand_balance = ConstraintList()
        for n in self.model.N:
            self.model.demand_balance.add(sum(self.model.demand_flow[j, n] for (j, _) in self.model.E if (j, n) in self.model.E) - 
                                            sum(self.model.demand_flow[n, j] for (_, j) in self.model.E if (n, j) in self.model.E) <= self.node_demand[n])

        # Charger capacity sufficiency
        self.model.charger_sufficiency = ConstraintList()
        for n in self.model.N:
            self.model.charger_sufficiency.add(self.model.charger_capacity[n] >= self.node_demand[n] + 
                                                sum(self.model.demand_flow[j, n] for (j, _) in self.model.E if (j, n) in self.model.E) - 
                                                sum(self.model.demand_flow[n, j] for (_, j) in self.model.E if (n, j) in self.model.E))

        # Feeder capacity constraint
        self.model.feeder_capacity = ConstraintList()
        for f in self.model.F:
            if f in self.feeder_node_proportion:
                load_expr = sum(self.feeder_node_proportion[f].get(n, 0) * (self.node_demand[n] + self.model.charger_capacity[n]) for n in self.model.N)
                self.model.feeder_capacity.add(load_expr <= self.feeder_capacity.get(f, 0) + maximum_upgrade_capacity * self.model.feeder_upgraded[f])

        print(f"Model built with {len(list(self.model.component_objects(Var)))} variables and {len(list(self.model.component_objects(Constraint)))} constraints")

    def solve(self, time_limit=300, gap=0.01):
        """
        Solve the optimization model.
        """
        print(f"\nSolving optimization model...")
        
        solver = SolverFactory('cbc')
        solver.options['timeLimit'] = time_limit
        solver.options['mipgap'] = gap
        
        results = solver.solve(self.model, tee=True)
        print(f"\nSolution Status: {results.solver.status}, {results.solver.termination_condition}")
        
        self.results = results
        if results.solver.termination_condition == TerminationCondition.optimal or results.solver.termination_condition == TerminationCondition.feasible:
            print(f"Objective Value: ${self.model.cost()}")
            return True
        else:
            print("No feasible solution found!")
            return False

    def get_results(self):
        """
        Extract and format optimization results.
        """
        if self.model is None or self.results.solver.termination_condition not in [TerminationCondition.optimal, TerminationCondition.feasible]:
            print("No solution available!")
            return None
        
        print("\nExtracting results...")

        node_results = []
        for n in self.model.N:
            capacity = value(self.model.charger_capacity[n])
            if capacity > 0:
                node_results.append({
                    'node': n,
                    'additional_capacity_kW': capacity,
                    'original_demand_kW': self.node_demand[n],
                    'upgraded': True
                })
        
        node_df = pd.DataFrame(node_results)

        feeder_results = []
        for f in self.model.F:
            if value(self.model.feeder_upgraded[f]) > 0.5:
                feeder_results.append({
                    'feeder_id': f,
                    'original_capacity': self.feeder_capacity[f],
                    'upgraded': True
                })
        
        feeder_df = pd.DataFrame(feeder_results)

        flow_results = []
        for (i, j) in self.model.E:
            flow = value(self.model.demand_flow[i, j])
            if flow > 0.01:
                distance = next(cost for (x, y, cost) in self.edges if x == i and y == j)
                flow_results.append({
                    'from_node': i,
                    'to_node': j,
                    'flow_kW': flow,
                    'distance_km': distance,
                    'cost': distance * flow
                })
        
        flow_df = pd.DataFrame(flow_results)

        summary = {
            'total_cost': value(self.model.cost),
            'nodes_upgraded': len(node_df),
            'feeders_upgraded': len(feeder_df),
            'total_additional_capacity_kW': node_df['additional_capacity_kW'].sum() if len(node_df) > 0 else 0,
            'total_demand_displaced_kW': flow_df['flow_kW'].sum() if len(flow_df) > 0 else 0,
            'avg_displacement_distance_km': (flow_df['distance_km'] * flow_df['flow_kW']).sum() / flow_df['flow_kW'].sum() if len(flow_df) > 0 and flow_df['flow_kW'].sum() > 0 else 0
        }

        return {
            'nodes': node_df,
            'feeders': feeder_df,
            'flows': flow_df,
            'summary': summary
        }

    def save_results(self, results, output_prefix='optimization_results'):
        """
        Save results to CSV files.
        """
        if results is None:
            return
        
        print(f"\nSaving results with prefix: {output_prefix}")
        
        if len(results['nodes']) > 0:
            results['nodes'].to_csv(f'{output_prefix}_nodes.csv', index=False)
            print(f"Saved {len(results['nodes'])} node upgrades to {output_prefix}_nodes.csv")
        
        if len(results['feeders']) > 0:
            results['feeders'].to_csv(f'{output_prefix}_feeders.csv', index=False)
            print(f"Saved {len(results['feeders'])} feeder upgrades to {output_prefix}_feeders.csv")
        
        if len(results['flows']) > 0:
            results['flows'].to_csv(f'{output_prefix}_flows.csv', index=False)
            print(f"Saved {len(results['flows'])} demand flows to {output_prefix}_flows.csv")
        
        summary_df = pd.DataFrame([results['summary']])
        summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
        print(f"Saved summary to {output_prefix}_summary.csv")
        
    def print_summary(self, results):
        """
        Print a summary of the optimization results.
        """
        if results is None:
            return
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        
        summary = results['summary']
        print(f"\nTotal Cost: ${summary['total_cost']:,.2f}")
        print(f"\nNodes Upgraded: {summary['nodes_upgraded']}")
        print(f"Feeders Upgraded: {summary['feeders_upgraded']}")
        print(f"Total Additional Capacity: {summary['total_additional_capacity_kW']:,.2f} kW")
        print(f"Total Demand Displaced: {summary['total_demand_displaced_kW']:,.2f} kW")
        print(f"Average Displacement Distance: {summary['avg_displacement_distance_km']:.2f} km")
        
        if len(results['nodes']) > 0:
            print(f"\nTop 5 Nodes by Additional Capacity:")
            print(results['nodes'].head()[['node', 'additional_capacity_kW', 'original_demand_kW']])
        
        if len(results['flows']) > 0:
            print(f"\nTop 5 Demand Flows:")
            print(results['flows'].head()[['from_node', 'to_node', 'flow_kW', 'distance_km']])
        
        print("\n" + "="*80)

def main():
    """
    Main function to run the optimization.
    """
    # Initialize optimization
    opt = EVChargerOptimization()
    
    # Prepare data
    opt.prepare_data()
    
    # Build model 
    opt.build_model(feeder_upgrade_cost=100000)
    
    # Solve
    success = opt.solve(time_limit=60, gap=0.05)
    
    if success:
        # Get results
        results = opt.get_results()
        
        # Print summary
        opt.print_summary(results)
        
        # Save results
        opt.save_results(results)
        
        return results
    
    return None

if __name__ == "__main__":
    results = main()