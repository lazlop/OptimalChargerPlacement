"""
EV Charger Placement Optimization

Implements Mixed Integer Linear Programming (MILP) to optimally place
EV chargers while minimizing grid upgrade costs and customer inconvenience costs for displacing demand.
"""

import pandas as pd
import numpy as np
from pulp import *
import warnings
warnings.filterwarnings('ignore')


class EVChargerOptimization:
    """
    Optimization model for EV charger placement
    """
    def __init__(self, 
                 network_df_path='data/network_analysis/network_df.parquet',
                 grid_constraint_df_path='data/network_analysis/example_grid_constraint_df.parquet',
                 feeder_block_matrix_path='data/network_analysis/feeder_block_matrix.parquet'):
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
        self.network_df = pd.read_parquet(network_df_path)
        self.grid_df = pd.read_parquet(grid_constraint_df_path)
        self.feeder_block_df = pd.read_parquet(feeder_block_matrix_path)
        
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
        
        # Extract unique nodes from network
        origin_nodes = self.network_df['geoid'].unique()
        dest_nodes = self.network_df['neighbor_geoid'].unique()
        self.nodes = list(set(origin_nodes) | set(dest_nodes))
        print(f"Total nodes (census block groups): {len(self.nodes)}")
        
        # Create node demand dictionary
        self.node_demand = {}
        for node in self.nodes:
            # Get demand from origin_demand column
            demand_rows = self.network_df[self.network_df['geoid'] == node]
            if len(demand_rows) > 0:
                self.node_demand[node] = demand_rows['origin_demand_(kW)'].iloc[0]
            else:
                self.node_demand[node] = 0.0
        
        # Create edges list with costs
        self.edges = []
        for idx, row in self.network_df.iterrows():
            origin = row['geoid']
            neighbor = row['neighbor_geoid']
            # Cost is based on distance (could be weighted differently)
            cost = row['distance_km']
            self.edges.append((origin, neighbor, cost))
        
        print(f"Total edges: {len(self.edges)}")
        
        # Prepare feeder data
        self.feeders = list(self.grid_df['feeder_id'].values)
        self.feeder_capacity = dict(zip(
            self.grid_df['feeder_id'], 
            self.grid_df['available_capacity']
        ))
        
        print(f"Total feeders: {len(self.feeders)}")
        
        # Create feeder-node mapping from matrix
        # The matrix has feeders as rows and GEOIDs as columns
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
        
        # REMOVE unrepresented nodes and edges
        nodes_removed = 0
        edges_removed = 0
        for geoid in self.nodes:
            if geoid not in nodes_represented:
                nodes_removed += 1
                self.nodes.remove(geoid)
                del self.node_demand[geoid]
                for edge in self.edges:
                    if edge[0] == geoid or edge[1] == geoid:
                        self.edges.remove(edge)
                        edges_removed += 1
        
        print(f"Nodes not mapped to feeders: {len(set(self.nodes) - nodes_represented)} of {len(self.nodes)}")
        print(f"Feeder-node mappings created: {len(self.feeder_node_proportion)}")
        print(f"Removed nodes: {len(nodes_represented)}")
        print(f"Removed edges: {edges_removed}")

        
    def build_model(self, 
                   feeder_upgrade_cost=10000,
                   charger_capacity_cost=100,
                   displacement_cost_multiplier=1.0,
                   auto_upgrade_overloaded_feeders=True,
                   maximum_node_capacity=10000,
                   maximum_upgrade_capacity=1e6):
        """
        Build the MILP optimization model.
        
        Parameters:
        -----------
        feeder_upgrade_cost : float
            Fixed cost for upgrading a feeder, used to tune optimization problem to avoid upgrading feeders 
        charger_capacity_cost : float
            Cost per kW of additional charger capacity, used to tune optimization problem
        displacement_cost_multiplier : float
            Multiplier for demand displacement costs, used to tune optimization problem to displace less demand
        auto_upgrade_overloaded_feeders : bool
            If True, automatically upgrade feeders that are already overloaded (without EVs present)
        maximum_node_capacity: float
            Maximum capacity of a node, default 10000
        maximum_upgrade_capacity: float
            Maximum amount of a feeder upgrade, default 1e6
        """
        print("\nBuilding optimization model...")
        
        # Identify feeders that are already overloaded (must be upgraded or handled some other way before optimization)
        self.must_upgrade_feeders = set()
        if auto_upgrade_overloaded_feeders:
            print("Checking for overloaded feeders...")
            for k in self.feeders:
                # Check if feeder has nodes mapped to it
                if k in self.feeder_node_proportion:
                    current_load = 0
                    for node, proportion in self.feeder_node_proportion[k].items():
                        if node in self.nodes:
                            current_load += proportion * self.node_demand.get(node, 0)
                    
                    capacity = self.feeder_capacity.get(k, 0)
                    if current_load > capacity:
                        self.must_upgrade_feeders.add(k)
                        print(f"  Feeder {k}: load={current_load:.1f} kW > capacity={capacity:.1f} kW")
                # Also check if feeder has negative capacity
                elif self.feeder_capacity.get(k, 0) < 0:
                    self.must_upgrade_feeders.add(k)
                    print(f"  Feeder {k}: negative capacity={self.feeder_capacity.get(k, 0):.1f} kW")
            
            if len(self.must_upgrade_feeders) > 0:
                print(f"Auto-upgrading {len(self.must_upgrade_feeders)} feeders that are already overloaded")
            else:
                print("No overloaded feeders found")
        
        # Create the model
        self.model = LpProblem("EV_Charger_Placement", LpMinimize)
        
        # Decision Variables
        # x[i] = additional charger capacity (kW) placed in node i
        self.x = LpVariable.dicts("charger_capacity", 
                                  self.nodes, 
                                  lowBound=0, 
                                  cat='Continuous')
        
        # y[i,j] = demand moved from node i to node j
        self.y = {}
        for (i, j, cost) in self.edges:
            self.y[(i, j)] = LpVariable(f"demand_flow_{i}_{j}", 
                                       lowBound=0, 
                                       cat='Continuous')
        
        # n[i] = binary variable indicating if node i is upgraded
        self.n = LpVariable.dicts("node_upgraded", 
                                  self.nodes, 
                                  cat='Binary')
        
        # f[k] = binary variable indicating if feeder k is upgraded
        self.f = LpVariable.dicts("feeder_upgraded",
                                  self.feeders,
                                  cat='Binary')
        
        # Objective Function
        # Minimize: feeder upgrade costs + charger capacity costs + displacement costs
        objective = 0
        
        # Feeder upgrade costs
        for k in self.feeders:
            objective += feeder_upgrade_cost * self.f[k]
        
        # Charger capacity costs
        for i in self.nodes:
            objective += charger_capacity_cost * self.x[i]
        
        # Demand displacement costs
        for (i, j, cost) in self.edges:
            objective += displacement_cost_multiplier * cost * self.y[(i, j)]
        
        self.model += objective, "Total_Cost"
        
        # Constraints
        
        # 1a. EV Conservation of Demand
        # The demand removed from the node must not exceed the original demand (or the original demand plus the demand moved in, if we would prefer it that way)
        # NOTE: I feel like we're missing an element of human behavior. We may be trying to shift one group to its adjacent node and another group to that node - would be good to find these wrinkles in analysis afterwoards. 
        for i in self.nodes:
            demand_in = lpSum([self.y[(j, i)] for (j, k, _) in self.edges if k == i])
            demand_out = lpSum([self.y[(i, j)] for (orig, j, _) in self.edges if orig == i])
            
            self.model += (
                self.node_demand.get(i, 0) >= demand_out,
                f"Demand_Balance_{i}"
            )

        # 1b. EV Charger capacity sufficiency
        # For each node: charger capacity >= demand flowing in - demand flowing out + original demand
        for i in self.nodes:
            demand_in = lpSum([self.y[(j, i)] for (j, k, _) in self.edges if k == i])
            demand_out = lpSum([self.y[(i, j)] for (orig, j, _) in self.edges if orig == i])
            
            self.model += (
                self.x[i] >= self.node_demand.get(i, 0) + demand_in - demand_out,
                f"Charger_Sufficiency_{i}"
            )
        
        # 2. Feeder Capacity Constraint
        # For each feeder, the total load from its nodes must not exceed capacity
        # Load at a node = original demand + charger capacity installed at that node
        for k in self.feeders:
            if k in self.feeder_node_proportion:
                feeder_load = 0
                for node, proportion in self.feeder_node_proportion[k].items():
                    if node in self.nodes:
                        # Load at node = original demand + new charger capacity
                        # (demand flows don't affect grid load, only where charging happens)
                        node_load = self.node_demand.get(node, 0) + self.x[node]
                        feeder_load += proportion * node_load
                
                # Capacity constraint with upgrade option
                # If feeder is upgraded (f[k]=1), we assume it can be upgraded by some amount (currently set very large)
                self.model += (
                    feeder_load <= self.feeder_capacity.get(k, 0) + maximum_upgrade_capacity * self.f[k],
                    f"Feeder_Capacity_{k}"
                )
        
        # 3a. Installation Constraints
        # Nodes can be upgraded to a certain maximum capacity, (can maybe consider whether the upgrades happen with feeder upgrades)
        for i in self.nodes:
            self.model += (
                self.x[i] <= maximum_node_capacity, # * self.n[i],
                f"Upgrade_Required_{i}"
            )
        
        # # 3b. Nodes are upgraded where the feeder has been upgraded
        # for i in self.nodes:
        #     for k in self.feeders:
        #         if k in self.feeder_node_proportion:
        #             if i in self.feeder_node_proportion[k]:
        #                 self.model += (
        #                     self.n[i] == self.f[k],
        #                     f"Node_Upgrade_{i}"
        #                 )

        # Maximum number of nodes can be upgraded
        self.model += (
            lpSum([self.n[i] for i in self.nodes]) <= max_upgrades,
            "Max_Upgrades"
        )

        # 4. Force upgrade of already-overloaded feeders
        for k in self.must_upgrade_feeders:
            self.model += (
                self.f[k] == 1,
                f"Force_Upgrade_Feeder_{k}"
            )
        
        print(f"Model built with {len(self.model.variables())} variables and {len(self.model.constraints)} constraints")
        
    def solve(self, time_limit=300, gap=0.01):
        """
        Solve the optimization model.
        
        Parameters:
        -----------
        time_limit : int
            Maximum time in seconds for solver
        gap : float
            MIP gap tolerance (0.01 = 1%)
        """
        print(f"\nSolving optimization model...")
        print(f"Time limit: {time_limit}s, MIP gap: {gap*100}%")
        
        # Solve with CBC 
        solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap, msg=1)
        self.model.solve(solver)
        
        # Check solution status
        status = LpStatus[self.model.status]
        print(f"\nSolution Status: {status}")
        
        if self.model.status == LpStatusOptimal or self.model.status == LpStatusNotSolved:
            print(f"Objective Value: ${value(self.model.objective):,.2f}")
            return True
        else:
            print("No feasible solution found!")
            return False
    
    def get_results(self):
        """
        Extract and format optimization results.
        
        Returns:
        --------
        dict : Dictionary containing results dataframes
        """
        if self.model is None or self.model.status not in [LpStatusOptimal, LpStatusNotSolved]:
            print("No solution available!")
            return None
        
        print("\nExtracting results...")
        
        # Node upgrades and capacity additions
        node_results = []
        for i in self.nodes:
            if value(self.n[i]) > 0.5:  # Binary variable is 1
                capacity = value(self.x[i])
                if capacity > 0:
                    node_results.append({
                        'node': i,
                        'additional_capacity_kW': capacity,
                        'original_demand_kW': self.node_demand.get(i, 0),
                        'upgraded': True
                    })
        
        node_df = pd.DataFrame(node_results)
        if len(node_df) > 0:
            node_df = node_df.sort_values('additional_capacity_kW', ascending=False)
        
        # Feeder upgrades
        feeder_results = []
        for k in self.feeders:
            if value(self.f[k]) > 0.5:  # Binary variable is 1
                feeder_results.append({
                    'feeder_id': k,
                    'original_capacity': self.feeder_capacity.get(k, 0),
                    'upgraded': True
                })
        
        feeder_df = pd.DataFrame(feeder_results)
        
        # Demand flows
        flow_results = []
        for (i, j, cost) in self.edges:
            flow = value(self.y[(i, j)])
            if flow > 0.01:  # Only significant flows
                flow_results.append({
                    'from_node': i,
                    'to_node': j,
                    'flow_kW': flow,
                    'distance_km': cost,
                    'cost': cost * flow
                })
        
        flow_df = pd.DataFrame(flow_results)
        if len(flow_df) > 0:
            flow_df = flow_df.sort_values('flow_kW', ascending=False)
        
        # Summary statistics
        summary = {
            'total_cost': value(self.model.objective),
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
        
        Parameters:
        -----------
        results : dict
            Results dictionary from get_results()
        output_prefix : str
            Prefix for output files
        """
        if results is None:
            return
        
        print(f"\nSaving results with prefix: {output_prefix}")
        
        # Save node results
        if len(results['nodes']) > 0:
            results['nodes'].to_csv(f'{output_prefix}_nodes.csv', index=False)
            print(f"Saved {len(results['nodes'])} node upgrades to {output_prefix}_nodes.csv")
        
        # Save feeder results
        if len(results['feeders']) > 0:
            results['feeders'].to_csv(f'{output_prefix}_feeders.csv', index=False)
            print(f"Saved {len(results['feeders'])} feeder upgrades to {output_prefix}_feeders.csv")
        
        # Save flow results
        if len(results['flows']) > 0:
            results['flows'].to_csv(f'{output_prefix}_flows.csv', index=False)
            print(f"Saved {len(results['flows'])} demand flows to {output_prefix}_flows.csv")
        
        # Save summary
        summary_df = pd.DataFrame([results['summary']])
        summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
        print(f"Saved summary to {output_prefix}_summary.csv")
        
    def print_summary(self, results):
        """
        Print a summary of the optimization results.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from get_results()
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
    
    # Build model with parameters
    opt.build_model(
        feeder_upgrade_cost=10000,  # Cost per feeder upgrade
        charger_capacity_cost=100,  # Cost per kW of capacity
        displacement_cost_multiplier=50  # Multiplier for displacement costs
    )
    
    # Solve
    success = opt.solve(time_limit=300, gap=0.05)
    
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
