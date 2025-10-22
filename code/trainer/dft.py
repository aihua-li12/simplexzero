# plot.py

# ======================================================================= #
#   This script contains Depth-First Search over the abundance data and   #
#   plotting function for the tree.                                       #
# ======================================================================= #

import polars as pl
import os
import random
import numpy as np
from plot import *

random.seed(1)
np.random.seed(1)



def generate_df(num_taxa: int = 7, num_samples: int = 4) -> pl.DataFrame:
    """Generates a large, realistic Polars DataFrame for testing microbiome data processing
    Args:
        num_taxa: The number of rows (features/ASVs) to generate.
        num_samples: The number of sample columns to generate.
    Returns:
        A Polars DataFrame with the specified dimensions.
    """

    # A plausible, simplified taxonomy to sample from
    taxonomy_map = {
        "Bacteria": {
            "Firmicutes": {"Bacilli": ["Lactobacillales", "Bacillales"], "Clostridia": ["Clostridiales"]},
            "Proteobacteria": {"Gammaproteobacteria": ["Enterobacterales", "Pasteurellales"]},
            "Actinobacteria": {"Actinomycetia": ["Corynebacteriales", "Bifidobacteriales"]},
        },
        "Archaea": {
            "Euryarchaeota": {"Methanomicrobia": ["Methanosarcinales"]}
        }
    }

    data = []
    # Use tqdm for a progress bar, as this can take a moment
    for i in range(num_taxa):
        # --- Generate Taxonomic Lineage ---
        domain = random.choice(list(taxonomy_map.keys()))
        phylum_options = list(taxonomy_map[domain].keys())
        phylum = random.choice(phylum_options) if random.random() > 0.01 else None

        klass = None
        if phylum:
            klass_options = list(taxonomy_map[domain][phylum].keys())
            klass = random.choice(klass_options) if random.random() > 0.05 else None
        
        order = None
        if klass:
            order_options = taxonomy_map[domain][phylum][klass]
            order = random.choice(order_options) if random.random() > 0.15 else None
        
        # Family and Genus are often missing, so we'll just simulate them with higher null probability
        family = f"Family_{i%50}" if random.random() > 0.3 else None
        genus = f"Genus_{i%200}" if random.random() > 0.6 else None

        row = {
            "asv_id": f"asv_{i:05d}",
            "Domain": domain,
            "Phylum": phylum,
            "Class": klass,
            "Order": order,
            "Family": family,
            "Genus": genus,
        }
        data.append(row)
    
    df_taxa = pl.DataFrame(data)

    # --- Generate Sparse Sample Data ---
    sample_cols = {}
    for j in range(num_samples):
        # Create sparse data: 90% are zeros, 10% are random counts
        is_zero = np.random.choice([True, False], size=num_taxa, p=[0.9, 0.1])
        counts = np.where(is_zero, 0, np.random.randint(1, 1000, size=num_taxa))
        sample_cols[f"sample_{j:03d}"] = counts

    df_samples = pl.DataFrame(sample_cols)

    # Combine taxonomic data with sample data
    df_large = pl.concat([df_taxa, df_samples], how="horizontal")
    
    return df_large



class DepthFirstSearch:
    """Depth-first search (DFS)
    Args:
        df: unordered dataframe, with columns ["asv_id", "Domain", "Phylum", 
        "Class", "Order", "Family", "Genus", ... sample ids ...]
    """
    def __init__(self, df:pl.DataFrame):
        self.df = df
        self.taxa_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus"]
        cols_to_check = ["asv_id"] + self.taxa_levels

        assert set(cols_to_check).issubset(df.columns), \
            f"Missing {set(cols_to_check) - set(df.columns)} column in DataFrame."

        print(".... Creating the tree ....")
        self.tree = self.build_tree()
        print(".... Ordering ASV ids ....")
        self.ordered_ids = self.search(self.tree)
        print(".... Rearranging DataFrame ....")
        self.df_reordered = self.rearrange_dataframe()
        print("✅ Done!")
    
    def build_tree(self) -> dict:
        """Build a nested dictionary tree"""
        tree = {}
        for row in self.df.iter_rows(named=True):
            current_level_node = tree
            for level in self.taxa_levels:
                taxon_name = row[level]
                # If null, then our path has run out and we can't go deeper
                if taxon_name is None: 
                    break
                # current_level_node points to the last valid node
                current_level_node = current_level_node.setdefault(taxon_name, {})
            # After completing all levels, add the asv_id to the final valid node
            if '_asv_ids' not in current_level_node:
                current_level_node['_asv_ids'] = []
            current_level_node['_asv_ids'].append(row['asv_id'])
        return tree

    def search(self, tree) -> list:
        """Search the tree and returned the ordered asv ids"""
        ordered_ids = []

        # --- Step 1: Go Deep First ---
        # Look for all the children taxa and traverse them first.
        # Sorting keys makes the traversal deterministic.
        for name in sorted(tree.keys()):
            if name == '_asv_ids':
                continue  # Ignore the leaf nodes for now
            
            # Recursively call on the child and add the results to our list
            ordered_ids.extend(self.search(tree[name]))

        # --- Step 2: Add Local Leaves Last ---
        # After returning from all deep dives, add the ASVs at this level.
        if '_asv_ids' in tree:
            ordered_ids.extend(sorted(tree.get('_asv_ids', [])))
            
        return ordered_ids

    def rearrange_dataframe(self) -> pl.DataFrame:
        """Rearrange the original DataFrame rows using a categorical sort"""
        # Create an ordered enum/categorical type from the search results
        ordered_asv_type = pl.Enum(self.ordered_ids)
        
        # Cast the column to this new ordered type and then sort
        df_reordered = self.df.with_columns(
            pl.col("asv_id").cast(ordered_asv_type)
        ).sort("asv_id")
        
        return df_reordered

    def _print_tree(self, node: dict, name: str = "root", 
                    prefix: str = "", is_last: bool = True) -> None:
        """Recursively print a text-based representation of the tree"""
        print(prefix + ("└── " if is_last else "├── ") + name) # Use the passed name
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Separate children and leaves
        children = {k: v for k, v in node.items() if k != '_asv_ids'}
        asv_ids = node.get('_asv_ids', [])
        child_items = list(children.items())

        # Recursively call for children
        for i, (child_name, child_node) in enumerate(child_items):
            is_last_child = (i == len(child_items) - 1) and not asv_ids
            # Pass the child's name as an argument
            self._print_tree(child_node, child_name, child_prefix, is_last_child)
            
        # Print the leaves
        for i, asv_id in enumerate(asv_ids):
            is_last_leaf = (i == len(asv_ids) - 1)
            print(child_prefix + ("└── " if is_last_leaf else "├── ") + f"*{asv_id}*")
    
    def __str__(self):
        self._print_tree(self.tree)
        return ""


if __name__ == "__main__":
    # agg_data_dir = "../../data/aggregation/"
    # file_path = os.path.join(agg_data_dir, "AGP.taxonomyASV.parquet")
    # df = pl.read_parquet(file_path).drop("Species")

    
    df = generate_df(num_taxa=200, num_samples=0)
    dfs = DepthFirstSearch(df)
    # dfs.plot_matplotlib_radial()

    # plot = PlotTree(dfs.tree)
    # plot.plot_matplotlib_radial()


    file_path = os.path.join(agg_data_dir, 'AGP.taxonomyASV.rearranged.parquet')
    dfs.df_reordered.write_parquet(file_path, compression='zstd')
    
    
