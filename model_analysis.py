import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path

def get_parameters(model):
    total=0
    for parameter in list(model.parameters()):
        dim_size=1
        for dim in list(parameter.size()):
            dim_size = dim_size*dim
        total += dim_size
    return total

def load_dataframes(tables: List[Path]):
    dataframes = []
    for table in tables:
        frame = pd.read_csv(table)
        dataframes.append(frame)

    return dataframes

def display_data(tables: List[Path], y_axis_index: int, x_axis_index: int, y_name: str, x_name: str, title: str, y_bounds: Optional[Tuple[float, float]] = None, x_bounds: Optional[Tuple[float, float]] = None, save_path = None):
    tables: List[pd.DataFrame] = load_dataframes(tables)

    for table in tables:
        table_len = len(table.columns)
        indexes = list(range(table_len))
        indexes.remove(y_axis_index)
        indexes.remove(x_axis_index)
        table.drop(labels=indexes, axis=1)

    joined_table = pd.DataFrame()
    joined_table[x_name] = tables[0][x_name]
    for i, table in enumerate(tables):
        joined_table[f'{y_name}_{i}'] = table[y_name]

    for i in range(len(tables)):
        plt.plot(joined_table[x_name], joined_table[f'{y_name}_{i}'], label=i)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(y_bounds)
    plt.xlim(x_bounds)

    if save_path != None:
        plt.savefig(save_path)

    plt.show()