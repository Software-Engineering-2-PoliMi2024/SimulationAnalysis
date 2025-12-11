from abc import ABC, abstractmethod
from db_interact import DBinteract
import pandas as pd
from typing import List, Tuple, Literal


class SimulationLoader(DBinteract, ABC):
    """Abstract base class defining the interface for loading simulation data."""

    @staticmethod
    def withTags(data: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
        """Returns a filtered DataFrame containing only the rows whose tags contain all the specified tags.

        Args:
            data (pd.DataFrame): The input DataFrame to filter.
            tags (List[str]): The tags to filter by.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        # Remove rows with Nan tags
        data = data[data["tags"].notna()]
        return data[data["tags"].apply(lambda x: all(tag in x for tag in tags))]

    @staticmethod
    def addAggregation(
        data: pd.DataFrame,
        columnName: str,
        aggregationFunction: Literal["avg", "min", "max", "element", "median"] = "avg",
        absolute: bool = True,
    ):
        """Adds a new column to the DataFrame containing the aggregated values of the specified column.
        The aggregation is performed row-wise, thus the given column must contain lists or tuples.
        The new column will be named `{columnName}.{aggregationFunction}`.

        Args:
            data (pd.DataFrame): The input DataFrame to modify.
            columnName (str): The name of the column to aggregate.
            aggregationFunction (Literal['avg', 'min', 'max', 'element', 'median'], optional): The aggregation function to use. Defaults to 'avg'.
            absolute (bool, optional): Whether to take the absolute value before aggregation. Defaults to False.

        Raises:
            KeyError: If the specified column is not found in the DataFrame.
            TypeError: If the specified column does not contain lists or tuples.
            ValueError: If the aggregation function is not one of 'avg', 'min', 'max', 'element', 'median'.
        """
        index = 0
        if aggregationFunction == "element":
            columnNameParts = columnName.split(".")
            if len(columnNameParts) < 2:
                raise ValueError(
                    "For 'element' aggregationFunction, columnName must be in the format 'baseColumn.index'"
                )
            columnName = ".".join(columnNameParts[:-1])
            index = int(columnNameParts[-1])

        if columnName not in data.columns:
            raise KeyError(f"Column '{columnName}' not found in data")

        if data[columnName].dtype not in [list, tuple]:
            raise TypeError(f"Column '{columnName}' must contain lists or tuples")

        if absolute:
            tmp = data[columnName].apply(lambda x: [abs(i) for i in x])
        else:
            tmp = data[columnName]

        mapping = {
            "avg": lambda x: sum(x) / len(x) if len(x) > 0 else float("nan"),
            "min": lambda x: min(x) if len(x) > 0 else float("nan"),
            "max": lambda x: max(x) if len(x) > 0 else float("nan"),
            "element": lambda x: x[index] if len(x) > index else float("nan"),
            "median": lambda x: sorted(x)[len(x) // 2] if len(x) > 0 else float("nan"),
        }

        if aggregationFunction not in mapping:
            raise ValueError(f"aggfunc must be one of {list(mapping.keys())}")

        data[f"{columnName}.{aggregationFunction}"] = tmp.apply(
            lambda x: mapping[aggregationFunction](x)
        )

    @staticmethod
    def addFps(data: pd.DataFrame):
        """Adds a new column to the DataFrame containing the frames per second (FPS) values.
        The new column will be named `fps`.

        Args:
            data (pd.DataFrame): The input DataFrame to modify.
        """
        if (
            "out.elapsedTime" not in data.columns
            or "out.iterations" not in data.columns
        ):
            raise KeyError(
                "Columns 'out.elapsedTime' and 'out.iterations' must be present in the DataFrame"
            )

        data["fps"] = data["out.iterations"] / data["out.elapsedTime"]
        data["fps"] = data["fps"].fillna(0)

    @abstractmethod
    def retrieve_simulations(self) -> pd.DataFrame:
        """Load simulations from the database into a pandas DataFrame."""
        raise RuntimeError("must be implemented in subclass")

    @abstractmethod
    def retrieve_simulations_io(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load simulations and their I/O data from the database into pandas DataFrames."""
        raise RuntimeError("must be implemented in subclass")
