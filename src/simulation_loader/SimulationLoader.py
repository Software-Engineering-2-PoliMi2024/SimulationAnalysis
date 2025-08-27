from abc import ABC, abstractmethod
from db_interact import DBinteract
import pandas as pd
from typing import Tuple

class SimulationLoader(DBinteract, ABC):
    """Abstract base class defining the interface for loading simulation data."""

    @abstractmethod
    def retrive_simulations(self) -> pd.DataFrame:
        """Load simulations from the database into a pandas DataFrame."""
        print("must be implemented in subclass")
        return None

    @abstractmethod
    def retrieve_simulations_io(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load simulations and their I/O data from the database into pandas DataFrames."""
        print("must be implemented in subclass")
        return None, None