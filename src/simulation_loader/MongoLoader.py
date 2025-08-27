from os import getenv

from dotenv import load_dotenv
import pandas as pd
from typing import Tuple
from db_interact import MongoInteract

from .SimulationLoader import SimulationLoader

class MongoLoader(MongoInteract, SimulationLoader):
    """Class to load simulation data from a Mongo database."""
    def __init__(self, path="./.env"):
        super().__init__(path)

        load_dotenv(path)
        self.sim_collection = getenv('SIMULATION_COLLECTION')

        if self.sim_collection is None:
            raise ValueError("SIMULATION_COLLECTION environment variable not set in .env file")

    def load_simulations(self) -> pd.DataFrame:
        """Load all simulations from the database and return them as a pandas DataFrame."""
        assert self.db is not None, "Database connection is not established."

        all_data = self.db[self.sim_collection].find({})

        return pd.DataFrame(list(all_data))

    def retrive_simulations(self) -> pd.DataFrame:
        """Load simulations from the database into a pandas DataFrame."""
        assert self.db is not None, "Database connection is not established."

        all_simulations = list(self.db[self.sim_collection].find({}))

        df = pd.json_normalize(all_simulations)
        df.set_index('_id', inplace=True)
      
        return df

    def retrieve_simulations_io(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load simulations and their I/O data from the database into pandas DataFrames."""
        assert self.db is not None, "Database connection is not established."

        all_simulations = list(self.db[self.sim_collection].find({}))
        inputs = []
        outputs = []
        for sim in all_simulations:
            sim_id = sim.get('_id')
            in_dict = sim.get('in')
            in_dict['_id'] = sim_id
            out_dict = sim.get('out')
            out_dict['_id'] = sim_id
            inputs.append(in_dict)
            outputs.append(out_dict)
            
        df_in = pd.DataFrame(inputs)
        df_in.set_index('_id', inplace=True)

        df_out = pd.DataFrame(outputs)
        df_out.set_index('_id', inplace=True)

        return df_in, df_out