from .SimulationLoader import SimulationLoader
from .MongoLoader import MongoLoader
from .JsonLoader import JsonLoader


def check_install():
    """
    Test function to verify that the module is working correctly.
    """
    print(f"🚀 Package {__package__} is working correctly 🚀")
