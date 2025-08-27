import simulation_loader
import simulation_analysis
from dotenv import find_dotenv

def test_import() -> None:
    """Test that the simulation_analysis module can be imported."""
    mongo_loader = simulation_loader.MongoLoader(path=find_dotenv())  
    assert True