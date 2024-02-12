from pathlib import Path
import grade_predictor.metadata.shared as shared


RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "moonGen_2016"
RAW_DATA_FILENAME = "moonGen_scrape_2016_final.pkl"
DATA_DIRNAME = Path(__file__).resolve().parents[3] / "data"
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / "processed"

NUM_SPECIAL_TOKENS = 0

MIN_REPEATS = 2
MIN_GRADE_COUNT = 50
MAX_START_HOLDS = 2
MIN_MID_HOLDS = 2
MAX_MID_HOLDS = 11
MAX_END_HOLDS = 2
