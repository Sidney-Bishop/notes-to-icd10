import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph_reranker import ZPhraseMatcher

def test_z_matcher():
    m = ZPhraseMatcher()
    print(f"Loaded {len(m.phrases)} phrases")
    print("Sample:", list(m.phrases.items())[:3])

    result = m.match("Patient here for WELL-CHILD check")
    print("Match result:", result)

    assert result is not None, "No match found!"
    assert result[1] == "Z00.129"
    print("✓ All Z-matcher tests pass")

if __name__ == "__main__":
    test_z_matcher()