import pickle
from typing import Dict

def load_skeleton(skel_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(skel_file, 'rb') as handle:
        skel_dict = pickle.load(handle)

    return skel_dict

def save_skeleton(skel_dict) -> None:
            """
            Writes the currently built skeleton to a pickle file
            """
            with open("new_human.pickle", 'wb') as f:
                pickle.dump(skel_dict, f)

if __name__ == "__main__":
    skelly = load_skeleton("new_human.pickle")
    parts = ["ankle1", "knee1", "hip1", "hip2", "knee2", "ankle2", "wrist1", "elbow1", "shoulder1", "shoulder2",
            "elbow2", "wrist2", "chin", "forehead", "neck"]
    skelly["markers"] = parts
    #save_skeleton(skelly)
    print(skelly)