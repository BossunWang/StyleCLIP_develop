from utils import ensure_checkpoint_exists
from mapper.scripts.inference import run
import os


def main():
    meta_data = {
        'afro': ['afro', False, False, True],
        'angry': ['angry', False, False, True],
        'Beyonce': ['beyonce', False, False, False],
        'bobcut': ['bobcut', False, False, True],
        'bowlcut': ['bowlcut', False, False, True],
        'curly hair': ['curly_hair', False, False, True],
        'Hilary Clinton': ['hilary_clinton', False, False, False],
        'Jhonny Depp': ['depp', False, False, False],
        'mohawk': ['mohawk', False, False, True],
        'purple hair': ['purple_hair', False, False, False],
        'surprised': ['surprised', False, False, True],
        'Taylor Swift': ['taylor_swift', False, False, False],
        'trump': ['trump', False, False, False],
        'Mark Zuckerberg': ['zuckerberg', False, False, False]
    }

    edit_type = 'surprised'  # @param ['afro', 'angry', 'Beyonce', 'bobcut', 'bowlcut', 'curly hair', 'Hilary Clinton', 'Jhonny Depp', 'mohawk', 'purple hair', 'surprised', 'Taylor Swift', 'trump', 'Mark Zuckerberg']
    edit_id = meta_data[edit_type][0]
    os.makedirs("mapper/pretrained", exist_ok=True)
    ensure_checkpoint_exists(f"mapper/pretrained/{edit_id}.pt")
    latent_path = "example_celebs.pt"  # @param {type:"string"}
    if latent_path == "example_celebs.pt":
        ensure_checkpoint_exists("example_celebs.pt")
    n_images = 1  # @param


if __name__ == '__main__':
    main()
