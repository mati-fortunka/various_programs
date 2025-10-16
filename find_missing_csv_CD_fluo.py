import os


def find_dsx_without_csv(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Utwórz zbiór nazw plików (bez rozszerzeń) dla szybszego sprawdzania
        base_names = {os.path.splitext(f)[0] for f in filenames}

        for filename in filenames:
            if filename.endswith('.FBKN'):
                base_name = os.path.splitext(filename)[0]
                csv_file = base_name + '.csv'
                if csv_file not in filenames:
                    print(os.path.join(dirpath, filename))


if __name__ == "__main__":
    root_directory = input("Podaj ścieżkę do folderu startowego: ").strip()
    if os.path.isdir(root_directory):
        find_dsx_without_csv(root_directory)
    else:
        print("Podana ścieżka nie istnieje lub nie jest folderem.")
