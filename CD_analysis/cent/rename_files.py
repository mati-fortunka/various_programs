import os


def rename_files_in_folder(folder_path, file_extension):
    """
    Renames files in the specified folder by removing the last four characters from the base name,
    while keeping the file extension intact.

    :param folder_path: Path to the folder containing the files
    :param file_extension: File extension to filter files (e.g., '.bka')
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            # Separate the base name and extension
            base_name, ext = os.path.splitext(filename)

            # Skip renaming if the base name has less than four characters
            if len(base_name) > 4:
                # Create the new base name by removing the last four characters
                new_base_name = base_name[:-4]
                new_filename = f"{new_base_name}{ext}"

                # Get full paths for renaming
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")


# Specify the folder path and file extension
folder_path = "/home/matifortunka/Documents/JS/CD/Mateusz_Fuzja/unfolding/domierzanie"  # Replace with the actual folder path
file_extension = ".bka"  # Replace with the desired file extension

# Call the function to rename files
rename_files_in_folder(folder_path, file_extension)
