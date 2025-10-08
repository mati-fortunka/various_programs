import re
import os

def prodata_to_csv(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_lines = []
    inside_table = False
    skip_header = False

    for line in lines:
        line = line.strip()

        # Detect start of a new property section (e.g., "CircularDichroism" or "HV")
        if re.match(r"^[A-Za-z]", line) and not line.startswith("Time"):
            inside_table = True
            skip_header = True
            continue

        # Skip the "Time,Wavelength" header and the line with just ",220,"
        if skip_header:
            if line.startswith("Time") or line.startswith(","):
                continue
            else:
                skip_header = False

        # End of table if empty line
        if inside_table and line == "":
            inside_table = False
            continue

        # Collect data rows (numeric rows with commas)
        if inside_table and "," in line:
            output_lines.append(line)

    # Write result to CSV
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Converted: {input_file} → {output_file}")


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            prodata_to_csv(input_path, output_path)


# Example usage
if __name__ == "__main__":
    input_dir = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/CD/cd_kin_2000s_final"   # replace with your input folder
    output_dir = "/home/matifortunka/Documents/JS/data_Cambridge/6_3/paper/CD/cd_kin_2000s_final/simple"  # where to save transformed CSVs
    process_folder(input_dir, output_dir)
