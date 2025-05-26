# File: ml5_program_robot.py

import xml.etree.ElementTree as ET
import shutil

def merge_ml5_files(part1_path, part2_path, output_path):
    tree1 = ET.parse(part1_path)
    tree2 = ET.parse(part2_path)

    root1 = tree1.getroot()
    root2 = tree2.getroot()

    method1 = root1.find('Method')
    method2 = root2.find('Method')

    exec_steps1 = method1.find('ExecutableSmartSteps')
    exec_steps2 = method2.find('ExecutableSmartSteps')

    wrappers1 = exec_steps1.findall('SmartStepWrapper')
    wrappers2 = exec_steps2.findall('SmartStepWrapper')

    init_wrapper = wrappers1[0]  # Initialization
    part1_wrapper = wrappers1[1]  # Serial Dispense Part 1
    part2_wrapper = wrappers2[1]  # Serial Dispense Part 2

    # Update IDs for merged context
    part1_wrapper.find('ID').text = '2'
    part1_wrapper.find('ItemNumber').text = '2'

    part2_wrapper.find('ID').text = '3'
    part2_wrapper.find('ItemNumber').text = '3'
    part2_wrapper.find('SmartStep').find('ID').text = '3'
    part2_wrapper.find('SmartStep').find('Name').text = 'Serial Dispense Part 2'
    part2_wrapper.find('SmartStep').find('Comment').text = 'Continued serial dispense procedure.'

    # Clear and insert
    exec_steps1.clear()
    exec_steps1.extend([init_wrapper, part1_wrapper, part2_wrapper])

    # Write merged file
    ET.ElementTree(root1).write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Merged file written to: {output_path}")

# Use the generated part files
merge_ml5_files('buf_01.ml5', 'buf_02.ml5', 'merged_buf_0.ml5')
