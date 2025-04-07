# File: ml5_program_robot.py

import xml.etree.ElementTree as ET

MAX_VOLUME = 1000

def read_volumes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [
            tuple(map(lambda x: float(x.replace(',', '.')), line.strip().split()))
            for line in file if line.strip()
        ]

def group_by_syringe_volume(steps):
    groups = []
    current_group = []
    left_total = right_total = 0.0

    for left, right in steps:
        if (left_total + left > MAX_VOLUME or right_total + right > MAX_VOLUME) and current_group:
            groups.append(current_group)
            current_group = []
            left_total = right_total = 0.0

        current_group.append((left, right))
        left_total += left
        right_total += right

    if current_group:
        groups.append(current_group)

    return groups

def create_variable_node(parent, tag, name):
    node = ET.SubElement(parent, 'Variable')
    inner = ET.SubElement(node, tag)
    ET.SubElement(inner, 'Name').text = str(name)
    ET.SubElement(inner, 'Mangle')
    ET.SubElement(inner, 'Variable').text = '0'
    ET.SubElement(inner, 'Value')
    ET.SubElement(inner, 'VariableList')
    ET.SubElement(inner, 'ApplicibleSmartStepWrapperList')
    return node

def create_side_node(parent, side_tag, valve_pos, volume, aspirate):
    side = ET.SubElement(parent, side_tag)
    valv = ET.SubElement(side, 'Variable')
    v = ET.SubElement(valv, 'VALVE')
    ET.SubElement(v, 'Name').text = valve_pos
    ET.SubElement(v, 'Position').text = valve_pos

    ET.SubElement(side, 'ValveDirection').text = '2'
    create_variable_node(side, 'VOLUME', volume)
    ET.SubElement(side, 'Aspirate').text = '1' if aspirate else '0'
    create_variable_node(side, 'DELAY', 0)
    create_variable_node(side, 'SYRINGE_SPEED', 5)
    ET.SubElement(side, 'Counter').text = '0'
    return side

def create_method_xml(steps, filename):
    left_total = sum(step[0] for step in steps)
    right_total = sum(step[1] for step in steps)

    root = ET.Element('MethodDoc')
    com_ports = ET.SubElement(root, 'ComPorts')
    com_port = ET.SubElement(com_ports, 'ComPort')
    ET.SubElement(com_port, 'PortName').text = 'COM1:'
    inst_config = ET.SubElement(com_port, 'InstConfig')
    ET.SubElement(inst_config, 'Model').text = 'ML540/ML560'
    ET.SubElement(inst_config, 'FWVersion').text = 'CV01.01.A'
    ET.SubElement(inst_config, 'Type').text = '16'
    ET.SubElement(inst_config, 'Address').text = '97'
    ET.SubElement(inst_config, 'ComPort').text = '1'

    for side in ['L', 'R']:
        side_elem = ET.SubElement(inst_config, 'InstConfigSide')
        valve = ET.SubElement(side_elem, 'Valve')
        ET.SubElement(valve, 'Description').text = 'Dispenser'
        ET.SubElement(valve, 'Type').text = f'Dispenser{side}'
        ET.SubElement(valve, 'Protocol1Type').text = '7' if side == 'L' else '1'
        ET.SubElement(valve, 'PositionName').text = 'Position 1'
        ET.SubElement(valve, 'PositionName').text = 'Position 2'
        syringe = ET.SubElement(side_elem, 'Syringe')
        ET.SubElement(syringe, 'Description').text = '1000 ul syringe'
        ET.SubElement(syringe, 'Type').text = '1000'
        ET.SubElement(syringe, 'Protocol1Type').text = '0'
        ET.SubElement(syringe, 'StrokeSpeed').text = '5'

    global_config = ET.SubElement(root, 'GlobalConfig')
    ET.SubElement(global_config, 'RunContinusouly').text = '1'
    ET.SubElement(global_config, 'CountExecutions').text = '1'
    ET.SubElement(global_config, 'LogData').text = '1'
    ET.SubElement(global_config, 'LogFileName').text = 'Serial Dispense.txt'
    ET.SubElement(global_config, 'LoopCount').text = '1'
    ET.SubElement(root, 'Variables')

    method = ET.SubElement(root, 'Method')
    ET.SubElement(method, 'Name')
    exec_steps = ET.SubElement(method, 'ExecutableSmartSteps')

    # Initialization Step
    init = ET.SubElement(exec_steps, 'SmartStepWrapper')
    ET.SubElement(init, 'ID').text = '1'
    ET.SubElement(init, 'ItemNumber').text = '1'
    ET.SubElement(init, 'RunOnce').text = '1'
    ET.SubElement(init, 'Loop').text = '1'
    init_step = ET.SubElement(init, 'SmartStep')
    ET.SubElement(init_step, 'ID').text = '1'
    ET.SubElement(init_step, 'Type').text = 'Hamilton'
    ET.SubElement(init_step, 'Name').text = 'Initialize'
    ET.SubElement(init_step, 'Comment').text = 'Initializes all instruments.'
    ET.SubElement(init_step, 'Reserved').text = '1'

    # Serial Dispense Step Group
    wrapper = ET.SubElement(exec_steps, 'SmartStepWrapper')
    ET.SubElement(wrapper, 'ID').text = '2'
    ET.SubElement(wrapper, 'ItemNumber').text = '2'
    ET.SubElement(wrapper, 'RunOnce').text = '0'
    ET.SubElement(wrapper, 'Loop').text = '1'
    smart_step = ET.SubElement(wrapper, 'SmartStep')
    ET.SubElement(smart_step, 'ID').text = '2'
    ET.SubElement(smart_step, 'Type').text = 'User'
    ET.SubElement(smart_step, 'Name').text = 'Serial Dispense'
    ET.SubElement(smart_step, 'Comment').text = 'Performs a serial dispense procedure.'
    ET.SubElement(smart_step, 'Reserved').text = '0'

    # Priming Step - Load full volume before dispense
    priming = ET.SubElement(smart_step, 'Step')
    ET.SubElement(priming, 'ComPort').text = '1'
    ET.SubElement(priming, 'InstAddr').text = '97'
    ET.SubElement(priming, 'InstType').text = '16'
    create_variable_node(priming, 'DELAY', 0)
    ET.SubElement(priming, 'Trigger').text = '0'
    ET.SubElement(priming, 'TTL').text = '0'
    ET.SubElement(priming, 'TTLDirection').text = '2'
    create_side_node(priming, 'Left', 'Position 1', left_total, True)
    create_side_node(priming, 'Right', 'Position 1', right_total, True)

    # Actual Dispense Steps
    for left, right in steps:
        step = ET.SubElement(smart_step, 'Step')
        ET.SubElement(step, 'ComPort').text = '1'
        ET.SubElement(step, 'InstAddr').text = '97'
        ET.SubElement(step, 'InstType').text = '16'
        create_variable_node(step, 'DELAY', 0)
        ET.SubElement(step, 'Trigger').text = '1'
        ET.SubElement(step, 'TTL').text = '0'
        ET.SubElement(step, 'TTLDirection').text = '2'
        create_side_node(step, 'Left', 'Position 2', left, False)
        create_side_node(step, 'Right', 'Position 2', right, False)

    ET.SubElement(method, 'EmbeddedSmartSteps')
    ET.ElementTree(root).write(filename, encoding='utf-8', xml_declaration=True)

def process_volumes(file_path):
    volumes = read_volumes(file_path)
    groups = group_by_syringe_volume(volumes)
    for i, group in enumerate(groups, start=1):
        filename = f'{file_path[8:-4]}_{i}.ml5'
        create_method_xml(group, filename)
        print(f"Wrote {filename} with {len(group)} steps.")

process_volumes('volumes_prot_buf.txt')
process_volumes('volumes_water_den.txt')
