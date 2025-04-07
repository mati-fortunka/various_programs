import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    parsed = minidom.parseString(rough_string)
    return parsed.toprettyxml(indent="")


def create_ml5_method(fill_volume: int, speed: int = 5) -> str:
    method_doc = ET.Element("MethodDoc")

    def add_step(smart_steps, step_id: int, name: str, comment: str, valve_pos: str, aspirate: int):
        smart_step_wrapper = ET.SubElement(smart_steps, "SmartStepWrapper")
        ET.SubElement(smart_step_wrapper, "ID").text = str(step_id)
        ET.SubElement(smart_step_wrapper, "ItemNumber").text = str(step_id)
        ET.SubElement(smart_step_wrapper, "RunOnce").text = "0"
        ET.SubElement(smart_step_wrapper, "Loop").text = "1"

        smart_step = ET.SubElement(smart_step_wrapper, "SmartStep")
        ET.SubElement(smart_step, "ID").text = str(step_id)
        ET.SubElement(smart_step, "Type").text = "User"
        ET.SubElement(smart_step, "Name").text = name
        ET.SubElement(smart_step, "Comment").text = comment
        ET.SubElement(smart_step, "Reserved").text = "0"

        step = ET.SubElement(smart_step, "Step")
        ET.SubElement(step, "ComPort").text = "1"
        ET.SubElement(step, "InstAddr").text = "97"
        ET.SubElement(step, "InstType").text = "16"

        variable_delay = ET.SubElement(step, "Variable")
        delay = ET.SubElement(variable_delay, "DELAY")
        ET.SubElement(delay, "Name").text = "0"
        ET.SubElement(delay, "Mangle")
        ET.SubElement(delay, "Variable").text = "0"
        ET.SubElement(delay, "Value")
        ET.SubElement(delay, "VariableList")
        ET.SubElement(delay, "ApplicibleSmartStepWrapperList")

        ET.SubElement(step, "Trigger").text = "0"
        ET.SubElement(step, "TTL").text = "0"
        ET.SubElement(step, "TTLDirection").text = "2"

        for side_name in ["Left", "Right"]:
            side = ET.SubElement(step, side_name)

            valve_var = ET.SubElement(side, "Variable")
            valve = ET.SubElement(valve_var, "VALVE")
            ET.SubElement(valve, "Name").text = valve_pos
            ET.SubElement(valve, "Position").text = valve_pos

            ET.SubElement(side, "ValveDirection").text = "2"

            volume_var = ET.SubElement(side, "Variable")
            volume = ET.SubElement(volume_var, "VOLUME")
            ET.SubElement(volume, "Name").text = str(fill_volume)
            ET.SubElement(volume, "Mangle")
            ET.SubElement(volume, "Variable").text = "0"
            ET.SubElement(volume, "Value")
            ET.SubElement(volume, "VariableList")
            ET.SubElement(volume, "ApplicibleSmartStepWrapperList")

            ET.SubElement(side, "Aspirate").text = str(aspirate)

            delay_var = ET.SubElement(side, "Variable")
            delay = ET.SubElement(delay_var, "DELAY")
            ET.SubElement(delay, "Name").text = "0"
            ET.SubElement(delay, "Mangle")
            ET.SubElement(delay, "Variable").text = "0"
            ET.SubElement(delay, "Value")
            ET.SubElement(delay, "VariableList")
            ET.SubElement(delay, "ApplicibleSmartStepWrapperList")

            speed_var = ET.SubElement(side, "Variable")
            syringe_speed = ET.SubElement(speed_var, "SYRINGE_SPEED")
            ET.SubElement(syringe_speed, "Name").text = str(speed)
            ET.SubElement(syringe_speed, "Mangle")
            ET.SubElement(syringe_speed, "Variable").text = "0"
            ET.SubElement(syringe_speed, "Value")
            ET.SubElement(syringe_speed, "VariableList")
            ET.SubElement(syringe_speed, "ApplicibleSmartStepWrapperList")

            ET.SubElement(side, "Counter").text = "0"

    method = ET.SubElement(method_doc, "Method")
    ET.SubElement(method, "Name")
    executable_steps = ET.SubElement(method, "ExecutableSmartSteps")

    # Initialization Step
    init_wrapper = ET.SubElement(executable_steps, "SmartStepWrapper")
    ET.SubElement(init_wrapper, "ID").text = "1"
    ET.SubElement(init_wrapper, "ItemNumber").text = "1"
    ET.SubElement(init_wrapper, "RunOnce").text = "1"
    ET.SubElement(init_wrapper, "Loop").text = "1"

    init_step = ET.SubElement(init_wrapper, "SmartStep")
    ET.SubElement(init_step, "ID").text = "4"
    ET.SubElement(init_step, "Type").text = "Hamilton"
    ET.SubElement(init_step, "Name").text = "Initialize"
    ET.SubElement(init_step, "Comment").text = "Initializes all instruments on the Daisy Chain."
    ET.SubElement(init_step, "Reserved").text = "1"

    # Fill Step
    add_step(executable_steps, 2, "Fill", "enter a comment here", "Position 1", 1)
    # Dispense Step
    add_step(executable_steps, 3, "Dispense", "enter a comment here", "Position 2", 0)

    ET.SubElement(method, "EmbeddedSmartSteps")
    return prettify(method_doc)


# Example Usage
xml_result = create_ml5_method(fill_volume=1000, speed=5)
print(xml_result)
