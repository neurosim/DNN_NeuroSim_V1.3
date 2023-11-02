import os

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = f'{THIS_SCRIPT_DIR}/Inference_pytorch/neurosim_out.txt'


def get_dynamic_energy(of: str) -> dict:
    lines = open(FILE, 'r').readlines()
    layer2energy = {}
    curr_layer = 'None'
    for line in lines:
        if 'Estimation of Layer' in line:
            curr_layer = line.strip().split(' ')[-2]
        if 'readDynamicEnergy' in line and of in line:
            layer2energy[curr_layer] = float(line.strip().split(' ')[-1][:-2])
        if 'Summary' in line:
            break
    return layer2energy


if __name__ == '__main__':
    print(f'Analog Energy (pJ)')
    for k, v in get_dynamic_energy('ADC').items():
        print(f'{v}')
