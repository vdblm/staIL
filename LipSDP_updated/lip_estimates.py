import os
rootdir = 'lip_constant_estimate'

for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    weight_path = os.path.join(subdir, 'base.mat')
    os.system("python solve_sdp.py --form neuron --weight-path "+weight_path)