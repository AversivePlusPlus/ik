from kinematics import *
from generator import *
import blender_utils
import sympy
import sys

if len(sys.argv) < 2:
    print("usage : " + sys.argv[0] + " <blend_file> [<endpoint>]")
    exit()

x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")

blender_utils.call_blender_export(sys.argv[1])

if len(sys.argv) == 2:
    l = blender_utils.read_json("blender.out.json")
    for e in l:
        print(e["name"])
elif len(sys.argv) == 3:
    endpoint = sys.argv[2].replace("/", "_")
    chain = blender_utils.extract_chain(
        blender_utils.read_json("blender.out.json"),
        sys.argv[2]
    )
    chain.name = chain.name.replace("/", "_")
    cpp = KinematicsCpp(chain)
    f = open(endpoint + ".cpp", "w+")
    f.write(str(cpp))
    f.close()
