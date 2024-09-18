import argparse
import sys
import numpy as np
import traceback

from duality.digit.digit_arm_diff_ik import DigitArmDiffInverseKinematics
from duality.digit.full_digit_ik import FullDigitInverseKinematics

from util.colors import OKGREEN, FAIL, ENDC

def test_pydrake_imports():
    try: 
        from pydrake.all import DiagramBuilder, RigidTransform, MultibodyPlant
        from pydrake.geometry import PenetrationAsPointPair 
        from pydrake.geometry.optimization import HPolyhedron, VPolytope

    except ImportError:
        print(f"{FAIL} failed to import pydrake functions")
        sys.exit()

    print(f"{OKGREEN}Passed all duality package import tests \u2713{ENDC}")


def test_duality_digit_ik_constructors():
    try: 
        from pydrake.all import DiagramBuilder
        digit = DigitArmDiffInverseKinematics()
        digit.wire_arm_V_G_tracking_diff_ik()
        assert(digit.plant is not None)
        assert(digit.plant_context is not None)
        digit = FullDigitInverseKinematics()
        digit.construct_digit_diagram()
        builder = DiagramBuilder()
        digit = FullDigitInverseKinematics()
        builder = digit.construct_digit_system(builder)

    except Exception:
        print(f"{FAIL} Duality digit IK test failed test with error:{ENDC}")
        print(traceback.format_exc())
        sys.exit()
    
    print(f"{OKGREEN}Passed duality diff IK function tests \u2713{ENDC}")

def test_all():
    test_pydrake_imports()
    test_duality_digit_ik_constructors()