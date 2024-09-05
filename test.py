import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default=False, action='store_true')
    parser.add_argument("--env", default=False, action='store_true')
    parser.add_argument("--sim", default=False, action='store_true')
    parser.add_argument("--ar", default=False, action='store_true')
    parser.add_argument("--clock", default=False, action='store_true')
    parser.add_argument("--nn", default=False, action='store_true')
    parser.add_argument("--mirror", default=False, action='store_true')
    parser.add_argument("--render", default=False, action='store_true')
    parser.add_argument("--timing", default=False, action='store_true')
    parser.add_argument("--all", default=False, action='store_true')
    parser.add_argument("--CI", default=False, action='store_true')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit()

    if args.all:
        args.algo = True
        args.env = True
        args.sim = True
        args.clock = True
        args.nn = True
        args.mirror = True
        args.render = True
        args.timing = True
    elif args.CI: # Tests run by CI
        args.algo = True
        args.env = True
        args.sim = False
        args.clock = True
        args.nn = True
        args.mirror = True
        args.render = False
        args.timing = True
        args.train = False # GLFW error on CI

    if args.algo:
        from testing.test_algo import test_all_algos
        test_all_algos()
    if args.env:
        import testing.test_env as test_env
        test_env.test_all_env()
    if args.sim:
        import testing.test_sim as test_sim
        test_sim.test_all_sim()
    if args.render:
        from testing.test_offscreen_render import test_offscreen_rendering, test_pointcloud_rendering
        test_offscreen_rendering()
        test_pointcloud_rendering()
    if args.clock:
        from testing.test_clock import test_all_clocks
        test_all_clocks()
    if args.nn:
        from testing.test_nn import test_nn
        test_nn()
    if args.mirror:
        from testing.test_mirror import test_mirror
        test_mirror()
    if args.timing:
        from testing.tracker_test import tracker_test
        from testing.sampling_speed import sampling_speed, run_PD_env_compare, run_model_compare
        sampling_speed()
        # run_PD_env_compare()
        # tracker_test()
        # run_model_compare()