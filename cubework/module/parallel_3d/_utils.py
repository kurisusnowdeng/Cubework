from cubework.global_vars import env


def get_depth_from_env():
    return env.depth_3d


def get_input_parallel_mode():
    return env.input_group_3d


def get_weight_parallel_mode():
    return env.weight_group_3d


def get_output_parallel_mode():
    return env.output_group_3d


def swap_in_out_group():
    env.input_group_3d, env.output_group_3d = env.output_group_3d, env.input_group_3d
