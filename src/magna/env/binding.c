#include "env.h"

#define Env MultiCellEnv
#define MY_GET
#define MY_PUT
#include "env_binding.h"

static PyObject *my_get(PyObject *dict, Env *env)
{
    npy_intp dims[2] = {env->num_agents, env->spatial_dim};
    PyObject *positions = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!positions)
    {
        return NULL;
    }

    float *out = (float *)PyArray_DATA((PyArrayObject *)positions);
    for (int i = 0; i < env->num_agents; i++)
    {
        int offset = env->spatial_dim * i;
        out[offset] = env->agents[i].x;
        out[offset + 1] = env->agents[i].y;
        if (env->spatial_dim == 3)
            out[offset + 2] = env->agents[i].z;
    }
    if (PyDict_SetItemString(dict, "positions", positions) < 0)
    {
        Py_DECREF(positions);
        return NULL;
    }
    Py_DECREF(positions);
    return dict;
}

static int my_put(Env *env, PyObject *args, PyObject *kwargs)
{
    (void)args;
    PyObject *obj = kwargs ? PyDict_GetItemString(kwargs, "positions") : NULL;
    if (!obj)
    {
        PyErr_SetString(PyExc_KeyError, "positions");
        return -1;
    }
    float *data = (float *)PyArray_DATA((PyArrayObject *)obj);
    for (int i = 0; i < env->num_agents; i++)
    {
        int o = env->spatial_dim * i;
        env->agents[i].x = data[o];
        env->agents[i].y = data[o + 1];
        if (env->spatial_dim == 3)
            env->agents[i].z = data[o + 2];
    }
    build_neighbor_cache(env);
    compute_observations(env);
    return 0;
}

static int required_int(PyObject *kwargs, char *key, int *out)
{
    *out = unpack(kwargs, key);
    return PyErr_Occurred() ? 1 : 0;
}

static int required_float(PyObject *kwargs, char *key, float *out)
{
    *out = (float)unpack(kwargs, key);
    return PyErr_Occurred() ? 1 : 0;
}

static int required_bool(PyObject *kwargs, char *key, int *out)
{
    *out = PyObject_IsTrue(PyDict_GetItemString(kwargs, key));
    return PyErr_Occurred() ? 1 : 0;
}

static void *optional_array_data(PyObject *obj)
{
    if (obj == NULL || obj == Py_None)
    {
        return NULL;
    }
    return PyArray_DATA((PyArrayObject *)obj);
}

static float *kwarg_array_data(PyObject *kwargs, const char *key)
{
    PyObject *obj = PyDict_GetItemString(kwargs, key);
    return obj == NULL ? NULL : optional_array_data(obj);
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs)
{
    int verbose;
    if (required_int(kwargs, "verbose", &verbose))
    {
        return 1;
    }
    SetTraceLogLevel(verbose == 0 ? LOG_NONE : verbose);

    if (required_int(kwargs, "width", &env->width) ||
        required_int(kwargs, "height", &env->height) ||
        required_bool(kwargs, "bounded", &env->bounded) ||
        required_bool(kwargs, "circular", &env->circular) ||
        required_int(kwargs, "capture_size", &env->capture_size) ||
        required_int(kwargs, "num_agents", &env->num_agents) ||
        required_int(kwargs, "k", &env->k) ||
        required_float(kwargs, "perception_radius", &env->perception_radius) ||
        required_int(kwargs, "neighbor_graph", &env->neighbor_graph) ||
        required_int(kwargs, "reset_distribution", &env->reset_distribution) ||
        required_bool(kwargs, "fit_init_to_target_bbox", &env->fit_init_to_target_bbox) ||
        required_int(kwargs, "reset_mode", &env->reset_mode) ||
        required_int(kwargs, "depth", &env->depth) ||
        required_int(kwargs, "spatial_dim", &env->spatial_dim) ||
        required_float(kwargs, "coordinate_scale", &env->coordinate_scale) ||
        required_bool(kwargs, "mutual_obs_bit", &env->mutual_obs_bit) ||
        required_bool(kwargs, "target_neighbor_bit", &env->target_neighbor_bit) ||
        required_bool(kwargs, "perceived_ratio_obs", &env->perceived_ratio_obs) ||
        required_bool(kwargs, "density_grad_obs", &env->density_grad_obs) ||
        required_bool(kwargs, "density_grad_log_norm", &env->density_grad_log_norm) ||
        required_bool(kwargs, "local_frame_obs", &env->local_frame_obs) ||
        required_bool(kwargs, "self_pos_delta_obs", &env->self_pos_delta_obs) ||
        required_bool(kwargs, "boundary_dist_obs", &env->boundary_dist_obs) ||
        required_float(kwargs, "boundary_penalty", &env->boundary_penalty) ||
        required_bool(kwargs, "comm_select_action", &env->comm_select_action) ||
        required_bool(kwargs, "enforce_connected", &env->enforce_connected) ||
        required_float(kwargs, "disconnect_penalty", &env->disconnect_penalty) ||
        required_bool(kwargs, "terminate_on_disconnect", &env->terminate_on_disconnect) ||
        required_float(kwargs, "lam_nbr_reward", &env->lam_nbr_reward) ||
        required_float(kwargs, "reward_scaler", &env->reward_scaler) ||
        required_float(kwargs, "reward_min_distance", &env->reward_min_distance) ||
        required_int(kwargs, "reward_mode", &env->reward_mode) ||
        required_float(kwargs, "reward_interval_multiplier", &env->reward_interval_multiplier) ||
        required_bool(kwargs, "match_roles_once", &env->match_roles_once) ||
        required_float(kwargs, "max_speed", &env->max_speed) ||
        required_int(kwargs, "time_horizon", &env->time_horizon) ||
        required_bool(kwargs, "reuse_initial_positions", &env->reuse_initial_positions) ||
        required_bool(kwargs, "recenter_on_step", &env->recenter_on_step) ||
        required_float(kwargs, "agent_radius", &env->agent_radius) ||
        required_float(kwargs, "render_agent_radius", &env->render_agent_radius) ||
        required_float(kwargs, "collision_grid_size", &env->collision_grid_size) ||
        required_int(kwargs, "collision_iters", &env->collision_iters))
    {
        return 1;
    }
    env->initial_positions = NULL;
    env->initial_positions_set = 0;
    env->initial_positions_external = 0;
    env->initial_headings = NULL;
    env->cell_colors = NULL;
    env->cell_sprite_path = NULL;
    env->draw_cell_ids = 0;
    env->draw_obs_graph = 0;
    env->draw_perception_radius = 0;

    env->initial_positions = kwarg_array_data(kwargs, "initial_positions");
    env->initial_positions_set = env->initial_positions != NULL;
    env->initial_positions_external = env->initial_positions != NULL;
    env->initial_headings = kwarg_array_data(kwargs, "initial_headings");

    PyObject *cell_sprite_path_obj = PyDict_GetItemString(kwargs, "cell_sprite_path");
    if (cell_sprite_path_obj != NULL && cell_sprite_path_obj != Py_None)
    {
        const char *cell_sprite_path = PyUnicode_AsUTF8(cell_sprite_path_obj);
        if (cell_sprite_path == NULL)
        {
            return 1;
        }
        env->cell_sprite_path = strdup(cell_sprite_path);
        if (env->cell_sprite_path == NULL)
        {
            PyErr_NoMemory();
            return 1;
        }
    }

    env->cell_colors =
        (unsigned char *)optional_array_data(PyDict_GetItemString(kwargs, "cell_colors"));

    int draw_cell_ids;
    if (required_bool(kwargs, "draw_cell_ids", &draw_cell_ids))
    {
        return 1;
    }
    env->draw_cell_ids = (unsigned char)draw_cell_ids;

    int draw_obs_graph;
    if (required_bool(kwargs, "draw_obs_graph", &draw_obs_graph))
    {
        return 1;
    }
    env->draw_obs_graph = (unsigned char)draw_obs_graph;

    int draw_perception_radius;
    if (required_bool(kwargs, "draw_perception_radius", &draw_perception_radius))
    {
        return 1;
    }
    env->draw_perception_radius = (unsigned char)draw_perception_radius;

    PyObject *target_neighbors_obj = PyDict_GetItemString(kwargs, "target_neighbors");
    PyArrayObject *target_neighbors = (PyArrayObject *)target_neighbors_obj;
    env->target_neighbors = (int *)PyArray_DATA(target_neighbors);

    PyObject *target_neighbor_distances_obj =
        PyDict_GetItemString(kwargs, "target_neighbor_distances");
    PyArrayObject *target_neighbor_distances = (PyArrayObject *)target_neighbor_distances_obj;
    env->target_neighbor_distances = (float *)PyArray_DATA(target_neighbor_distances);
    env->target_neighbor_pair_distances =
        kwarg_array_data(kwargs, "target_neighbor_pair_distances");
    env->target_positions = kwarg_array_data(kwargs, "target_positions");
    if ((env->reset_mode == RESET_MODE_ROTATED_TARGET ||
         env->reset_mode == RESET_MODE_INTERPOLATED) &&
        env->target_positions == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "target_positions required for reset_mode");
        return 1;
    }

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log, int include_init)
{
    if (include_init &&
        (assign_to_dict(dict, "error", log->error) || assign_to_dict(dict, "sa_h0", log->sa_h0) ||
         assign_to_dict(dict, "sa_h1", log->sa_h1) || assign_to_dict(dict, "sa_h2", log->sa_h2) ||
         assign_to_dict(dict, "sa_h3", log->sa_h3) || assign_to_dict(dict, "sa_h4", log->sa_h4) ||
         assign_to_dict(dict, "sa_h5", log->sa_h5)))
        return 1;
    return assign_to_dict(dict, "episode_return", log->episode_return) ||
           assign_to_dict(dict, "episode_length", log->episode_length);
}
