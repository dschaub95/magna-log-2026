#include <Python.h>
#include <numpy/arrayobject.h>

// Forward declarations for env-specific functions supplied by user
static int my_log(PyObject *dict, Log *log, int include_init);
static int my_init(Env *env, PyObject *args, PyObject *kwargs);

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs);
#ifndef MY_SHARED
static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return NULL;
}
#endif

static PyObject *my_get(PyObject *dict, Env *env);
#ifndef MY_GET
static PyObject *my_get(PyObject *dict, Env *env)
{
    return NULL;
}
#endif

static int my_put(Env *env, PyObject *args, PyObject *kwargs);
#ifndef MY_PUT
static int my_put(Env *env, PyObject *args, PyObject *kwargs)
{
    return 0;
}
#endif

#ifndef MY_METHODS
#define MY_METHODS {NULL, NULL, 0, NULL}
#endif

static int require_dtype(PyArrayObject *array, int type, const char *name, const char *dtype)
{
    if (PyArray_TYPE(array) == type)
    {
        return 0;
    }
    PyErr_Format(PyExc_ValueError, "%s must have dtype %s", name, dtype);
    return 1;
}

static Env *unpack_env(PyObject *args)
{
    PyObject *handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    Env *env = (Env *)PyLong_AsVoidPtr(handle_obj);
    if (!env)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }

    return env;
}

// Python function to initialize the environment
static PyObject *env_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (PyTuple_Size(args) != 6)
    {
        PyErr_SetString(PyExc_TypeError, "Environment requires 5 arguments");
        return NULL;
    }

    Env *env = (Env *)calloc(1, sizeof(Env));
    if (!env)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
        return NULL;
    }

    PyObject *obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations))
    {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    if (require_dtype(observations, NPY_FLOAT32, "Observations", "float32"))
    {
        return NULL;
    }
    env->observations = (float *)PyArray_DATA(observations);

    PyObject *act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions))
    {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    if (require_dtype(actions, NPY_INT32, "Actions", "int32"))
    {
        return NULL;
    }
    env->actions = (int *)PyArray_DATA(actions);

    PyObject *rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards))
    {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (require_dtype(rewards, NPY_FLOAT32, "Rewards", "float32"))
    {
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }
    env->rewards = (float *)PyArray_DATA(rewards);

    PyObject *term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals))
    {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (require_dtype(terminals, NPY_BOOL, "Terminals", "bool"))
    {
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }
    env->terminals = (unsigned char *)PyArray_DATA(terminals);

    PyObject *trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *truncations = (PyArrayObject *)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations))
    {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }
    // env->truncations = PyArray_DATA(truncations);

    PyObject *seed_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_arg);

    // Assumes each process has the same number of environments
    // Add 1 to avoid glibc bug where srand(0) is converted to srand(1)
    srand(seed + 1);

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL)
    {
        kwargs = PyDict_New();
    }
    else
    {
        Py_INCREF(kwargs); // We need to increment the reference since we'll be
                           // modifying it
    }

    // Add the seed to kwargs
    PyObject *py_seed = PyLong_FromLong(seed);
    if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
        Py_DECREF(py_seed);
        Py_DECREF(kwargs);
        return NULL;
    }
    Py_DECREF(py_seed);

    PyObject *empty_args = PyTuple_New(0);
    my_init(env, empty_args, kwargs);
    Py_DECREF(kwargs);
    if (PyErr_Occurred())
    {
        return NULL;
    }

    return PyLong_FromVoidPtr(env);
}

// Python function to reset the environment
static PyObject *env_reset(PyObject *self, PyObject *args)
{
    if (PyTuple_Size(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "env_reset requires 2 arguments");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }
    if (c_reset(env))
    {
        PyErr_SetString(PyExc_ValueError, "reset positions must form one connected component");
        return NULL;
    }
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject *env_step(PyObject *self, PyObject *args)
{
    int num_args = PyTuple_Size(args);
    if (num_args != 1)
    {
        PyErr_SetString(PyExc_TypeError, "vec_render requires 1 argument");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }
    if (c_step(env))
    {
        PyErr_SetString(PyExc_ValueError, "reset positions must form one connected component");
        return NULL;
    }
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject *env_render(PyObject *self, PyObject *args)
{
    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }
    c_render(env);
    Py_RETURN_NONE;
}

static PyObject *env_capture(PyObject *self, PyObject *args)
{
    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }

    npy_intp dims[3] = {env->capture_size, env->capture_size, 4};
    PyObject *array = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!array)
    {
        return NULL;
    }
    unsigned char *out = (unsigned char *)PyArray_DATA((PyArrayObject *)array);
    c_capture(env, out);
    return array;
}

// Python function to close the environment
static PyObject *env_close(PyObject *self, PyObject *args)
{
    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }
    c_close(env);
    free(env);
    Py_RETURN_NONE;
}

static PyObject *env_get(PyObject *self, PyObject *args)
{
    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }
    PyObject *dict = PyDict_New();
    if (!dict)
    {
        return NULL;
    }
    my_get(dict, env);
    if (PyErr_Occurred())
    {
        Py_DECREF(dict);
        return NULL;
    }
    return dict;
}

static PyObject *env_put(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int num_args = PyTuple_Size(args);
    if (num_args != 1)
    {
        PyErr_SetString(PyExc_TypeError, "env_put requires 1 positional argument");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env)
    {
        return NULL;
    }

    PyObject *empty_args = PyTuple_New(0);
    my_put(env, empty_args, kwargs);
    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

typedef struct
{
    Env **envs;
    int num_envs;
} VecEnv;

static VecEnv *unpack_vecenv(PyObject *args)
{
    PyObject *handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    VecEnv *vec = (VecEnv *)PyLong_AsVoidPtr(handle_obj);
    if (!vec)
    {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }

    if (vec->num_envs <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }

    return vec;
}

static PyObject *vec_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (PyTuple_Size(args) != 7)
    {
        PyErr_SetString(PyExc_TypeError, "vec_init requires 6 arguments");
        return NULL;
    }

    VecEnv *vec = (VecEnv *)calloc(1, sizeof(VecEnv));
    if (!vec)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }
    PyObject *num_envs_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(num_envs_arg, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "num_envs must be an integer");
        return NULL;
    }
    int num_envs = PyLong_AsLong(num_envs_arg);
    if (num_envs <= 0)
    {
        PyErr_SetString(PyExc_TypeError, "num_envs must be greater than 0");
        return NULL;
    }
    vec->num_envs = num_envs;
    vec->envs = (Env **)calloc(num_envs, sizeof(Env *));
    if (!vec->envs)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    PyObject *seed_obj = PyTuple_GetItem(args, 6);
    if (!PyObject_TypeCheck(seed_obj, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_obj);

    PyObject *obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations))
    {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    if (require_dtype(observations, NPY_FLOAT32, "Observations", "float32"))
    {
        return NULL;
    }
    if (PyArray_NDIM(observations) < 2)
    {
        PyErr_SetString(PyExc_ValueError, "Batched Observations must be at least 2D");
        return NULL;
    }

    PyObject *act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions))
    {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    if (require_dtype(actions, NPY_INT32, "Actions", "int32"))
    {
        return NULL;
    }

    PyObject *rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards))
    {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (require_dtype(rewards, NPY_FLOAT32, "Rewards", "float32"))
    {
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }

    PyObject *term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals))
    {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (require_dtype(terminals, NPY_BOOL, "Terminals", "bool"))
    {
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }

    PyObject *trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *truncations = (PyArrayObject *)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations))
    {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL)
    {
        kwargs = PyDict_New();
    }
    else
    {
        Py_INCREF(kwargs); // We need to increment the reference since we'll be
                           // modifying it
    }

    for (int i = 0; i < num_envs; i++)
    {
        Env *env = (Env *)calloc(1, sizeof(Env));
        if (!env)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
            Py_DECREF(kwargs);
            return NULL;
        }
        vec->envs[i] = env;

        // // Make sure the log is initialized to 0
        memset(&env->log, 0, sizeof(Log));

        env->observations =
            (float *)((char *)PyArray_DATA(observations) + i * PyArray_STRIDE(observations, 0));
        env->actions = (int *)((char *)PyArray_DATA(actions) + i * PyArray_STRIDE(actions, 0));
        env->rewards = (float *)((char *)PyArray_DATA(rewards) + i * PyArray_STRIDE(rewards, 0));
        env->terminals =
            (unsigned char *)((char *)PyArray_DATA(terminals) + i * PyArray_STRIDE(terminals, 0));
        // env->truncations = (void*)((char*)PyArray_DATA(truncations) +
        // i*PyArray_STRIDE(truncations, 0));

        // Assumes each process has the same number of environments
        // Add 1 to avoid glibc bug where srand(0) is converted to srand(1)
        int env_seed = i + seed * vec->num_envs + 1;
        srand(env_seed);

        // Add the seed to kwargs for this environment
        PyObject *py_seed = PyLong_FromLong(env_seed);
        if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
            Py_DECREF(py_seed);
            Py_DECREF(kwargs);
            return NULL;
        }
        Py_DECREF(py_seed);

        PyObject *empty_args = PyTuple_New(0);
        my_init(env, empty_args, kwargs);
        if (PyErr_Occurred())
        {
            return NULL;
        }
    }

    Py_DECREF(kwargs);
    return PyLong_FromVoidPtr(vec);
}

// Python function to close the environment
static PyObject *vectorize(PyObject *self, PyObject *args)
{
    int num_envs = PyTuple_Size(args);
    if (num_envs == 0)
    {
        PyErr_SetString(PyExc_TypeError, "make_vec requires at least 1 env id");
        return NULL;
    }

    VecEnv *vec = (VecEnv *)calloc(1, sizeof(VecEnv));
    if (!vec)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->envs = (Env **)calloc(num_envs, sizeof(Env *));
    if (!vec->envs)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->num_envs = num_envs;
    for (int i = 0; i < num_envs; i++)
    {
        PyObject *handle_obj = PyTuple_GetItem(args, i);
        if (!PyObject_TypeCheck(handle_obj, &PyLong_Type))
        {
            PyErr_SetString(PyExc_TypeError, "Env ids must be integers. Pass them as separate args "
                                             "with *env_ids, not as a list.");
            return NULL;
        }
        vec->envs[i] = (Env *)PyLong_AsVoidPtr(handle_obj);
    }

    return PyLong_FromVoidPtr(vec);
}

static PyObject *vec_reset(PyObject *self, PyObject *args)
{
    if (PyTuple_Size(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "vec_reset requires 2 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec)
    {
        return NULL;
    }

    PyObject *seed_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_arg);

    for (int i = 0; i < vec->num_envs; i++)
    {
        // Assumes each process has the same number of environments
        // Add 1 to avoid glibc bug where srand(0) is converted to srand(1)
        int env_seed = i + seed * vec->num_envs + 1;
        srand(env_seed);
        if (c_reset(vec->envs[i]))
        {
            PyErr_SetString(PyExc_ValueError, "reset positions must form one connected component");
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyObject *vec_step(PyObject *self, PyObject *arg)
{
    int num_args = PyTuple_Size(arg);
    if (num_args != 1)
    {
        PyErr_SetString(PyExc_TypeError, "vec_step requires 1 argument");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(arg);
    if (!vec)
    {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++)
    {
        if (c_step(vec->envs[i]))
        {
            PyErr_SetString(PyExc_ValueError, "reset positions must form one connected component");
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyObject *vec_render(PyObject *self, PyObject *args)
{
    int num_args = PyTuple_Size(args);
    if (num_args != 2)
    {
        PyErr_SetString(PyExc_TypeError, "vec_render requires 2 arguments");
        return NULL;
    }

    VecEnv *vec = (VecEnv *)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }

    PyObject *env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);

    c_render(vec->envs[env_id]);
    Py_RETURN_NONE;
}

static PyObject *vec_capture(PyObject *self, PyObject *args)
{
    int num_args = PyTuple_Size(args);
    if (num_args != 2)
    {
        PyErr_SetString(PyExc_TypeError, "vec_capture requires 2 arguments");
        return NULL;
    }

    VecEnv *vec = (VecEnv *)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }

    PyObject *env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type))
    {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);
    if (env_id < 0 || env_id >= vec->num_envs)
    {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    Env *env = vec->envs[env_id];
    npy_intp dims[3] = {env->capture_size, env->capture_size, 4};
    PyObject *array = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!array)
    {
        return NULL;
    }
    unsigned char *out = (unsigned char *)PyArray_DATA((PyArrayObject *)array);
    c_capture(env, out);
    return array;
}

static int assign_to_dict(PyObject *dict, char *key, float value)
{
    PyObject *v = PyFloat_FromDouble(value);
    if (v == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Failed to convert log value");
        return 1;
    }
    if (PyDict_SetItemString(dict, key, v) < 0)
    {
        PyErr_SetString(PyExc_TypeError, "Failed to set log value");
        return 1;
    }
    Py_DECREF(v);
    return 0;
}

static PyObject *vec_log(PyObject *self, PyObject *args)
{
    VecEnv *vec = unpack_vecenv(args);
    if (!vec)
    {
        return NULL;
    }

    // Iterates over logs one float at a time. Will break
    // horribly if Log has non-float data.
    Log aggregate = {0};
    int num_keys = sizeof(Log) / sizeof(float);
    int include_init = vec->num_envs > 0 && vec->envs[0]->reset_mode == RESET_MODE_INIT;
    float init_sum = 0.0f;
    float error_min = 0.0f;
    float error_max = 0.0f;
    int error_count = 0;
    for (int i = 0; i < vec->num_envs; i++)
    {
        Env *env = vec->envs[i];
        if (include_init)
            init_sum += env->error_init;
        if (include_init && env->log.n > 0.0f)
        {
            float error = env->log.error / env->log.n;
            if (error_count == 0 || error < error_min)
                error_min = error;
            if (error_count == 0 || error > error_max)
                error_max = error;
            error_count++;
        }
        for (int j = 0; j < num_keys; j++)
        {
            ((float *)&aggregate)[j] += ((float *)&env->log)[j];
            ((float *)&env->log)[j] = 0.0f;
        }
    }

    PyObject *dict = PyDict_New();
    if (!dict)
    {
        return NULL;
    }
    if (include_init)
    {
        float init_mean = init_sum / (float)vec->num_envs;
        if (assign_to_dict(dict, "error_init", init_mean))
        {
            Py_DECREF(dict);
            return NULL;
        }
    }

    if (aggregate.n == 0.0f)
    {
        return dict;
    }

    float n = aggregate.n;
    for (int i = 0; i < num_keys; i++)
    {
        ((float *)&aggregate)[i] /= n;
    }

    if (my_log(dict, &aggregate, include_init) ||
        (include_init && (assign_to_dict(dict, "error_min", error_min) ||
                          assign_to_dict(dict, "error_max", error_max))) ||
        assign_to_dict(dict, "n", n))
    {
        Py_DECREF(dict);
        return NULL;
    }

    return dict;
}

static PyObject *vec_close(PyObject *self, PyObject *args)
{
    VecEnv *vec = unpack_vecenv(args);
    if (!vec)
    {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++)
    {
        c_close(vec->envs[i]);
        free(vec->envs[i]);
    }
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

static double unpack(PyObject *kwargs, char *key)
{
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (val == NULL)
    {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Missing required keyword argument '%s'", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
        return 1;
    }
    if (PyLong_Check(val))
    {
        long out = PyLong_AsLong(val);
        if (out > INT_MAX || out < INT_MIN)
        {
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg),
                     "Value %ld of integer argument %s is out of range", out, key);
            PyErr_SetString(PyExc_TypeError, error_msg);
            return 1;
        }
        // Cast on return. Safe because double can represent all 32-bit ints exactly
        return out;
    }
    if (PyFloat_Check(val))
    {
        return PyFloat_AsDouble(val);
    }
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to unpack keyword %s as int", key);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return 1;
}

// Method table
static PyMethodDef methods[] = {
    {"env_init", (PyCFunction)env_init, METH_VARARGS | METH_KEYWORDS,
     "Init environment with observation, action, reward, terminal, truncation "
     "arrays"},
    {"env_reset", env_reset, METH_VARARGS, "Reset the environment"},
    {"env_step", env_step, METH_VARARGS, "Step the environment"},
    {"env_render", env_render, METH_VARARGS, "Render the environment"},
    {"env_capture", env_capture, METH_VARARGS, "Capture a rendered frame"},
    {"env_close", env_close, METH_VARARGS, "Close the environment"},
    {"env_get", env_get, METH_VARARGS, "Get the environment state"},
    {"env_put", (PyCFunction)env_put, METH_VARARGS | METH_KEYWORDS, "Put stuff into env"},
    {"vectorize", vectorize, METH_VARARGS, "Make a vector of environment handles"},
    {"vec_init", (PyCFunction)vec_init, METH_VARARGS | METH_KEYWORDS,
     "Initialize a vector of environments"},
    {"vec_reset", vec_reset, METH_VARARGS, "Reset the vector of environments"},
    {"vec_step", vec_step, METH_VARARGS, "Step the vector of environments"},
    {"vec_log", vec_log, METH_VARARGS, "Log the vector of environments"},
    {"vec_render", vec_render, METH_VARARGS, "Render the vector of environments"},
    {"vec_capture", vec_capture, METH_VARARGS, "Capture a rendered frame from the vector"},
    {"vec_close", vec_close, METH_VARARGS, "Close the vector of environments"},
    {"shared", (PyCFunction)my_shared, METH_VARARGS | METH_KEYWORDS, "Shared state"},
    MY_METHODS,
    {NULL, NULL, 0, NULL}};

// Module definition
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "binding", NULL, -1, methods};

PyMODINIT_FUNC PyInit_binding(void)
{
    import_array();
    return PyModule_Create(&module);
}
