#include "libqhull_r/libqhull_r.h"
#include "raylib.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "env_types.h"
#include "utils.h"

struct Client
{
    Texture2D sprite;
    RenderTexture2D capture_target;
};

#define NEIGHBOR_GRAPH_RADIUS_TOPK 0
#define NEIGHBOR_GRAPH_DELAUNAY_TOPK 1
#define RESET_DISTRIBUTION_SUNFLOWER 0
#define RESET_DISTRIBUTION_GAUSSIAN 1
#define RESET_MODE_INIT 0
#define RESET_MODE_ROTATED_TARGET 1
#define RESET_MODE_INTERPOLATED 2

#include "rewards.h"

/* Sunflower init: scale bbox to this fraction of the arena (rest is border slack).*/
#define SUNFLOWER_FIT_FRACTION 0.88f
#define BORDER_CLIP_OFFSET 1.0f
/* Physical cell radius (um); median sunflower spacing uses 3 * this * coordinate_scale (env
 * units).*/
#define SUNFLOWER_CELL_RADIUS_MICRONS 4.0f

#define DEFAULT_CELL_FILL_COLOR (Color){222, 228, 232, 255}
#define DEFAULT_CELL_OUTLINE_COLOR (Color){118, 126, 134, 255}
#define DEFAULT_CIRCULAR_OUTSIDE_COLOR (Color){243, 241, 235, 255}
#define DEFAULT_CIRCULAR_GLASS_COLOR (Color){252, 252, 249, 255}
#define DEFAULT_CIRCULAR_BORDER_COLOR (Color){112, 118, 122, 255}
#define DEFAULT_CIRCULAR_INTERIOR_COLOR (Color){220, 226, 210, 255}
#define DEFAULT_CIRCULAR_INTERIOR_EDGE_COLOR (Color){192, 201, 182, 255}
#define DEFAULT_CIRCULAR_INNER_RIM_COLOR (Color){190, 196, 198, 255}
#define DEFAULT_CIRCULAR_SHADOW_COLOR (Color){90, 98, 104, 44}
#define DEFAULT_OBSERVATION_EDGE_COLOR (Color){96, 108, 124, 200}
#define DEFAULT_PERCEPTION_RADIUS_OUTLINE_COLOR (Color){118, 130, 146, 72}

float clip(float val, float min, float max);

static inline float env_radius_limit(const MultiCellEnv *env)
{
    float radius = ((float)env->width * 0.5f) - BORDER_CLIP_OFFSET;
    return radius > 0.0f ? radius : 0.0f;
}

static inline float agent_boundary_distance(const MultiCellEnv *env, float x, float y, float z)
{
    float margin = BORDER_CLIP_OFFSET;
    if (!env->circular)
    {
        float xy = fminf(fminf(x - margin, (float)env->width - margin - x),
                         fminf(y - margin, (float)env->height - margin - y));
        return env->spatial_dim == 3 ? fminf(xy, fminf(z - margin, (float)env->depth - margin - z))
                                     : xy;
    }

    float cx = (float)env->width * 0.5f;
    float cy = (float)env->height * 0.5f;
    float dx = x - cx;
    float dy = y - cy;
    float dz = z - (float)env->depth * 0.5f;
    return env_radius_limit(env) -
           sqrtf(dx * dx + dy * dy + (env->spatial_dim == 3 ? dz * dz : 0.0f));
}

static inline void clip_agents(MultiCellEnv *env);

static inline void clear_comm_selected_targets(MultiCellEnv *env)
{
    if (env->comm_selected_target_cache != NULL)
    {
        size_t n = (size_t)env->num_agents * (size_t)env->k;
        for (size_t i = 0; i < n; i++)
        {
            env->comm_selected_target_cache[i] = -1;
        }
    }
}

static inline void cache_comm_selected_targets(MultiCellEnv *env)
{
    if (!env->comm_select_action)
    {
        return;
    }
    clear_comm_selected_targets(env);
    for (int i = 0; i < env->num_agents; i++)
    {
        int *neighbors = &env->obs_topk_idx_cache[i * env->k];
        int *targets = &env->comm_selected_target_cache[i * env->k];
        int *actions = &env->actions[i * env->action_dim + env->spatial_dim];
        for (int n = 0; n < env->k; n++)
        {
            if (actions[n])
            {
                targets[n] = neighbors[n];
            }
        }
    }
}

static inline float comm_selected_target(MultiCellEnv *env, int sender, int receiver)
{
    if (!env->comm_select_action)
    {
        return 0.0f;
    }
    int *targets = &env->comm_selected_target_cache[sender * env->k];
    for (int n = 0; n < env->k; n++)
    {
        if (targets[n] == receiver)
        {
            return 1.0f;
        }
    }
    return 0.0f;
}

void init(MultiCellEnv *env)
{
    env->agents = (Agent *)calloc(env->num_agents, sizeof(Agent));
    env->action_dim = env->spatial_dim + env->comm_select_action * env->k;
    if (env->reuse_initial_positions)
    {
        if (env->initial_positions == NULL && !env->initial_positions_external)
        {
            env->initial_positions =
                (float *)calloc((size_t)env->num_agents * (size_t)env->spatial_dim, sizeof(float));
        }
    }

    // Initialize collision detection parameters
    env->min_agent_distance = env->collision_iters > 0 ? env->agent_radius * 2.0f : 0.0f;

    // Compute grid dimensions
    env->grid_width = (int)ceilf((float)env->width / env->collision_grid_size);
    env->grid_h = (int)ceilf((float)env->height / env->collision_grid_size);

    // Allocate spatial grid arrays
    int grid_size = env->grid_width * env->grid_h;
    env->cell_head = (int *)calloc(grid_size, sizeof(int));
    env->agent_next = (int *)calloc(env->num_agents, sizeof(int));
    size_t state_size = (size_t)env->num_agents * (size_t)env->k;
    env->prev_target_matched = (unsigned char *)calloc(state_size, sizeof(unsigned char));
    env->prev_target_err = (float *)calloc(state_size, sizeof(float));
    env->prev_distance_bin = (int *)calloc((size_t)env->num_agents, sizeof(int));
    env->prev_distance_bin_split = (int *)calloc(state_size, sizeof(int));
    env->target_match_sig = NULL;
    env->match_agent_sig = NULL;
    env->match_costs = NULL;
    env->match_role_gap = NULL;
    env->match_role_order = NULL;
    env->matched_role_to_agent = NULL;
    env->prev_seen_distances = NULL;
    env->prev_visible_count = NULL;
    env->match_used = NULL;
    if (env->reward_mode == REWARD_MODE_INTERVAL_SPLIT_MATCHED)
    {
        int pair_count = pair_signature_len(env->k);
        int use_pair_sig = pair_count > 0 && env->target_neighbor_pair_distances != NULL;
        int sig_len = 1 + env->k + use_pair_sig * pair_count;
        env->target_match_sig =
            (float *)calloc((size_t)env->num_agents * (size_t)sig_len, sizeof(float));
        env->match_agent_sig =
            (float *)calloc((size_t)env->num_agents * (size_t)sig_len, sizeof(float));
        env->match_costs =
            (float *)calloc((size_t)env->num_agents * (size_t)env->num_agents, sizeof(float));
        env->match_role_gap = (float *)calloc((size_t)env->num_agents, sizeof(float));
        env->match_role_order = (int *)calloc((size_t)env->num_agents, sizeof(int));
        env->matched_role_to_agent = (int *)calloc((size_t)env->num_agents, sizeof(int));
        env->prev_seen_distances = (float *)calloc(state_size, sizeof(float));
        env->prev_visible_count = (int *)calloc((size_t)env->num_agents, sizeof(int));
        env->match_used = (unsigned char *)calloc((size_t)env->num_agents, sizeof(unsigned char));
        for (int i = 0; i < env->num_agents; i++)
        {
            float *sig = &env->target_match_sig[i * sig_len];
            int count = 0;
            for (int t = 0; t < env->k; t++)
            {
                int offset = i * env->k + t;
                if (env->target_neighbors[offset] >= 0)
                {
                    sig[1 + count++] =
                        env->target_neighbor_distances[offset] / env->perception_radius;
                }
            }
            sig[0] = (float)count;
            sort_small_float_array(sig + 1, count);
            if (use_pair_sig)
            {
                float *pair_sig = sig + 1 + env->k;
                float *pair_dist = env->target_neighbor_pair_distances + i * pair_count;
                int pair = 0;
                int pair_offset = 0;
                for (int p = 0; p < env->k - 1; p++)
                {
                    float rp = env->target_neighbor_distances[i * env->k + p];
                    for (int q = p + 1; q < env->k; q++, pair_offset++)
                    {
                        float rq = env->target_neighbor_distances[i * env->k + q];
                        if (env->target_neighbors[i * env->k + p] >= 0 &&
                            env->target_neighbors[i * env->k + q] >= 0 && rp > 1e-6f &&
                            rq > 1e-6f && isfinite(pair_dist[pair_offset]))
                        {
                            pair_sig[pair++] = pair_cosine(
                                rp, rq, pair_dist[pair_offset] * pair_dist[pair_offset]);
                        }
                    }
                }
                sort_small_float_array(pair_sig, pair);
            }
        }
    }
    env->reward_state_ready =
        (unsigned char *)calloc((size_t)env->num_agents, sizeof(unsigned char));
    env->prev_obs_x = (float *)calloc((size_t)env->num_agents, sizeof(float));
    env->prev_obs_y = (float *)calloc((size_t)env->num_agents, sizeof(float));
    env->prev_obs_z = (float *)calloc((size_t)env->num_agents, sizeof(float));
    env->obs_topk_idx_cache = (int *)calloc(state_size, sizeof(int));
    env->obs_topk_d2_cache = (float *)calloc(state_size, sizeof(float));
    env->obs_perceived_count_cache = (int *)calloc((size_t)env->num_agents, sizeof(int));
    env->obs_density_grad_x = NULL;
    env->obs_density_grad_y = NULL;
    env->obs_density_grad_z = NULL;
    if (env->density_grad_obs)
    {
        env->obs_density_grad_x = (float *)calloc((size_t)env->num_agents, sizeof(float));
        env->obs_density_grad_y = (float *)calloc((size_t)env->num_agents, sizeof(float));
        env->obs_density_grad_z = (float *)calloc((size_t)env->num_agents, sizeof(float));
    }
    env->comm_selected_target_cache = NULL;
    if (env->comm_select_action)
    {
        env->comm_selected_target_cache = (int *)calloc(state_size, sizeof(int));
        clear_comm_selected_targets(env);
    }
    env->radius_neighbor_head = (int *)calloc((size_t)env->num_agents, sizeof(int));
    env->radius_neighbor_to = NULL;
    env->radius_neighbor_next = NULL;
    env->radius_neighbor_count = 0;
    env->radius_neighbor_capacity = 0;
    env->connected_seen = (unsigned char *)calloc((size_t)env->num_agents, sizeof(unsigned char));
    env->connected_queue = (int *)calloc((size_t)env->num_agents, sizeof(int));
}

void init_reset_points(int n, float *x_coords, float *y_coords, float *z_coords, float median_dist,
                       int permute, float center_x, float center_y, float center_z,
                       float avail_width, float avail_height, float avail_depth)
{
    int is_3d = z_coords != NULL;
    float golden_angle = PI * (3.0f - sqrtf(5.0f));
    float sqrt2 = is_3d ? sqrtf(2.0f) : 0.0f;
    // 2D keeps the old empirical sunflower spacing; 3D uses volume-per-point scaling.
    float radius = is_3d ? median_dist * cbrtf(3.0f * (float)n / (4.0f * PI))
                         : (median_dist / 1.787f) * sqrtf((float)n);

    for (int i = 0; i < n; i++)
    {
        float theta = (float)i * golden_angle;
        if (is_3d)
        {
            float fi = (float)i + 0.5f;
            float z = 1.0f - 2.0f * fi / (float)n;
            float xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
            float r = radius * cbrtf(fmodf(fi * sqrt2, 1.0f));
            x_coords[i] = r * xy * cosf(theta);
            y_coords[i] = r * xy * sinf(theta);
            z_coords[i] = r * z;
        }
        else
        {
            float r = radius * sqrtf((float)i / (float)n);
            x_coords[i] = r * cosf(theta);
            y_coords[i] = r * sinf(theta);
        }
    }

    if (permute)
    {
        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--)
        {
            int j = rand() % (i + 1);
            float temp_x = x_coords[i];
            float temp_y = y_coords[i];
            float temp_z = z_coords ? z_coords[i] : 0.0f;
            x_coords[i] = x_coords[j];
            y_coords[i] = y_coords[j];
            if (z_coords)
                z_coords[i] = z_coords[j];
            x_coords[j] = temp_x;
            y_coords[j] = temp_y;
            if (z_coords)
                z_coords[j] = temp_z;
        }
    }

    float min_x = x_coords[0], max_x = x_coords[0];
    float min_y = y_coords[0], max_y = y_coords[0];
    float min_z = z_coords ? z_coords[0] : 0.0f, max_z = min_z;
    for (int i = 1; i < n; i++)
    {
        if (x_coords[i] < min_x)
            min_x = x_coords[i];
        if (x_coords[i] > max_x)
            max_x = x_coords[i];
        if (y_coords[i] < min_y)
            min_y = y_coords[i];
        if (y_coords[i] > max_y)
            max_y = y_coords[i];
        if (z_coords && z_coords[i] < min_z)
            min_z = z_coords[i];
        if (z_coords && z_coords[i] > max_z)
            max_z = z_coords[i];
    }

    /* Shrink only if axis-aligned bbox would exceed SUNFLOWER_FIT_FRACTION of arena; never
     * grow. */
    float point_width = max_x - min_x;
    float point_height = max_y - min_y;
    float point_depth = max_z - min_z;
    float scale = 1.0f;
    if (point_width > 0.0f && point_height > 0.0f && avail_width > 0.0f && avail_height > 0.0f &&
        (!z_coords || (point_depth > 0.0f && avail_depth > 0.0f)))
    {
        float max_w = avail_width * SUNFLOWER_FIT_FRACTION;
        float max_h = avail_height * SUNFLOWER_FIT_FRACTION;
        if (point_width > max_w || point_height > max_h ||
            (z_coords && point_depth > avail_depth * SUNFLOWER_FIT_FRACTION))
        {
            scale = fminf(max_w / point_width, max_h / point_height);
            if (z_coords)
                scale = fminf(scale, avail_depth * SUNFLOWER_FIT_FRACTION / point_depth);
        }
    }

    float point_center_x = (min_x + max_x) / 2.0f;
    float point_center_y = (min_y + max_y) / 2.0f;
    float point_center_z = (min_z + max_z) / 2.0f;
    for (int i = 0; i < n; i++)
    {
        x_coords[i] = center_x + (x_coords[i] - point_center_x) * scale;
        y_coords[i] = center_y + (y_coords[i] - point_center_y) * scale;
        if (z_coords)
            z_coords[i] = center_z + (z_coords[i] - point_center_z) * scale;
    }
}

static inline float rand_open_unit(void)
{
    return ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static inline float rand_normal(float sigma)
{
    return sigma * sqrtf(-2.0f * logf(rand_open_unit())) * cosf(2.0f * PI * rand_open_unit());
}

static inline void init_gaussian_reset_points(int n, float *x_coords, float *y_coords,
                                              float *z_coords, float sigma, float center_x,
                                              float center_y, float center_z)
{
    float sigma2 = sigma * sigma;
    for (int i = 0; i < n; i++)
    {
        do
        {
            x_coords[i] = rand_normal(sigma);
            y_coords[i] = rand_normal(sigma);
            if (z_coords)
                z_coords[i] = rand_normal(sigma);
        } while (x_coords[i] * x_coords[i] + y_coords[i] * y_coords[i] +
                     (z_coords ? z_coords[i] * z_coords[i] : 0.0f) >
                 sigma2);
        x_coords[i] += center_x;
        y_coords[i] += center_y;
        if (z_coords)
            z_coords[i] += center_z;
    }
}

static void sample_proper_rotation(int spatial_dim, float R[9])
{
    memset(R, 0, 9 * sizeof(float));
    if (spatial_dim == 2)
    {
        float theta = rand_open_unit() * 2.0f * PI;
        float c = cosf(theta), s = sinf(theta);
        R[0] = c;
        R[1] = -s;
        R[2] = s;
        R[3] = c;
        return;
    }
    float x = rand_normal(1.0f), y = rand_normal(1.0f), z = rand_normal(1.0f),
          w = rand_normal(1.0f);
    float n = sqrtf(x * x + y * y + z * z + w * w);
    x /= n;
    y /= n;
    z /= n;
    w /= n;
    R[0] = 1.0f - 2.0f * (y * y + z * z);
    R[1] = 2.0f * (x * y - z * w);
    R[2] = 2.0f * (x * z + y * w);
    R[3] = 2.0f * (x * y + z * w);
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z - x * w);
    R[6] = 2.0f * (x * z - y * w);
    R[7] = 2.0f * (y * z + x * w);
    R[8] = 1.0f - 2.0f * (x * x + y * y);
}

static void apply_rotation_about_center(float *x, float *y, float *z, int spatial_dim,
                                        const float R[9], float cx, float cy, float cz)
{
    float dx = *x - cx, dy = *y - cy;
    if (spatial_dim == 2)
    {
        *x = cx + R[0] * dx + R[1] * dy;
        *y = cy + R[2] * dx + R[3] * dy;
        return;
    }
    float dz = *z - cz;
    *x = cx + R[0] * dx + R[1] * dy + R[2] * dz;
    *y = cy + R[3] * dx + R[4] * dy + R[5] * dz;
    *z = cz + R[6] * dx + R[7] * dy + R[8] * dz;
}

static void assign_rotated_targets(MultiCellEnv *env, const float R[9], float cx, float cy,
                                   float cz)
{
    for (int i = 0; i < env->num_agents; i++)
    {
        int offset = env->spatial_dim * i;
        float x = env->target_positions[offset];
        float y = env->target_positions[offset + 1];
        float z = env->spatial_dim == 3 ? env->target_positions[offset + 2] : 0.0f;
        apply_rotation_about_center(&x, &y, &z, env->spatial_dim, R, cx, cy, cz);
        env->agents[i].x = x;
        env->agents[i].y = y;
        env->agents[i].z = env->spatial_dim == 3 ? z : 0.0f;
    }
}

static void fit_packed_coords_to_target_bbox(MultiCellEnv *env, float *points_x, float *points_y,
                                             float *points_z)
{
    int n = env->num_agents, dim = env->spatial_dim;
    float tmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, tmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float pmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, pmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float *packed[3] = {points_x, points_y, points_z};
    for (int i = 0; i < n; i++)
    {
        float *t = &env->target_positions[i * dim];
        for (int d = 0; d < dim; d++)
        {
            tmin[d] = fminf(tmin[d], t[d]);
            tmax[d] = fmaxf(tmax[d], t[d]);
            pmin[d] = fminf(pmin[d], packed[d][i]);
            pmax[d] = fmaxf(pmax[d], packed[d][i]);
        }
    }
    float scale = FLT_MAX;
    for (int d = 0; d < dim; d++)
    {
        float tspan = tmax[d] - tmin[d], pspan = pmax[d] - pmin[d];
        if (tspan > 1e-6f && pspan > 1e-6f)
            scale = fminf(scale, tspan / pspan);
    }
    if (!(scale < FLT_MAX))
        scale = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float tc = 0.5f * (tmin[d] + tmax[d]);
        if (tmax[d] - tmin[d] <= 1e-6f)
        {
            for (int i = 0; i < n; i++)
                packed[d][i] = tc;
            continue;
        }
        float pc = 0.5f * (pmin[d] + pmax[d]);
        for (int i = 0; i < n; i++)
            packed[d][i] = tc + (packed[d][i] - pc) * scale;
    }
}

static float *alloc_packed_reset_coords(MultiCellEnv *env, float center_x, float center_y,
                                        float center_z, float avail_width, float avail_height,
                                        float avail_depth, float **points_y, float **points_z)
{
    float *coords =
        (float *)malloc((size_t)env->num_agents * (size_t)env->spatial_dim * sizeof(float));
    float *points_x = coords;
    *points_y = coords + env->num_agents;
    *points_z = env->spatial_dim == 3 ? *points_y + env->num_agents : NULL;
    float median_dist = 3.0f * SUNFLOWER_CELL_RADIUS_MICRONS * env->coordinate_scale;
    if (env->reset_distribution == RESET_DISTRIBUTION_GAUSSIAN)
    {
        float sigma = (env->spatial_dim == 3)
                          ? median_dist * cbrtf(3.0f * (float)env->num_agents / (4.0f * PI))
                          : (median_dist / 1.787f) * sqrtf((float)env->num_agents);
        float fit_radius = 0.5f * fminf(avail_width, avail_height);
        if (*points_z)
            fit_radius = fminf(fit_radius, 0.5f * avail_depth);
        sigma = fminf(sigma, fit_radius * SUNFLOWER_FIT_FRACTION);
        init_gaussian_reset_points(env->num_agents, points_x, *points_y, *points_z, sigma, center_x,
                                   center_y, center_z);
    }
    else
    {
        init_reset_points(env->num_agents, points_x, *points_y, *points_z, median_dist, 1, center_x,
                          center_y, center_z, avail_width, avail_height, avail_depth);
    }
    if (env->fit_init_to_target_bbox && env->target_positions != NULL)
        fit_packed_coords_to_target_bbox(env, points_x, *points_y, *points_z);
    return points_x;
}

void add_log(MultiCellEnv *env)
{
    env->log.error += env->buffer.error;
    env->log.sa_h0 += env->buffer.sa_h0;
    env->log.sa_h1 += env->buffer.sa_h1;
    env->log.sa_h2 += env->buffer.sa_h2;
    env->log.sa_h3 += env->buffer.sa_h3;
    env->log.sa_h4 += env->buffer.sa_h4;
    env->log.sa_h5 += env->buffer.sa_h5;
    env->log.episode_return += env->buffer.episode_return;
    env->log.episode_length += env->buffer.episode_length;
    env->log.n += 1.0f;
    env->buffer = (Log){0};
}

static inline float compute_distance_error(MultiCellEnv *env)
{
    int k = env->k;
    if (env->num_agents <= 0 || k <= 0)
    {
        return 0.0f;
    }
    if (env->reward_mode == REWARD_MODE_INTERVAL_SPLIT_MATCHED)
    {
        return matched_interval_split_distance_error(env);
    }

    float sum_agent_mae = 0.0f;
    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent = &env->agents[i];
        int target_offset = i * k;
        float agent_abs_err_sum = 0.0f;
        int valid_targets = 0;

        for (int t = 0; t < k; t++)
        {
            int target_neighbor = env->target_neighbors[target_offset + t];
            if (target_neighbor < 0)
            {
                continue;
            }
            valid_targets++;

            Agent *neighbor = &env->agents[target_neighbor];
            float actual_dist = sqrtf(agent_distance_squared(env, agent, neighbor));
            float target_dist = env->target_neighbor_distances[target_offset + t];
            agent_abs_err_sum += fabsf(actual_dist - target_dist);
        }

        if (valid_targets > 0)
        {
            sum_agent_mae += agent_abs_err_sum / (float)valid_targets;
        }
    }

    return sum_agent_mae / (float)env->num_agents;
}

static inline void insert_topk_neighbor(MultiCellEnv *env, int a, int b, float d2)
{
    int k = env->k;
    int *idx = &env->obs_topk_idx_cache[a * k];
    float *dists = &env->obs_topk_d2_cache[a * k];
    for (int i = 0; i < k; i++)
    {
        if (idx[i] == b)
        {
            return;
        }
    }
    if (d2 >= dists[k - 1])
    {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && d2 < dists[pos - 1])
    {
        dists[pos] = dists[pos - 1];
        idx[pos] = idx[pos - 1];
        pos--;
    }
    dists[pos] = d2;
    idx[pos] = b;
}

static inline void add_radius_topk_edges(MultiCellEnv *env, float r2)
{
    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent_i = &env->agents[i];
        for (int j = i + 1; j < env->num_agents; j++)
        {
            Agent *agent_j = &env->agents[j];
            float d2 = agent_distance_squared(env, agent_i, agent_j);
            if (d2 <= r2)
            {
                insert_topk_neighbor(env, i, j, d2);
                insert_topk_neighbor(env, j, i, d2);
            }
        }
    }
}

static inline void add_delaunay_topk_edges(MultiCellEnv *env, float r2)
{
    int n = env->num_agents;
    if (n < 3)
    {
        add_radius_topk_edges(env, r2);
        return;
    }

    coordT *points = (coordT *)malloc((size_t)n * 2 * sizeof(coordT));
    if (points == NULL)
    {
        return;
    }
    for (int i = 0; i < n; i++)
    {
        points[i * 2] = (coordT)env->agents[i].x;
        points[i * 2 + 1] = (coordT)env->agents[i].y;
    }

    FILE *errfile = fopen("/dev/null", "w");
    if (errfile == NULL)
    {
        errfile = stderr;
    }

    qhT qh_qh;
    qhT *qh = &qh_qh;
    qh_zero(qh, errfile);
    char options[] = "qhull d Qbb Qc Qz Qt";
    int exitcode = qh_new_qhull(qh, 2, n, points, False, options, NULL, errfile);
    if (!exitcode)
    {
        facetT *facet;
        vertexT *vertex, **vertexp;

        for (facet = qh->facet_list; facet; facet = facet->next)
        {
            if (facet->upperdelaunay)
            {
                continue;
            }
            int ids[8];
            int nids = 0;
            FOREACHvertex_(facet->vertices)
            {
                int id = qh_pointid(qh, vertex->point);
                if (id >= 0 && id < n && nids < 8)
                {
                    ids[nids++] = id;
                }
            }
            if (nids != 3)
            {
                continue;
            }
            for (int a = 0; a < 3; a++)
            {
                for (int b = a + 1; b < 3; b++)
                {
                    int i = ids[a];
                    int j = ids[b];
                    Agent *agent_i = &env->agents[i];
                    Agent *agent_j = &env->agents[j];
                    float dx = agent_j->x - agent_i->x;
                    float dy = agent_j->y - agent_i->y;
                    float d2 = dx * dx + dy * dy;
                    if (d2 <= r2)
                    {
                        insert_topk_neighbor(env, i, j, d2);
                        insert_topk_neighbor(env, j, i, d2);
                    }
                }
            }
        }
    }

    int curlong = 0;
    int totlong = 0;
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (errfile != stderr)
    {
        fclose(errfile);
    }
    free(points);
    if (exitcode)
    {
        add_radius_topk_edges(env, r2);
    }
}

static inline void append_radius_neighbor(MultiCellEnv *env, int a, int b)
{
    if (env->radius_neighbor_count >= env->radius_neighbor_capacity)
    {
        int capacity = env->radius_neighbor_capacity > 0 ? env->radius_neighbor_capacity * 2 : 16;
        while (capacity <= env->radius_neighbor_count)
        {
            capacity *= 2;
        }
        env->radius_neighbor_to =
            (int *)realloc(env->radius_neighbor_to, (size_t)capacity * sizeof(int));
        env->radius_neighbor_next =
            (int *)realloc(env->radius_neighbor_next, (size_t)capacity * sizeof(int));
        env->radius_neighbor_capacity = capacity;
    }
    int edge = env->radius_neighbor_count++;
    env->radius_neighbor_to[edge] = b;
    env->radius_neighbor_next[edge] = env->radius_neighbor_head[a];
    env->radius_neighbor_head[a] = edge;
}

static inline void spiky_grad_accumulate(MultiCellEnv *env, float dx, float dy, float dz, float d2,
                                         float h, float *gx, float *gy, float *gz)
{
    float d = sqrtf(d2);
    if (d <= 0.0f || d >= h)
    {
        return;
    }
    float t = h - d;
    float mag = 3.0f * t * t / d;
    *gx += mag * dx;
    *gy += mag * dy;
    if (env->spatial_dim == 3)
        *gz += mag * dz;
}

static inline void write_density_grad_obs(MultiCellEnv *env, int agent_idx, float *out_x,
                                          float *out_y, float *out_z)
{
    float gx = env->obs_density_grad_x[agent_idx];
    float gy = env->obs_density_grad_y[agent_idx];
    float gz = env->obs_density_grad_z[agent_idx];
    if (env->density_grad_log_norm)
    {
        float norm_sq = gx * gx + gy * gy + (env->spatial_dim == 3 ? gz * gz : 0.0f);
        if (norm_sq <= 1e-24f)
        {
            gx = 0.0f;
            gy = 0.0f;
            gz = 0.0f;
        }
        else
        {
            float norm = sqrtf(norm_sq);
            float scale = logf(1.0f + norm) / norm;
            gx *= scale;
            gy *= scale;
            gz *= scale;
        }
    }
    *out_x = gx;
    *out_y = gy;
    *out_z = gz;
}

static inline void build_neighbor_cache(MultiCellEnv *env)
{
    int k = env->k;
    size_t state_size = (size_t)env->num_agents * (size_t)k;
    for (size_t i = 0; i < state_size; i++)
    {
        env->obs_topk_idx_cache[i] = -1;
        env->obs_topk_d2_cache[i] = FLT_MAX;
    }
    memset(env->obs_perceived_count_cache, 0, (size_t)env->num_agents * sizeof(int));
    if (env->density_grad_obs)
    {
        memset(env->obs_density_grad_x, 0, (size_t)env->num_agents * sizeof(float));
        memset(env->obs_density_grad_y, 0, (size_t)env->num_agents * sizeof(float));
        memset(env->obs_density_grad_z, 0, (size_t)env->num_agents * sizeof(float));
    }
    for (int i = 0; i < env->num_agents; i++)
    {
        env->radius_neighbor_head[i] = -1;
    }
    env->radius_neighbor_count = 0;

    float r2 = env->perception_radius * env->perception_radius;
    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent_i = &env->agents[i];
        for (int j = i + 1; j < env->num_agents; j++)
        {
            Agent *agent_j = &env->agents[j];
            float dx = agent_j->x - agent_i->x;
            float dy = agent_j->y - agent_i->y;
            float dz = agent_j->z - agent_i->z;
            float d2 = agent_distance_squared(env, agent_i, agent_j);
            if (d2 > r2)
            {
                continue;
            }

            env->obs_perceived_count_cache[i]++;
            env->obs_perceived_count_cache[j]++;
            if (env->density_grad_obs)
            {
                spiky_grad_accumulate(env, dx, dy, dz, d2, env->perception_radius,
                                      &env->obs_density_grad_x[i], &env->obs_density_grad_y[i],
                                      &env->obs_density_grad_z[i]);
                spiky_grad_accumulate(env, -dx, -dy, -dz, d2, env->perception_radius,
                                      &env->obs_density_grad_x[j], &env->obs_density_grad_y[j],
                                      &env->obs_density_grad_z[j]);
            }
            if (env->neighbor_graph == NEIGHBOR_GRAPH_RADIUS_TOPK)
            {
                insert_topk_neighbor(env, i, j, d2);
                insert_topk_neighbor(env, j, i, d2);
            }
            if (env->enforce_connected)
            {
                append_radius_neighbor(env, i, j);
                append_radius_neighbor(env, j, i);
            }
        }
    }
    if (env->neighbor_graph == NEIGHBOR_GRAPH_DELAUNAY_TOPK)
    {
        add_delaunay_topk_edges(env, r2);
    }
    if (env->density_grad_obs)
    {
        float h = env->perception_radius;
        float scale = (env->spatial_dim == 3 ? 15.0f / (M_PI * powf(h, 6.0f))
                                             : 10.0f / (M_PI * powf(h, 5.0f))) /
                      (float)env->num_agents;
        for (int a = 0; a < env->num_agents; a++)
        {
            env->obs_density_grad_x[a] *= scale;
            env->obs_density_grad_y[a] *= scale;
            env->obs_density_grad_z[a] *= scale;
        }
    }
}

static inline int agents_connected(MultiCellEnv *env)
{
    int n = env->num_agents;
    if (n <= 1)
    {
        return 1;
    }
    memset(env->connected_seen, 0, (size_t)n * sizeof(unsigned char));
    int head = 0;
    int tail = 0;
    env->connected_queue[tail++] = 0;
    env->connected_seen[0] = 1;
    while (head < tail)
    {
        int i = env->connected_queue[head++];
        for (int edge = env->radius_neighbor_head[i]; edge != -1;
             edge = env->radius_neighbor_next[edge])
        {
            int j = env->radius_neighbor_to[edge];
            if (env->connected_seen[j])
            {
                continue;
            }
            env->connected_seen[j] = 1;
            env->connected_queue[tail++] = j;
        }
    }
    return tail == n;
}

static inline float normalize_heading(float heading)
{
    heading = fmodf(heading, 2.0f * PI);
    return heading < 0.0f ? heading + 2.0f * PI : heading;
}

static inline void sync_initial_reuse(MultiCellEnv *env, int *set, int external)
{
    if (env->reuse_initial_positions && !external)
        *set = 1;
    else if (*set && !env->reuse_initial_positions)
        *set = 0;
}

static inline void vec_to_ego_polar(float wx, float wy, float heading, float *out_r, float *out_sin,
                                    float *out_cos)
{
    float cos_h = cosf(heading);
    float sin_h = sinf(heading);
    float fwd = cos_h * wx + sin_h * wy;
    float lat = -sin_h * wx + cos_h * wy;
    float r = sqrtf(fwd * fwd + lat * lat);
    if (r > 0.0f)
    {
        *out_r = r;
        *out_sin = lat / r;
        *out_cos = fwd / r;
    }
    else
    {
        *out_r = 0.0f;
        *out_sin = 0.0f;
        *out_cos = 0.0f;
    }
}

static inline void vec_to_ego_3d(float x, float y, float z, Agent *agent, float *distance,
                                 float *forward, float *right, float *up)
{
    *distance = sqrtf(x * x + y * y + z * z);
    float scale = *distance > 0.0f ? 1.0f / *distance : 0.0f;
    float sin_yaw = sinf(agent->heading), cos_yaw = cosf(agent->heading);
    float sin_pitch = sinf(agent->pitch), cos_pitch = cosf(agent->pitch);
    *forward = (cos_pitch * cos_yaw * x + cos_pitch * sin_yaw * y + sin_pitch * z) * scale;
    *right = (-sin_yaw * x + cos_yaw * y) * scale;
    *up = (-sin_pitch * cos_yaw * x - sin_pitch * sin_yaw * y + cos_pitch * z) * scale;
}

void compute_observations(MultiCellEnv *env)
{
    int obs_idx = 0;
    int k = env->k;

    for (int a = 0; a < env->num_agents; a++)
    {
        Agent *agent = &env->agents[a];
        int *best_idx = &env->obs_topk_idx_cache[a * k];
        int perceived = env->obs_perceived_count_cache[a];

        for (int n = 0; n < k; n++)
        {
            int idx = best_idx[n];
            if (idx < 0)
            {
                int payload = 3 * env->spatial_dim - 1 + env->local_frame_obs;
                for (int c = 0; c < payload; c++)
                    env->observations[obs_idx++] = 0.0f;
                if (env->mutual_obs_bit)
                    env->observations[obs_idx++] = 0.0f;
                if (env->target_neighbor_bit)
                    env->observations[obs_idx++] = 0.0f;
                if (env->comm_select_action)
                    env->observations[obs_idx++] = 0.0f;
                env->observations[obs_idx++] = 0.0f;
                continue;
            }
            Agent *neighbor = &env->agents[idx];
            float dx = (neighbor->x - agent->x) / env->perception_radius;
            float dy = (neighbor->y - agent->y) / env->perception_radius;
            float dz = (neighbor->z - agent->z) / env->perception_radius;
            float neighbor_speed = neighbor->speed / env->max_speed;
            float mutual_observation = 0.0f;
            if (env->mutual_obs_bit)
            {
                int *reciprocal_best_idx = &env->obs_topk_idx_cache[idx * k];
                for (int r = 0; r < k; r++)
                {
                    if (reciprocal_best_idx[r] == a)
                    {
                        mutual_observation = 1.0f;
                        break;
                    }
                }
            }
            float target_neighbor = 0.0f;
            if (env->target_neighbor_bit)
            {
                int *targets = &env->target_neighbors[a * k];
                for (int t = 0; t < k; t++)
                {
                    if (targets[t] == idx)
                    {
                        target_neighbor = 1.0f;
                        break;
                    }
                }
            }
            if (env->spatial_dim == 3 && env->local_frame_obs)
            {
                float dist, forward, right, up;
                vec_to_ego_3d(dx, dy, dz, agent, &dist, &forward, &right, &up);
                env->observations[obs_idx++] = dist;
                env->observations[obs_idx++] = forward;
                env->observations[obs_idx++] = right;
                env->observations[obs_idx++] = up;
                env->observations[obs_idx++] = sinf(neighbor->heading - agent->heading);
                env->observations[obs_idx++] = cosf(neighbor->heading - agent->heading);
                env->observations[obs_idx++] = sinf(neighbor->pitch - agent->pitch);
                env->observations[obs_idx++] = cosf(neighbor->pitch - agent->pitch);
            }
            else if (env->local_frame_obs)
            {
                float dist, sin_bear, cos_bear;
                vec_to_ego_polar(dx, dy, agent->heading, &dist, &sin_bear, &cos_bear);
                float rel_heading = neighbor->heading - agent->heading;
                env->observations[obs_idx++] = dist;
                env->observations[obs_idx++] = sin_bear;
                env->observations[obs_idx++] = cos_bear;
                env->observations[obs_idx++] = sinf(rel_heading);
                env->observations[obs_idx++] = cosf(rel_heading);
            }
            else
            {
                env->observations[obs_idx++] = dx;
                env->observations[obs_idx++] = dy;
                if (env->spatial_dim == 3)
                    env->observations[obs_idx++] = dz;
                env->observations[obs_idx++] = sinf(neighbor->heading);
                env->observations[obs_idx++] = cosf(neighbor->heading);
                if (env->spatial_dim == 3)
                {
                    env->observations[obs_idx++] = sinf(neighbor->pitch);
                    env->observations[obs_idx++] = cosf(neighbor->pitch);
                }
            }
            env->observations[obs_idx++] = neighbor_speed;
            if (env->mutual_obs_bit)
            {
                env->observations[obs_idx++] = mutual_observation;
            }
            if (env->target_neighbor_bit)
                env->observations[obs_idx++] = target_neighbor;
            if (env->comm_select_action)
            {
                env->observations[obs_idx++] = comm_selected_target(env, idx, a);
            }
            env->observations[obs_idx++] = (float)(idx + 1);
        }

        env->observations[obs_idx++] = (float)(a + 1);

        float denom = (float)(env->num_agents - 1);
        float perceived_ratio = denom > 0.0f ? ((float)perceived / denom) : 0.0f;
        if (env->perceived_ratio_obs)
        {
            env->observations[obs_idx++] = perceived_ratio;
        }
        if (!env->local_frame_obs)
        {
            env->observations[obs_idx++] = sinf(agent->heading);
            env->observations[obs_idx++] = cosf(agent->heading);
            if (env->spatial_dim == 3)
            {
                env->observations[obs_idx++] = sinf(agent->pitch);
                env->observations[obs_idx++] = cosf(agent->pitch);
            }
        }
        env->observations[obs_idx++] = agent->speed / env->max_speed;
        if (env->self_pos_delta_obs)
        {
            float step_x = (agent->x - env->prev_obs_x[a]) / env->perception_radius;
            float step_y = (agent->y - env->prev_obs_y[a]) / env->perception_radius;
            float step_z = (agent->z - env->prev_obs_z[a]) / env->perception_radius;
            if (env->spatial_dim == 3 && env->local_frame_obs)
            {
                float step_r, forward, right, up;
                vec_to_ego_3d(step_x, step_y, step_z, agent, &step_r, &forward, &right, &up);
                env->observations[obs_idx++] = step_r;
                env->observations[obs_idx++] = forward;
                env->observations[obs_idx++] = right;
                env->observations[obs_idx++] = up;
            }
            else if (env->local_frame_obs)
            {
                float step_r, sin_step, cos_step;
                vec_to_ego_polar(step_x, step_y, agent->heading, &step_r, &sin_step, &cos_step);
                env->observations[obs_idx++] = step_r;
                env->observations[obs_idx++] = sin_step;
                env->observations[obs_idx++] = cos_step;
            }
            else
            {
                env->observations[obs_idx++] = step_x;
                env->observations[obs_idx++] = step_y;
                if (env->spatial_dim == 3)
                    env->observations[obs_idx++] = step_z;
            }
        }
        if (env->boundary_dist_obs)
        {
            float dist = fmaxf(0.0f, agent_boundary_distance(env, agent->x, agent->y, agent->z));
            env->observations[obs_idx++] = fminf(1.0f, dist / env->perception_radius);
        }
        if (env->density_grad_obs)
        {
            float grad_x = 0.0f;
            float grad_y = 0.0f;
            float grad_z = 0.0f;
            write_density_grad_obs(env, a, &grad_x, &grad_y, &grad_z);
            if (env->spatial_dim == 3 && env->local_frame_obs)
            {
                float grad_r, forward, right, up;
                vec_to_ego_3d(grad_x, grad_y, grad_z, agent, &grad_r, &forward, &right, &up);
                env->observations[obs_idx++] = grad_r;
                env->observations[obs_idx++] = forward;
                env->observations[obs_idx++] = right;
                env->observations[obs_idx++] = up;
            }
            else if (env->local_frame_obs)
            {
                float grad_r, sin_grad, cos_grad;
                vec_to_ego_polar(grad_x, grad_y, agent->heading, &grad_r, &sin_grad, &cos_grad);
                env->observations[obs_idx++] = grad_r;
                env->observations[obs_idx++] = sin_grad;
                env->observations[obs_idx++] = cos_grad;
            }
            else
            {
                env->observations[obs_idx++] = grad_x;
                env->observations[obs_idx++] = grad_y;
                if (env->spatial_dim == 3)
                    env->observations[obs_idx++] = grad_z;
            }
        }
        env->prev_obs_x[a] = agent->x;
        env->prev_obs_y[a] = agent->y;
        env->prev_obs_z[a] = agent->z;
    }
}

int c_reset(MultiCellEnv *env)
{
    float margin = BORDER_CLIP_OFFSET;
    float max_x = (float)env->width - margin;
    float max_y = (float)env->height - margin;
    float max_z = (float)env->depth - margin;
    if (max_x < margin)
        max_x = margin;
    if (max_y < margin)
        max_y = margin;
    if (max_z < margin)
        max_z = margin;
    float center_x = (max_x + margin) * 0.5f;
    float center_y = (max_y + margin) * 0.5f;
    float center_z = (max_z + margin) * 0.5f;
    float avail_width = max_x - margin;
    float avail_height = max_y - margin;
    float avail_depth = max_z - margin;

    if (!env->initial_positions_set)
    {
        if (env->reset_mode == RESET_MODE_ROTATED_TARGET ||
            env->reset_mode == RESET_MODE_INTERPOLATED)
        {
            float R[9];
            sample_proper_rotation(env->spatial_dim, R);
            assign_rotated_targets(env, R, center_x, center_y, center_z);
            if (env->reset_mode == RESET_MODE_INTERPOLATED)
            {
                float *points_y, *points_z;
                float *points_x =
                    alloc_packed_reset_coords(env, center_x, center_y, center_z, avail_width,
                                              avail_height, avail_depth, &points_y, &points_z);
                float alpha = rand_open_unit();
                float one_minus = 1.0f - alpha;
                for (int i = 0; i < env->num_agents; i++)
                {
                    env->agents[i].x = one_minus * env->agents[i].x + alpha * points_x[i];
                    env->agents[i].y = one_minus * env->agents[i].y + alpha * points_y[i];
                    if (env->spatial_dim == 3)
                        env->agents[i].z = one_minus * env->agents[i].z + alpha * points_z[i];
                }
                free(points_x);
            }
        }
        else
        {
            float *points_y, *points_z;
            float *points_x =
                alloc_packed_reset_coords(env, center_x, center_y, center_z, avail_width,
                                          avail_height, avail_depth, &points_y, &points_z);
            for (int i = 0; i < env->num_agents; i++)
            {
                env->agents[i].x = points_x[i];
                env->agents[i].y = points_y[i];
                env->agents[i].z = env->spatial_dim == 3 ? points_z[i] : 0.0f;
            }
            free(points_x);
        }
    }

    for (int i = 0; i < env->num_agents; i++)
    {
        if (env->initial_positions_set)
        {
            int offset = env->spatial_dim * i;
            env->agents[i].x = env->initial_positions[offset];
            env->agents[i].y = env->initial_positions[offset + 1];
            env->agents[i].z = env->spatial_dim == 3 ? env->initial_positions[offset + 2] : 0.0f;
        }
        env->agents[i].heading =
            normalize_heading(env->initial_headings ? env->initial_headings[i] : 0.0f);
        env->agents[i].pitch = 0.0f;
        env->agents[i].speed = 0.0f;
    }
    clip_agents(env);
    if (env->reuse_initial_positions && env->initial_positions != NULL &&
        !env->initial_positions_external && !env->initial_positions_set)
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            int offset = env->spatial_dim * i;
            env->initial_positions[offset] = env->agents[i].x;
            env->initial_positions[offset + 1] = env->agents[i].y;
            if (env->spatial_dim == 3)
                env->initial_positions[offset + 2] = env->agents[i].z;
        }
    }
    for (int i = 0; i < env->num_agents; i++)
    {
        env->prev_obs_x[i] = env->agents[i].x;
        env->prev_obs_y[i] = env->agents[i].y;
        env->prev_obs_z[i] = env->agents[i].z;
    }
    build_neighbor_cache(env);
    if (env->enforce_connected && !agents_connected(env))
    {
        return 1;
    }
    if (env->reward_mode == REWARD_MODE_INTERVAL_SPLIT_MATCHED)
    {
        match_interval_split_roles(env, env->matched_role_to_agent);
        snapshot_matched_agent_history(env);
    }
    env->error_init = compute_distance_error(env);
    sync_initial_reuse(env, &env->initial_positions_set, env->initial_positions_external);
    memset(env->reward_state_ready, 0, (size_t)env->num_agents * sizeof(unsigned char));
    size_t distance_state_size = (size_t)env->num_agents * (size_t)env->k;
    for (size_t i = 0; i < distance_state_size; i++)
    {
        env->prev_target_err[i] = -1.0f;
    }
    memset(env->prev_distance_bin, 0, (size_t)env->num_agents * sizeof(int));
    memset(env->prev_distance_bin_split, -1, distance_state_size * sizeof(int));
    clear_comm_selected_targets(env);
    compute_observations(env);
    return 0;
}

float clip(float val, float min, float max)
{
    if (val < min)
    {
        return min;
    }
    else if (val > max)
    {
        return max;
    }
    return val;
}

static inline void clear_collision_grid(MultiCellEnv *env)
{
    int grid_size_total = env->grid_width * env->grid_h;
    for (int i = 0; i < grid_size_total; i++)
    {
        env->cell_head[i] = -1;
    }
}

static inline void build_collision_grid(MultiCellEnv *env, float grid_size)
{
    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent = &env->agents[i];
        int x_cell = (int)floorf(agent->x / grid_size);
        int y_cell = (int)floorf(agent->y / grid_size);

        // Clamp to valid grid bounds
        if (x_cell < 0)
            x_cell = 0;
        if (x_cell >= env->grid_width)
            x_cell = env->grid_width - 1;
        if (y_cell < 0)
            y_cell = 0;
        if (y_cell >= env->grid_h)
            y_cell = env->grid_h - 1;

        int c = env->grid_width * y_cell + x_cell;
        env->agent_next[i] = env->cell_head[c];
        env->cell_head[c] = i;
    }
}

static inline void resolve_collisions_in_grid(MultiCellEnv *env, float min_dist_sq, int k)
{
    for (int cy = 0; cy < env->grid_h; cy++)
    {
        for (int cx = 0; cx < env->grid_width; cx++)
        {
            int cell_idx = env->grid_width * cy + cx;
            if (env->cell_head[cell_idx] == -1)
            {
                continue;
            }

            // Check pairs within same cell
            for (int i = env->cell_head[cell_idx]; i != -1; i = env->agent_next[i])
            {
                for (int j = env->agent_next[i]; j != -1; j = env->agent_next[j])
                {
                    Agent *agent_i = &env->agents[i];
                    Agent *agent_j = &env->agents[j];

                    float dx = agent_j->x - agent_i->x;
                    float dy = agent_j->y - agent_i->y;
                    float d2 = dx * dx + dy * dy;

                    if (d2 >= min_dist_sq)
                        continue;

                    // Handle d2 ≈ 0 case
                    float d, inv_d;
                    if (d2 < 1e-6f)
                    {
                        dx = 1.0f;
                        dy = 0.0f;
                        d = 1.0f;
                        inv_d = 1.0f;
                    }
                    else
                    {
                        d = sqrtf(d2);
                        inv_d = 1.0f / d;
                        dx *= inv_d;
                        dy *= inv_d;
                    }

                    float penetration = env->min_agent_distance - d;
                    float correction = penetration * 0.5f;

                    agent_i->x -= dx * correction;
                    agent_i->y -= dy * correction;
                    agent_j->x += dx * correction;
                    agent_j->y += dy * correction;
                }
            }

            // Check pairs with neighbor cells
            for (int dy_off = -k; dy_off <= k; dy_off++)
            {
                for (int dx_off = -k; dx_off <= k; dx_off++)
                {
                    if (dx_off == 0 && dy_off == 0)
                        continue; // Already checked same cell

                    int nx = cx + dx_off;
                    int ny = cy + dy_off;

                    // Only check forward neighbors to avoid duplicate pairs
                    if (nx < cx || (nx == cx && ny < cy))
                        continue;

                    if (nx < 0 || nx >= env->grid_width || ny < 0 || ny >= env->grid_h)
                        continue;

                    int neighbor_idx = env->grid_width * ny + nx;
                    if (env->cell_head[neighbor_idx] == -1)
                    {
                        continue;
                    }

                    // Check all pairs between current cell and neighbor cell
                    for (int i = env->cell_head[cell_idx]; i != -1; i = env->agent_next[i])
                    {
                        for (int j = env->cell_head[neighbor_idx]; j != -1; j = env->agent_next[j])
                        {
                            Agent *agent_i = &env->agents[i];
                            Agent *agent_j = &env->agents[j];

                            float dx = agent_j->x - agent_i->x;
                            float dy = agent_j->y - agent_i->y;
                            float d2 = dx * dx + dy * dy;

                            if (d2 >= min_dist_sq)
                                continue;

                            // Handle d2 ≈ 0 case
                            float d, inv_d;
                            if (d2 < 1e-6f)
                            {
                                dx = 1.0f;
                                dy = 0.0f;
                                d = 1.0f;
                                inv_d = 1.0f;
                            }
                            else
                            {
                                d = sqrtf(d2);
                                inv_d = 1.0f / d;
                                dx *= inv_d;
                                dy *= inv_d;
                            }

                            float penetration = env->min_agent_distance - d;
                            float correction = penetration * 0.5f;

                            agent_i->x -= dx * correction;
                            agent_i->y -= dy * correction;
                            agent_j->x += dx * correction;
                            agent_j->y += dy * correction;
                        }
                    }
                }
            }
        }
    }
}

static inline void clip_agents(MultiCellEnv *env)
{
    if (!env->bounded)
    {
        return;
    }

    float margin = BORDER_CLIP_OFFSET;
    float max_x = (float)env->width - margin;
    float max_y = (float)env->height - margin;
    if (max_x < margin)
        max_x = margin;
    if (max_y < margin)
        max_y = margin;

    if (!env->circular)
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            Agent *agent = &env->agents[i];
            agent->x = clip(agent->x, margin, max_x);
            agent->y = clip(agent->y, margin, max_y);
            if (env->spatial_dim == 3)
                agent->z = clip(agent->z, margin, (float)env->depth - margin);
        }
        return;
    }

    float center_x = (float)env->width * 0.5f;
    float center_y = (float)env->height * 0.5f;
    float radius_limit = env_radius_limit(env);
    float radius_limit_sq = radius_limit * radius_limit;

    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent = &env->agents[i];
        float dx = agent->x - center_x;
        float dy = agent->y - center_y;
        float dz = agent->z - (float)env->depth * 0.5f;
        float dist_sq = dx * dx + dy * dy + (env->spatial_dim == 3 ? dz * dz : 0.0f);
        if (dist_sq <= radius_limit_sq)
        {
            continue;
        }
        if (dist_sq <= 1e-12f)
        {
            agent->x = center_x + radius_limit;
            agent->y = center_y;
            continue;
        }

        float scale = radius_limit / sqrtf(dist_sq);
        agent->x = center_x + dx * scale;
        agent->y = center_y + dy * scale;
        if (env->spatial_dim == 3)
            agent->z = (float)env->depth * 0.5f + dz * scale;
    }
}

static inline void recenter_agents(MultiCellEnv *env)
{
    if (env->num_agents <= 0)
    {
        return;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    for (int i = 0; i < env->num_agents; i++)
    {
        sum_x += env->agents[i].x;
        sum_y += env->agents[i].y;
        sum_z += env->agents[i].z;
    }
    float centroid_x = sum_x / (float)env->num_agents;
    float centroid_y = sum_y / (float)env->num_agents;
    float centroid_z = sum_z / (float)env->num_agents;

    float center_x = (float)env->width / 2.0f;
    float center_y = (float)env->height / 2.0f;
    float center_z = (float)env->depth / 2.0f;

    float dx = center_x - centroid_x;
    float dy = center_y - centroid_y;
    float dz = center_z - centroid_z;
    for (int i = 0; i < env->num_agents; i++)
    {
        env->agents[i].x += dx;
        env->agents[i].y += dy;
        if (env->spatial_dim == 3)
            env->agents[i].z += dz;
        // Keep previous-observation positions in the same translated frame,
        // so recentering contributes zero to self-position deltas.
        env->prev_obs_x[i] += dx;
        env->prev_obs_y[i] += dy;
        if (env->spatial_dim == 3)
            env->prev_obs_z[i] += dz;
    }
}

void resolve_collisions(MultiCellEnv *env)
{
    // Collision response intentionally uses only the XY projection in both dimensions.
    float min_dist_sq = env->min_agent_distance * env->min_agent_distance;
    float grid_size = env->collision_grid_size;

    // Compute neighborhood radius k
    int k = 1;
    if (grid_size < env->min_agent_distance)
    {
        k = (int)ceilf(env->min_agent_distance / grid_size);
    }

    // Main solver iterations
    for (int iter = 0; iter < env->collision_iters; iter++)
    {
        clear_collision_grid(env);
        build_collision_grid(env, grid_size);
        resolve_collisions_in_grid(env, min_dist_sq, k);
        clip_agents(env);
    }
}

int c_step(MultiCellEnv *env)
{
    env->buffer.episode_length += 1.0f;
    cache_comm_selected_targets(env);
    float prev_x[env->num_agents];
    float prev_y[env->num_agents];
    float prev_z[env->num_agents];
    float prev_heading[env->num_agents];
    float prev_pitch[env->num_agents];
    float prev_speed[env->num_agents];
    int disconnected = 0;
    unsigned char boundary_hit[env->num_agents];

    for (int i = 0; i < env->num_agents; i++)
    {
        env->terminals[i] = 0;
        env->rewards[i] = 0.0f;
        Agent *agent = &env->agents[i];
        prev_x[i] = agent->x;
        prev_y[i] = agent->y;
        prev_z[i] = agent->z;
        prev_heading[i] = agent->heading;
        prev_pitch[i] = agent->pitch;
        prev_speed[i] = agent->speed;

        int *actions = &env->actions[env->action_dim * i];
        float turn_deg = (float)actions[0] * 5.0f - 20.0f;
        agent->heading += turn_deg * (PI / 180.0f);
        agent->heading = normalize_heading(agent->heading);

        if (env->spatial_dim == 3)
        {
            agent->pitch += ((float)actions[1] * 5.0f - 20.0f) * (PI / 180.0f);
            agent->pitch = clip(agent->pitch, -PI * 0.5f, PI * 0.5f);
        }
        int speed_action = actions[env->spatial_dim - 1];
        if (speed_action >= 0 && speed_action <= 5)
        {
            if (speed_action == 0)
                env->buffer.sa_h0 += 1.0f;
            else if (speed_action == 1)
                env->buffer.sa_h1 += 1.0f;
            else if (speed_action == 2)
                env->buffer.sa_h2 += 1.0f;
            else if (speed_action == 3)
                env->buffer.sa_h3 += 1.0f;
            else if (speed_action == 4)
                env->buffer.sa_h4 += 1.0f;
            else
                env->buffer.sa_h5 += 1.0f;
        }
        if (speed_action == 5)
        {
            agent->speed = 0.0f;
        }
        else
        {
            agent->speed += 1.0f * ((float)speed_action - 2.0f);
            agent->speed = clip(agent->speed, -env->max_speed, env->max_speed);
        }

        if (env->spatial_dim == 3)
        {
            float horizontal_speed = agent->speed * cosf(agent->pitch);
            agent->x += horizontal_speed * cosf(agent->heading);
            agent->y += horizontal_speed * sinf(agent->heading);
            agent->z += agent->speed * sinf(agent->pitch);
        }
        else
        {
            agent->x += agent->speed * cosf(agent->heading);
            agent->y += agent->speed * sinf(agent->heading);
        }
    }
    if (env->boundary_penalty != 0.0f)
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            Agent *agent = &env->agents[i];
            boundary_hit[i] = agent_boundary_distance(env, agent->x, agent->y, agent->z) < 0.0f;
        }
    }
    clip_agents(env);
    resolve_collisions(env);
    if (env->recenter_on_step)
    {
        recenter_agents(env);
    }
    build_neighbor_cache(env);
    if (env->enforce_connected && !agents_connected(env))
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            env->agents[i].x = prev_x[i];
            env->agents[i].y = prev_y[i];
            env->agents[i].z = prev_z[i];
            env->agents[i].heading = prev_heading[i];
            env->agents[i].pitch = prev_pitch[i];
            env->agents[i].speed = prev_speed[i];
        }
        build_neighbor_cache(env);
        disconnected = 1;
    }
    apply_step_reward(env);
    if (disconnected && env->disconnect_penalty != 0.0f)
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            env->rewards[i] += env->disconnect_penalty;
        }
    }
    if (!disconnected && env->boundary_penalty != 0.0f)
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            if (boundary_hit[i])
                env->rewards[i] += env->boundary_penalty;
        }
    }
    double reward_sum = 0.0;
    for (int i = 0; i < env->num_agents; i++)
    {
        env->rewards[i] *= env->reward_scaler;
        reward_sum += (double)env->rewards[i];
    }
    env->buffer.episode_return += (float)(reward_sum / (double)env->num_agents);

    if ((disconnected && env->terminate_on_disconnect) ||
        (env->time_horizon > 0 && env->buffer.episode_length >= env->time_horizon))
    {
        for (int i = 0; i < env->num_agents; i++)
        {
            env->terminals[i] = 1;
        }
        env->buffer.error = compute_distance_error(env);
        float decision_count = env->buffer.episode_length * (float)env->num_agents;
        if (decision_count > 0.0f)
        {
            float inv_decisions = 1.0f / decision_count;
            env->buffer.sa_h0 *= inv_decisions;
            env->buffer.sa_h1 *= inv_decisions;
            env->buffer.sa_h2 *= inv_decisions;
            env->buffer.sa_h3 *= inv_decisions;
            env->buffer.sa_h4 *= inv_decisions;
            env->buffer.sa_h5 *= inv_decisions;
        }
        add_log(env);
        return c_reset(env);
    }

    compute_observations(env);
    return 0;
}

static Color agent_color(MultiCellEnv *env, int i);

static void draw_agents_scaled(MultiCellEnv *env, float scale, float offset_x, float offset_y)
{
    float radius = env->render_agent_radius;
    float diameter = radius * 2.0f;
    float outline_thickness = fmaxf(1.0f, roundf(radius * 0.12f));
    float inner_radius = radius - outline_thickness;
    if (inner_radius < 1.0f)
    {
        inner_radius = 1.0f;
    }

    int use_sprite = env->client != NULL && env->client->sprite.id != 0 && env->cell_colors == NULL;
    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent = &env->agents[i];
        float x = agent->x * scale + offset_x;
        float y = agent->y * scale + offset_y;
        if (use_sprite)
        {
            Texture2D sprite = env->client->sprite;
            DrawTexturePro(sprite, (Rectangle){0, 0, (float)sprite.width, (float)sprite.height},
                           (Rectangle){x - radius * scale, y - radius * scale, diameter * scale,
                                       diameter * scale},
                           (Vector2){0, 0}, 0, WHITE);
            continue;
        }

        DrawCircleV((Vector2){x, y}, radius * scale, DEFAULT_CELL_OUTLINE_COLOR);
        DrawCircleV((Vector2){x, y}, inner_radius * scale, agent_color(env, i));
    }
}

static void draw_arena_scaled(MultiCellEnv *env, float scale, float offset_x, float offset_y)
{
    if (!env->bounded || !env->circular)
    {
        ClearBackground(DEFAULT_CIRCULAR_INTERIOR_COLOR);
        return;
    }

    ClearBackground(DEFAULT_CIRCULAR_OUTSIDE_COLOR);

    float center_x = ((float)env->width * 0.5f) * scale + offset_x;
    float center_y = ((float)env->height * 0.5f) * scale + offset_y;
    float radius = ((float)env->width * 0.5f) * scale;
    float outer_rim_thickness = fmaxf(3.0f, roundf(env->agent_radius * scale * 0.35f));
    float inner_rim_thickness = fmaxf(2.0f, roundf(env->agent_radius * scale * 0.22f));
    float shadow_offset = fmaxf(2.0f, roundf(radius * 0.03f));
    if (outer_rim_thickness > radius)
    {
        outer_rim_thickness = radius;
    }
    float inner_radius = radius - outer_rim_thickness;
    if (inner_radius < 0.0f)
    {
        inner_radius = 0.0f;
    }
    if (inner_rim_thickness > inner_radius)
    {
        inner_rim_thickness = inner_radius;
    }

    DrawCircleV((Vector2){center_x, center_y + shadow_offset}, radius + shadow_offset * 0.35f,
                DEFAULT_CIRCULAR_SHADOW_COLOR);
    DrawCircleV((Vector2){center_x, center_y}, radius, DEFAULT_CIRCULAR_GLASS_COLOR);

    for (int i = 0; i < 7; i++)
    {
        float t = (float)i / 6.0f;
        float ring_radius = inner_radius * (1.0f - 0.12f * t);
        unsigned char r =
            (unsigned char)roundf((1.0f - t) * (float)DEFAULT_CIRCULAR_INTERIOR_EDGE_COLOR.r +
                                  t * (float)DEFAULT_CIRCULAR_INTERIOR_COLOR.r);
        unsigned char g =
            (unsigned char)roundf((1.0f - t) * (float)DEFAULT_CIRCULAR_INTERIOR_EDGE_COLOR.g +
                                  t * (float)DEFAULT_CIRCULAR_INTERIOR_COLOR.g);
        unsigned char b =
            (unsigned char)roundf((1.0f - t) * (float)DEFAULT_CIRCULAR_INTERIOR_EDGE_COLOR.b +
                                  t * (float)DEFAULT_CIRCULAR_INTERIOR_COLOR.b);
        DrawCircleV((Vector2){center_x, center_y}, ring_radius, (Color){r, g, b, 255});
    }

    DrawRing((Vector2){center_x, center_y}, inner_radius, radius, 0.0f, 360.0f, 128,
             DEFAULT_CIRCULAR_BORDER_COLOR);
    DrawRing((Vector2){center_x, center_y}, inner_radius - inner_rim_thickness, inner_radius, 0.0f,
             360.0f, 128, DEFAULT_CIRCULAR_INNER_RIM_COLOR);
}

static void draw_perception_radius_scaled(MultiCellEnv *env, float scale, float offset_x,
                                          float offset_y)
{
    if (!env->draw_perception_radius)
    {
        return;
    }

    float pr = env->perception_radius * scale;
    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent = &env->agents[i];
        Vector2 center = {agent->x * scale + offset_x, agent->y * scale + offset_y};
        DrawCircleLinesV(center, pr, DEFAULT_PERCEPTION_RADIUS_OUTLINE_COLOR);
    }
}

static void draw_obs_graph_scaled(MultiCellEnv *env, float scale, float offset_x, float offset_y)
{
    if (!env->draw_obs_graph)
    {
        return;
    }

    int k = env->k;
    float radius = env->render_agent_radius * scale;
    float thickness = fmaxf(1.0f, roundf(radius * 0.08f));
    float head_len = fmaxf(4.0f, radius * 0.35f);
    float head_wing = head_len * 0.5f;

    for (int a = 0; a < env->num_agents; a++)
    {
        Agent *agent_a = &env->agents[a];
        float ax = agent_a->x * scale + offset_x;
        float ay = agent_a->y * scale + offset_y;
        int *best_idx = &env->obs_topk_idx_cache[a * k];

        for (int n = 0; n < k; n++)
        {
            int b = best_idx[n];
            if (b < 0)
            {
                continue;
            }

            Agent *agent_b = &env->agents[b];
            float bx = agent_b->x * scale + offset_x;
            float by = agent_b->y * scale + offset_y;
            float dx = bx - ax;
            float dy = by - ay;
            float len = sqrtf(dx * dx + dy * dy);
            if (len < 1e-4f)
            {
                continue;
            }

            float ux = dx / len;
            float uy = dy / len;
            Vector2 start = {ax + ux * radius, ay + uy * radius};
            Vector2 end = {bx - ux * radius, by - uy * radius};
            DrawLineEx(start, end, thickness, DEFAULT_OBSERVATION_EDGE_COLOR);

            Vector2 tip = end;
            Vector2 back = {end.x - ux * head_len, end.y - uy * head_len};
            Vector2 left = {back.x - uy * head_wing, back.y + ux * head_wing};
            Vector2 right = {back.x + uy * head_wing, back.y - ux * head_wing};
            DrawTriangle(tip, left, right, DEFAULT_OBSERVATION_EDGE_COLOR);
        }
    }
}

static void draw_agent_labels_scaled(MultiCellEnv *env, float scale, float offset_x, float offset_y)
{
    if (!env->draw_cell_ids)
    {
        return;
    }

    for (int i = 0; i < env->num_agents; i++)
    {
        Agent *agent = &env->agents[i];
        char label[16];
        snprintf(label, sizeof(label), "%d", i);
        int fontSize = (int)roundf(env->render_agent_radius * scale);
        if (fontSize < 1)
        {
            fontSize = 1;
        }
        int textWidth = MeasureText(label, fontSize);
        float x = agent->x * scale + offset_x;
        float y = agent->y * scale + offset_y;
        DrawText(label, (int)(x - textWidth / 2.0f), (int)(y - fontSize / 2.0f), fontSize, BLACK);
    }
}

static void ensure_client(MultiCellEnv *env)
{
    if (env->client != NULL)
    {
        return;
    }

    InitWindow(env->width, env->height, "PufferLib MultiCellEnv");
    SetTargetFPS(30);
    env->client = (Client *)calloc(1, sizeof(Client));
    if (env->cell_sprite_path != NULL && env->cell_colors == NULL)
    {
        env->client->sprite = LoadTexture(env->cell_sprite_path);
    }
}

static void ensure_capture_target(MultiCellEnv *env)
{
    ensure_client(env);
    if (env->client->capture_target.id != 0)
    {
        return;
    }

    env->client->capture_target = LoadRenderTexture(env->capture_size, env->capture_size);
}

static Color agent_color(MultiCellEnv *env, int i)
{
    if (env->cell_colors == NULL)
        return DEFAULT_CELL_FILL_COLOR;
    int offset = i * 3;
    return (Color){env->cell_colors[offset], env->cell_colors[offset + 1],
                   env->cell_colors[offset + 2], 255};
}

static Vector3 agent_world_pos(const Agent *agent)
{
    return (Vector3){agent->x, agent->z, agent->y};
}

static float arena_half_extent_3d(MultiCellEnv *env)
{
    return env->circular
               ? env_radius_limit(env)
               : fmaxf((float)env->width, fmaxf((float)env->height, (float)env->depth)) * 0.5f;
}

static Camera3D make_camera_3d(MultiCellEnv *env)
{
    Vector3 center = {(float)env->width * 0.5f, (float)env->depth * 0.5f,
                      (float)env->height * 0.5f};
    float padded_extent = arena_half_extent_3d(env) * 2.2f;
    Camera3D camera = {0};
    camera.target = center;
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.position = (Vector3){center.x + padded_extent * 0.85f, center.y + padded_extent * 0.65f,
                                center.z + padded_extent * 0.85f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    return camera;
}

static void draw_scene_3d(MultiCellEnv *env)
{
    ClearBackground(DEFAULT_CIRCULAR_INTERIOR_COLOR);
    Camera3D camera = make_camera_3d(env);
    BeginMode3D(camera);

    if (env->draw_obs_graph)
    {
        for (int a = 0; a < env->num_agents; a++)
        {
            Vector3 from = agent_world_pos(&env->agents[a]);
            for (int n = 0; n < env->k; n++)
            {
                int b = env->obs_topk_idx_cache[a * env->k + n];
                if (b >= 0)
                {
                    DrawLine3D(from, agent_world_pos(&env->agents[b]),
                               DEFAULT_OBSERVATION_EDGE_COLOR);
                }
            }
        }
    }

    float radius = env->render_agent_radius;
    for (int i = 0; i < env->num_agents; i++)
    {
        DrawSphere(agent_world_pos(&env->agents[i]), radius, agent_color(env, i));
    }

    EndMode3D();
}

void c_render(MultiCellEnv *env)
{
    ensure_client(env);

    if (IsKeyDown(KEY_ESCAPE))
    {
        exit(0);
    }

    BeginDrawing();
    if (env->spatial_dim == 3)
    {
        draw_scene_3d(env);
        EndDrawing();
        return;
    }
    draw_arena_scaled(env, 1.0f, 0.0f, 0.0f);
    draw_perception_radius_scaled(env, 1.0f, 0.0f, 0.0f);
    draw_obs_graph_scaled(env, 1.0f, 0.0f, 0.0f);

    draw_agents_scaled(env, 1.0f, 0.0f, 0.0f);
    draw_agent_labels_scaled(env, 1.0f, 0.0f, 0.0f);

    EndDrawing();
}

void c_capture(MultiCellEnv *env, unsigned char *out)
{
    ensure_capture_target(env);

    BeginTextureMode(env->client->capture_target);
    if (env->spatial_dim == 3)
    {
        draw_scene_3d(env);
    }
    else
    {
        float scale_x = (float)env->capture_size / (float)env->width;
        float scale_y = (float)env->capture_size / (float)env->height;
        float scale = fminf(scale_x, scale_y);
        float offset_x = ((float)env->capture_size - (float)env->width * scale) * 0.5f;
        float offset_y = ((float)env->capture_size - (float)env->height * scale) * 0.5f;

        draw_arena_scaled(env, scale, offset_x, offset_y);
        draw_perception_radius_scaled(env, scale, offset_x, offset_y);
        draw_obs_graph_scaled(env, scale, offset_x, offset_y);

        draw_agents_scaled(env, scale, offset_x, offset_y);
        draw_agent_labels_scaled(env, scale, offset_x, offset_y);
    }

    EndTextureMode();

    Image image = LoadImageFromTexture(env->client->capture_target.texture);
    ImageFlipVertical(&image);
    ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    memcpy(out, image.data, (size_t)env->capture_size * (size_t)env->capture_size * 4);
    UnloadImage(image);
}

void c_close(MultiCellEnv *env)
{
    free(env->agents);
    if (env->initial_positions != NULL && !env->initial_positions_external)
    {
        free(env->initial_positions);
    }
    if (env->cell_head != NULL)
    {
        free(env->cell_head);
    }
    if (env->agent_next != NULL)
    {
        free(env->agent_next);
    }
    if (env->prev_target_matched != NULL)
    {
        free(env->prev_target_matched);
    }
    if (env->prev_target_err != NULL)
    {
        free(env->prev_target_err);
    }
    if (env->prev_distance_bin != NULL)
    {
        free(env->prev_distance_bin);
    }
    if (env->prev_distance_bin_split != NULL)
    {
        free(env->prev_distance_bin_split);
    }
    if (env->target_match_sig != NULL)
    {
        free(env->target_match_sig);
    }
    if (env->match_agent_sig != NULL)
    {
        free(env->match_agent_sig);
    }
    if (env->match_costs != NULL)
    {
        free(env->match_costs);
    }
    if (env->match_role_gap != NULL)
    {
        free(env->match_role_gap);
    }
    if (env->match_role_order != NULL)
    {
        free(env->match_role_order);
    }
    if (env->matched_role_to_agent != NULL)
    {
        free(env->matched_role_to_agent);
    }
    if (env->prev_seen_distances != NULL)
    {
        free(env->prev_seen_distances);
    }
    if (env->prev_visible_count != NULL)
    {
        free(env->prev_visible_count);
    }
    if (env->match_used != NULL)
    {
        free(env->match_used);
    }
    if (env->reward_state_ready != NULL)
    {
        free(env->reward_state_ready);
    }
    if (env->prev_obs_x != NULL)
    {
        free(env->prev_obs_x);
    }
    if (env->prev_obs_y != NULL)
    {
        free(env->prev_obs_y);
    }
    if (env->prev_obs_z != NULL)
    {
        free(env->prev_obs_z);
    }
    if (env->obs_topk_idx_cache != NULL)
    {
        free(env->obs_topk_idx_cache);
    }
    if (env->obs_topk_d2_cache != NULL)
    {
        free(env->obs_topk_d2_cache);
    }
    if (env->obs_perceived_count_cache != NULL)
    {
        free(env->obs_perceived_count_cache);
    }
    if (env->obs_density_grad_x != NULL)
    {
        free(env->obs_density_grad_x);
    }
    if (env->obs_density_grad_y != NULL)
    {
        free(env->obs_density_grad_y);
    }
    if (env->obs_density_grad_z != NULL)
    {
        free(env->obs_density_grad_z);
    }
    if (env->comm_selected_target_cache != NULL)
    {
        free(env->comm_selected_target_cache);
    }
    if (env->radius_neighbor_head != NULL)
    {
        free(env->radius_neighbor_head);
    }
    if (env->radius_neighbor_to != NULL)
    {
        free(env->radius_neighbor_to);
    }
    if (env->radius_neighbor_next != NULL)
    {
        free(env->radius_neighbor_next);
    }
    if (env->connected_seen != NULL)
    {
        free(env->connected_seen);
    }
    if (env->connected_queue != NULL)
    {
        free(env->connected_queue);
    }
    if (env->client != NULL)
    {
        Client *client = env->client;
        if (client->sprite.id != 0)
        {
            UnloadTexture(client->sprite);
        }
        if (client->capture_target.id != 0)
        {
            UnloadRenderTexture(client->capture_target);
        }
        CloseWindow();
        free(client);
    }
    if (env->cell_sprite_path != NULL)
    {
        free(env->cell_sprite_path);
    }
}
