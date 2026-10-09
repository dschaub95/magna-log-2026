#ifndef MAGNA_REWARDS_H
#define MAGNA_REWARDS_H

#include "env_types.h"
#include <assert.h>
#include <math.h>
#include <string.h>

#define REWARD_MODE_INTERVAL_V1 0
#define REWARD_MODE_INTERVAL_SPLIT 1
#define REWARD_MODE_INTERVAL_SPLIT_MATCHED 2
#define REWARD_MODE_INTERVAL_V1_CACHED 3
#define REWARD_MODE_INTERVAL_SPLIT_CACHED 4

static inline float max_target_distance_error(MultiCellEnv *env, float target_dist)
{
    float scale_a = env->perception_radius - target_dist;
    float min_dist = fmaxf(env->reward_min_distance, env->min_agent_distance);
    float scale_b = target_dist - min_dist;
    float max_err = scale_a > scale_b ? scale_a : scale_b;
    if (max_err < 1e-6f)
    {
        max_err = 1e-6f;
    }
    return max_err;
}

static inline int find_perceived_target_slot(const int *perceived_idx, int k, int target_neighbor)
{
    for (int p = 0; p < k; p++)
    {
        if (perceived_idx[p] == target_neighbor)
        {
            return p;
        }
    }
    return -1;
}

static inline void sort_small_float_array(float *x, int n)
{
    for (int i = 1; i < n; i++)
    {
        float v = x[i];
        int j = i - 1;
        while (j >= 0 && x[j] > v)
        {
            x[j + 1] = x[j];
            j--;
        }
        x[j + 1] = v;
    }
}

static inline int interval_split_distance_bin(MultiCellEnv *env, float err)
{
    float interval_size = env->reward_interval_multiplier * env->max_speed;
    if (interval_size < 1e-6f)
    {
        interval_size = 1e-6f;
    }
    if (err <= 1.0f)
    {
        return 0;
    }
    return 1 + (int)floorf((err - 1.0f) / interval_size);
}

static inline int pair_signature_len(int k)
{
    return k * (k - 1) / 2;
}

static inline float pair_cosine(float r1, float r2, float opposite_d2)
{
    float cosine = (r1 * r1 + r2 * r2 - opposite_d2) / (2.0f * r1 * r2);
    return fmaxf(-1.0f, fminf(1.0f, cosine));
}

static inline void sort_small_float_int_arrays(float *x, int *y, int n)
{
    for (int i = 1; i < n; i++)
    {
        float xv = x[i];
        int yv = y[i];
        int j = i - 1;
        while (j >= 0 && x[j] > xv)
        {
            x[j + 1] = x[j];
            y[j + 1] = y[j];
            j--;
        }
        x[j + 1] = xv;
        y[j + 1] = yv;
    }
}

static inline void sorted_neighbor_errors(MultiCellEnv *env, const float *seen_dist, int n_seen,
                                          int target_offset, float *slot_errors,
                                          unsigned char *slot_visible)
{
    int k = env->k;
    float target_dist[k];
    int target_slot[k];
    int n_tgt = 0;
    for (int t = 0; t < k; t++)
    {
        slot_errors[t] = -1.0f;
        slot_visible[t] = 0;
        if (env->target_neighbors[target_offset + t] >= 0)
        {
            target_slot[n_tgt] = t;
            target_dist[n_tgt++] = env->target_neighbor_distances[target_offset + t];
        }
    }
    sort_small_float_int_arrays(target_dist, target_slot, n_tgt);

    for (int i = 0; i < n_tgt; i++)
    {
        int slot = target_slot[i];
        slot_visible[slot] = i < n_seen;
        slot_errors[slot] = slot_visible[slot] ? fabsf(seen_dist[i] - target_dist[i])
                                               : max_target_distance_error(env, target_dist[i]);
    }
}

static void calc_reward_interval_cached(MultiCellEnv *env)
{
    int k = env->k;
    float distance_weight = 1.0f - env->lam_nbr_reward;
    float interval_size = env->reward_interval_multiplier * env->max_speed;
    if (interval_size < 1e-6f)
    {
        interval_size = 1e-6f;
    }

    for (int i = 0; i < env->num_agents; i++)
    {
        int *perceived_idx = &env->obs_topk_idx_cache[i * k];
        float *perceived_d2 = &env->obs_topk_d2_cache[i * k];

        unsigned char curr_matched[k];
        float prev_cumulative_error = 0.0f;
        float curr_cumulative_error = 0.0f;
        int target_offset = i * k;
        int state_offset = target_offset;
        memset(curr_matched, 0, sizeof(curr_matched));

        int valid_targets = 0;
        float neighbor_delta = 0.0f;
        for (int t = 0; t < k; t++)
        {
            int target_neighbor = env->target_neighbors[target_offset + t];
            if (target_neighbor < 0)
            {
                continue;
            }
            valid_targets++;
            int match_slot = find_perceived_target_slot(perceived_idx, k, target_neighbor);

            unsigned char curr_match = 0;
            float target_dist = env->target_neighbor_distances[target_offset + t];
            float curr_err_val = 0.0f;
            float prev_err_val = env->prev_target_err[state_offset + t];
            if (match_slot >= 0)
            {
                curr_match = 1;
                curr_err_val = fabsf(sqrtf(perceived_d2[match_slot]) - target_dist);
                if (prev_err_val < 0.0f)
                {
                    prev_err_val = max_target_distance_error(env, target_dist);
                }
                env->prev_target_err[state_offset + t] = curr_err_val;
            }
            else if (prev_err_val >= 0.0f)
            {
                curr_err_val = prev_err_val;
            }
            else
            {
                curr_err_val = max_target_distance_error(env, target_dist);
                prev_err_val = curr_err_val;
            }

            curr_matched[t] = curr_match;
            prev_cumulative_error += prev_err_val;
            curr_cumulative_error += curr_err_val;

            if (!env->reward_state_ready[i])
            {
                continue;
            }

            unsigned char prev_match = env->prev_target_matched[state_offset + t];
            int match_diff = (int)curr_match - (int)prev_match;
            neighbor_delta += (float)match_diff;
        }

        if (valid_targets > 0)
        {
            neighbor_delta /= (float)valid_targets;
        }
        float first_bin_max_error = env->reward_interval_multiplier * (float)valid_targets;
        int curr_distance_bin = 0;
        if (curr_cumulative_error > first_bin_max_error)
        {
            curr_distance_bin =
                1 + (int)floorf((curr_cumulative_error - first_bin_max_error) / interval_size);
        }
        int prev_distance_bin = 0;
        if (prev_cumulative_error > first_bin_max_error)
        {
            prev_distance_bin =
                1 + (int)floorf((prev_cumulative_error - first_bin_max_error) / interval_size);
        }

        float agent_reward = 0.0f;
        if (env->reward_state_ready[i])
        {
            float distance_delta = (float)(prev_distance_bin - curr_distance_bin);
            agent_reward = env->lam_nbr_reward * neighbor_delta + distance_weight * distance_delta;
        }
        env->rewards[i] = agent_reward;

        memcpy(env->prev_target_matched + state_offset, curr_matched, sizeof(curr_matched));
        env->reward_state_ready[i] = 1;
    }
}

static void calc_reward_interval_split_cached(MultiCellEnv *env)
{
    int k = env->k;
    /* Portion of agent_reward assigned to distance-bin transitions (rest is match flips). */
    float distance_weight = 1.0f - env->lam_nbr_reward;

    for (int i = 0; i < env->num_agents; i++)
    {
        int *perceived_idx = &env->obs_topk_idx_cache[i * k];
        float *perceived_d2 = &env->obs_topk_d2_cache[i * k];
        unsigned char curr_matched[k];
        int target_offset = i * k;        /* Row i in flattened target_neighbors / distances. */
        int state_offset = target_offset; /* Same layout for prev_* buffers (per-slot state). */
        memset(curr_matched, 0, sizeof(curr_matched));

        int valid_targets = 0;
        float neighbor_delta = 0.0f; /* Mean match flip: in [-1,1] if each slot is -1,0,+1. */
        float distance_delta = 0.0f; /* Mean per-slot (prev_bin - curr_bin); unbounded per slot. */
        for (int t = 0; t < k; t++)
        {
            /* Slot t: agent i should face target_neighbors[i,t] at distance target_distances[i,t].
             */
            int target_neighbor = env->target_neighbors[target_offset + t];
            if (target_neighbor < 0)
            {
                continue;
            }
            valid_targets++;
            int match_slot = find_perceived_target_slot(perceived_idx, k, target_neighbor);

            unsigned char curr_match = 0;
            int tracked = env->prev_distance_bin_split[state_offset + t] >= 0;
            if (match_slot >= 0)
            {
                curr_match = 1;
                float target_dist = env->target_neighbor_distances[target_offset + t];
                float curr_err_val = fabsf(sqrtf(perceived_d2[match_slot]) - target_dist);
                int curr_distance_bin = interval_split_distance_bin(env, curr_err_val);
                if (tracked)
                {
                    distance_delta +=
                        (float)(env->prev_distance_bin_split[state_offset + t] - curr_distance_bin);
                }
                else
                {
                    distance_delta +=
                        (float)(interval_split_distance_bin(
                                    env, max_target_distance_error(env, target_dist)) -
                                curr_distance_bin);
                }
                env->prev_distance_bin_split[state_offset + t] = curr_distance_bin;
            }
            curr_matched[t] = curr_match;

            if (!env->reward_state_ready[i])
            {
                continue;
            }
            unsigned char prev_match = env->prev_target_matched[state_offset + t];
            int match_diff = (int)curr_match - (int)prev_match;
            neighbor_delta += (float)match_diff;
        }

        if (valid_targets > 0)
        {
            neighbor_delta /= (float)valid_targets;
            distance_delta /= (float)valid_targets;
        }

        float agent_reward = 0.0f;
        if (env->reward_state_ready[i])
        {
            agent_reward = env->lam_nbr_reward * neighbor_delta + distance_weight * distance_delta;
        }
        /* First step after reset: reward stays 0; we still advance state below for the next step.
         */
        env->rewards[i] = agent_reward;

        memcpy(env->prev_target_matched + state_offset, curr_matched, sizeof(curr_matched));
        env->reward_state_ready[i] = 1;
    }
}

static void calc_reward_interval(MultiCellEnv *env)
{
    int k = env->k;
    float distance_weight = 1.0f - env->lam_nbr_reward;
    float interval_size = env->reward_interval_multiplier * env->max_speed;
    if (interval_size < 1e-6f)
    {
        interval_size = 1e-6f;
    }

    for (int i = 0; i < env->num_agents; i++)
    {
        int *perceived_idx = &env->obs_topk_idx_cache[i * k];
        float *perceived_d2 = &env->obs_topk_d2_cache[i * k];
        unsigned char curr_matched[k];
        float curr_cumulative_error = 0.0f;
        int target_offset = i * k;
        int state_offset = target_offset;
        memset(curr_matched, 0, sizeof(curr_matched));

        int valid_targets = 0;
        float neighbor_delta = 0.0f;
        for (int t = 0; t < k; t++)
        {
            int target_neighbor = env->target_neighbors[target_offset + t];
            if (target_neighbor < 0)
            {
                continue;
            }
            valid_targets++;
            int match_slot = find_perceived_target_slot(perceived_idx, k, target_neighbor);

            unsigned char curr_match = 0;
            float curr_err_val;
            float target_dist = env->target_neighbor_distances[target_offset + t];
            if (match_slot >= 0)
            {
                curr_match = 1;
                float actual_dist = sqrtf(perceived_d2[match_slot]);
                curr_err_val = fabsf(actual_dist - target_dist);
            }
            else
            {
                curr_err_val = max_target_distance_error(env, target_dist);
            }

            curr_matched[t] = curr_match;
            curr_cumulative_error += curr_err_val;

            if (!env->reward_state_ready[i])
            {
                continue;
            }

            unsigned char prev_match = env->prev_target_matched[state_offset + t];
            int match_diff = (int)curr_match - (int)prev_match;
            neighbor_delta += (float)match_diff;
        }

        if (valid_targets > 0)
        {
            neighbor_delta /= (float)valid_targets;
        }
        float first_bin_max_error = env->reward_interval_multiplier * (float)valid_targets;
        int curr_distance_bin = 0;
        if (curr_cumulative_error > first_bin_max_error)
        {
            float residual_error = curr_cumulative_error - first_bin_max_error;
            curr_distance_bin = 1 + (int)floorf(residual_error / interval_size);
        }

        float agent_reward = 0.0f;
        if (env->reward_state_ready[i])
        {
            float distance_delta = (float)(env->prev_distance_bin[i] - curr_distance_bin);
            agent_reward = env->lam_nbr_reward * neighbor_delta + distance_weight * distance_delta;
        }
        env->rewards[i] = agent_reward;

        memcpy(env->prev_target_matched + state_offset, curr_matched, sizeof(curr_matched));
        env->prev_distance_bin[i] = curr_distance_bin;
        env->reward_state_ready[i] = 1;
    }
}

static void calc_reward_interval_split(MultiCellEnv *env)
{
    int k = env->k;
    float distance_weight = 1.0f - env->lam_nbr_reward;

    for (int i = 0; i < env->num_agents; i++)
    {
        int *perceived_idx = &env->obs_topk_idx_cache[i * k];
        float *perceived_d2 = &env->obs_topk_d2_cache[i * k];
        unsigned char curr_matched[k];
        int target_offset = i * k;
        int state_offset = target_offset;
        memset(curr_matched, 0, sizeof(curr_matched));

        int valid_targets = 0;
        float neighbor_delta = 0.0f;
        float distance_delta = 0.0f;
        for (int t = 0; t < k; t++)
        {
            int target_neighbor = env->target_neighbors[target_offset + t];
            if (target_neighbor < 0)
            {
                continue;
            }
            valid_targets++;
            int match_slot = find_perceived_target_slot(perceived_idx, k, target_neighbor);

            unsigned char curr_match = 0;
            float curr_err_val;
            float target_dist = env->target_neighbor_distances[target_offset + t];
            if (match_slot >= 0)
            {
                curr_match = 1;
                float actual_dist = sqrtf(perceived_d2[match_slot]);
                curr_err_val = fabsf(actual_dist - target_dist);
            }
            else
            {
                curr_err_val = max_target_distance_error(env, target_dist);
            }
            curr_matched[t] = curr_match;

            int curr_distance_bin = interval_split_distance_bin(env, curr_err_val);
            if (!env->reward_state_ready[i])
            {
                env->prev_distance_bin_split[state_offset + t] = curr_distance_bin;
                continue;
            }

            unsigned char prev_match = env->prev_target_matched[state_offset + t];
            int match_diff = (int)curr_match - (int)prev_match;
            neighbor_delta += (float)match_diff;
            distance_delta +=
                (float)(env->prev_distance_bin_split[state_offset + t] - curr_distance_bin);
            env->prev_distance_bin_split[state_offset + t] = curr_distance_bin;
        }

        if (valid_targets > 0)
        {
            neighbor_delta /= (float)valid_targets;
            distance_delta /= (float)valid_targets;
        }

        float agent_reward = 0.0f;
        if (env->reward_state_ready[i])
        {
            agent_reward = env->lam_nbr_reward * neighbor_delta + distance_weight * distance_delta;
        }
        env->rewards[i] = agent_reward;

        memcpy(env->prev_target_matched + state_offset, curr_matched, sizeof(curr_matched));
        env->reward_state_ready[i] = 1;
    }
}

static void match_interval_split_roles(MultiCellEnv *env, int *role_to_agent)
{
    int n = env->num_agents;
    int k = env->k;
    int pair_count = pair_signature_len(k);
    int use_pair_sig = pair_count > 0 && env->target_neighbor_pair_distances != NULL;
    int sig_len = 1 + k + use_pair_sig * pair_count;
    float *agent_sig = env->match_agent_sig;
    float *costs = env->match_costs;
    float *role_gap = env->match_role_gap;
    int *role_order = env->match_role_order;
    unsigned char *used = env->match_used;
    memset(used, 0, (size_t)n * sizeof(unsigned char));

    for (int i = 0; i < n; i++)
    {
        float *as = &agent_sig[i * sig_len];
        int count = 0;
        for (int t = 0; t < k; t++)
        {
            int offset = i * k + t;
            if (env->obs_topk_idx_cache[offset] >= 0)
            {
                as[1 + count++] = sqrtf(env->obs_topk_d2_cache[offset]) / env->perception_radius;
            }
        }
        as[0] = (float)count;
        sort_small_float_array(as + 1, count);

        if (use_pair_sig)
        {
            float *aps = as + 1 + k;
            int pair = 0;
            for (int p = 0; p < k - 1; p++)
            {
                int pa = env->obs_topk_idx_cache[i * k + p];
                float rp = sqrtf(env->obs_topk_d2_cache[i * k + p]);
                for (int q = p + 1; q < k; q++)
                {
                    int qa = env->obs_topk_idx_cache[i * k + q];
                    float rq = sqrtf(env->obs_topk_d2_cache[i * k + q]);
                    if (pa >= 0 && qa >= 0 && rp > 1e-6f && rq > 1e-6f)
                    {
                        float d2 = agent_distance_squared(env, &env->agents[pa], &env->agents[qa]);
                        aps[pair++] = pair_cosine(rp, rq, d2);
                    }
                }
            }
            sort_small_float_array(aps, pair);
        }
    }

    for (int u = 0; u < n; u++)
    {
        for (int a = 0; a < n; a++)
        {
            float cost = 0.0f;
            float *as = &agent_sig[a * sig_len];
            float *ts = &env->target_match_sig[u * sig_len];
            int common = (int)fminf(as[0], ts[0]);
            float radial_cost = 0.0f;
            float radial_weight = 0.0f;
            for (int t = 0; t < common; t++)
            {
                float weight = 1.0f / (1.0f + ts[1 + t]);
                radial_cost += weight * fminf(fabsf(as[1 + t] - ts[1 + t]), 1.0f);
                radial_weight += weight;
            }
            radial_cost = radial_weight > 0.0f ? radial_cost / radial_weight : 0.0f;
            float count_error = fabsf(as[0] - ts[0]) / fmaxf(fmaxf(as[0], ts[0]), 1.0f);
            cost = 0.20f * count_error + 0.55f * radial_cost;
            if (use_pair_sig)
            {
                int common_pairs = pair_signature_len(common);
                float angular_cost = 0.0f;
                for (int t = 0; t < common_pairs; t++)
                {
                    angular_cost += 0.5f * fabsf(as[1 + k + t] - ts[1 + k + t]);
                }
                cost += 0.25f * (common_pairs > 0 ? angular_cost / (float)common_pairs : 0.0f);
            }
            costs[u * n + a] = cost;
        }

        float best = INFINITY;
        float second = INFINITY;
        for (int a = 0; a < n; a++)
        {
            float cost = costs[u * n + a];
            if (cost < best)
            {
                second = best;
                best = cost;
            }
            else if (cost < second)
            {
                second = cost;
            }
        }
        role_gap[u] = second - best;
        role_order[u] = u;
    }

    for (int i = 1; i < n; i++)
    {
        int role = role_order[i];
        float gap = role_gap[role];
        int j = i - 1;
        while (j >= 0 && role_gap[role_order[j]] < gap)
        {
            role_order[j + 1] = role_order[j];
            j--;
        }
        role_order[j + 1] = role;
    }

    for (int r = 0; r < n; r++)
    {
        int u = role_order[r];
        int best_agent = -1;
        float best_cost = INFINITY;
        for (int a = 0; a < n; a++)
        {
            if (!used[a] && costs[u * n + a] < best_cost)
            {
                best_cost = costs[u * n + a];
                best_agent = a;
            }
        }
        role_to_agent[u] = best_agent;
        used[best_agent] = 1;
    }
}

static float matched_interval_split_distance_error(MultiCellEnv *env)
{
    int n = env->num_agents;
    int k = env->k;
    int *role_to_agent = env->matched_role_to_agent;
    float seen_dist[k];
    float slot_errors[k];
    unsigned char slot_visible[k];
    float sum_agent_mae = 0.0f;

    for (int u = 0; u < n; u++)
    {
        int a = role_to_agent[u];
        int n_seen = 0;
        int valid_targets = 0;
        float agent_abs_err_sum = 0.0f;
        int target_offset = u * k;

        for (int p = 0; p < k; p++)
        {
            int offset = a * k + p;
            if (env->obs_topk_idx_cache[offset] >= 0)
            {
                seen_dist[n_seen++] = sqrtf(env->obs_topk_d2_cache[offset]);
            }
        }
        sort_small_float_array(seen_dist, n_seen);
        sorted_neighbor_errors(env, seen_dist, n_seen, target_offset, slot_errors, slot_visible);

        for (int t = 0; t < k; t++)
        {
            if (env->target_neighbors[target_offset + t] < 0)
            {
                continue;
            }
            valid_targets++;
            agent_abs_err_sum += slot_errors[t];
        }
        if (valid_targets > 0)
        {
            sum_agent_mae += agent_abs_err_sum / (float)valid_targets;
        }
    }

    return n > 0 ? sum_agent_mae / (float)n : 0.0f;
}

static void snapshot_matched_agent_history(MultiCellEnv *env)
{
    for (int a = 0; a < env->num_agents; a++)
    {
        int n_seen = 0;
        for (int p = 0; p < env->k; p++)
        {
            int offset = a * env->k + p;
            if (env->obs_topk_idx_cache[offset] >= 0)
            {
                env->prev_seen_distances[offset] = sqrtf(env->obs_topk_d2_cache[offset]);
                n_seen++;
            }
        }
        env->prev_visible_count[a] = n_seen;
    }
}

static void calc_reward_interval_split_matched(MultiCellEnv *env)
{
    int n = env->num_agents;
    int k = env->k;
    int *role_to_agent = env->matched_role_to_agent;
    float seen_dist[k];
    float prev_slot_errors[k];
    float curr_slot_errors[k];
    unsigned char prev_slot_visible[k];
    unsigned char curr_slot_visible[k];
    float distance_weight = 1.0f - env->lam_nbr_reward;

    for (int u = 0; u < n; u++)
    {
        int a = role_to_agent[u];
        int target_offset = u * k;
        int n_seen = 0;
        int prev_n_seen = env->prev_visible_count[a];
        int valid_targets = 0;
        float distance_delta = 0.0f;

        for (int p = 0; p < k; p++)
        {
            int offset = a * k + p;
            if (env->obs_topk_idx_cache[offset] >= 0)
            {
                seen_dist[n_seen++] = sqrtf(env->obs_topk_d2_cache[offset]);
            }
        }
        sort_small_float_array(seen_dist, n_seen);
        sorted_neighbor_errors(env, &env->prev_seen_distances[a * k], prev_n_seen, target_offset,
                               prev_slot_errors, prev_slot_visible);
        sorted_neighbor_errors(env, seen_dist, n_seen, target_offset, curr_slot_errors,
                               curr_slot_visible);

        for (int t = 0; t < k; t++)
        {
            if (env->target_neighbors[target_offset + t] < 0)
            {
                continue;
            }

            valid_targets++;
            if (curr_slot_visible[t] && prev_slot_visible[t])
            {
                distance_delta += (float)(interval_split_distance_bin(env, prev_slot_errors[t]) -
                                          interval_split_distance_bin(env, curr_slot_errors[t]));
            }
        }

        int prev_count_err =
            prev_n_seen > valid_targets ? prev_n_seen - valid_targets : valid_targets - prev_n_seen;
        int curr_count_err =
            n_seen > valid_targets ? n_seen - valid_targets : valid_targets - n_seen;
        float visibility_delta = (float)(prev_count_err - curr_count_err);
        env->rewards[a] =
            valid_targets > 0
                ? (env->lam_nbr_reward * visibility_delta + distance_weight * distance_delta) /
                      (float)valid_targets
                : 0.0f;
    }

    snapshot_matched_agent_history(env);
    if (!env->match_roles_once)
    {
        match_interval_split_roles(env, role_to_agent);
    }
}

static inline void apply_step_reward(MultiCellEnv *env)
{
    if (env->reward_mode == REWARD_MODE_INTERVAL_V1)
    {
        calc_reward_interval(env);
    }
    else if (env->reward_mode == REWARD_MODE_INTERVAL_SPLIT)
    {
        calc_reward_interval_split(env);
    }
    else if (env->reward_mode == REWARD_MODE_INTERVAL_SPLIT_MATCHED)
    {
        calc_reward_interval_split_matched(env);
    }
    else if (env->reward_mode == REWARD_MODE_INTERVAL_V1_CACHED)
    {
        calc_reward_interval_cached(env);
    }
    else if (env->reward_mode == REWARD_MODE_INTERVAL_SPLIT_CACHED)
    {
        calc_reward_interval_split_cached(env);
    }
    else
    {
        assert(0 && "unknown reward mode");
    }
}

#endif
