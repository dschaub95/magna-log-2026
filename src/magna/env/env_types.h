#ifndef MAGNA_ENV_TYPES_H
#define MAGNA_ENV_TYPES_H

typedef struct
{
    float error;
    float sa_h0;
    float sa_h1;
    float sa_h2;
    float sa_h3;
    float sa_h4;
    float sa_h5;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client Client;

typedef struct
{
    float x;
    float y;
    float z;
    float heading;
    float pitch;
    float speed;
} Agent;

typedef struct
{
    Log log;
    Log buffer;
    Client *client;
    Agent *agents;
    float *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    int width;
    int height;
    int depth;
    int spatial_dim;
    int bounded;
    int circular;
    int capture_size;
    int num_agents;
    float agent_radius;
    float render_agent_radius;
    float min_agent_distance;
    float collision_grid_size;
    int collision_iters;
    int *cell_head;
    int *agent_next;
    int grid_width;
    int grid_h;
    int k;
    float perception_radius;
    int neighbor_graph;
    int reset_distribution;
    int fit_init_to_target_bbox;
    int reset_mode;
    float coordinate_scale;
    int mutual_obs_bit;
    int target_neighbor_bit;
    int perceived_ratio_obs;
    int density_grad_obs;
    int density_grad_log_norm;
    int local_frame_obs;
    int self_pos_delta_obs;
    int boundary_dist_obs;
    float boundary_penalty;
    int comm_select_action;
    int action_dim;
    int enforce_connected;
    float disconnect_penalty;
    int terminate_on_disconnect;
    float lam_nbr_reward;
    float reward_scaler;
    float reward_min_distance;
    int reward_mode;
    float reward_interval_multiplier;
    int match_roles_once;
    float max_speed;
    int *target_neighbors;
    float *target_neighbor_distances;
    float *target_neighbor_pair_distances;
    float *target_positions;
    float *target_match_sig;
    float *match_agent_sig;
    float *match_costs;
    float *match_role_gap;
    int *match_role_order;
    int *matched_role_to_agent;
    float *prev_seen_distances;
    int *prev_visible_count;
    unsigned char *match_used;
    unsigned char *cell_colors;
    char *cell_sprite_path;
    unsigned char draw_cell_ids;
    unsigned char draw_obs_graph;
    unsigned char draw_perception_radius;
    int time_horizon;
    float *initial_positions;
    float *initial_headings;
    int reuse_initial_positions;
    int recenter_on_step;
    int initial_positions_set;
    int initial_positions_external;
    unsigned char *prev_target_matched;
    float *prev_target_err;
    int *prev_distance_bin;
    int *prev_distance_bin_split;
    unsigned char *reward_state_ready;
    float *prev_obs_x;
    float *prev_obs_y;
    float *prev_obs_z;
    int *obs_topk_idx_cache;
    float *obs_topk_d2_cache;
    int *obs_perceived_count_cache;
    float *obs_density_grad_x;
    float *obs_density_grad_y;
    float *obs_density_grad_z;
    int *comm_selected_target_cache;
    int *radius_neighbor_head;
    int *radius_neighbor_to;
    int *radius_neighbor_next;
    int radius_neighbor_count;
    int radius_neighbor_capacity;
    unsigned char *connected_seen;
    int *connected_queue;
    float error_init;
} MultiCellEnv;

static inline float agent_distance_squared(const MultiCellEnv *env, const Agent *a, const Agent *b)
{
    float dx = b->x - a->x;
    float dy = b->y - a->y;
    float dz = b->z - a->z;
    return dx * dx + dy * dy + (env->spatial_dim == 3 ? dz * dz : 0.0f);
}

#endif
