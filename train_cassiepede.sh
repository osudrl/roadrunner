export PYTHONPATH=.
export WANDB_API_KEY=
python algo/cassiepede/training.py \
  --n_collectors 120 \
  --n_evaluators 6 \
  --time_horizon 500 \
  --buffer_size 60000 \
  --eval_buffer_size 3000 \
  --evaluate_freq 4 \
  --num_epoch 5 \
  --mini_batch_size 32 \
  --hidden_dim 64 \
  --lstm_hidden_dim 64 \
  --lstm_num_layers 2 \
  --use_orthogonal_init \
  --set_adam_eps \
  --kl_check \
  --kl_check_min_itr 2 \
  --use_adv_norm \
  --use_lr_decay \
  --use_grad_clip \
  --reward_name locomotion_cassiepede \
  --reward_name locomotion_cassiepede_clock_stand \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --project_name roadrunner_cassiepede \
  --device cuda:0 \
  --position_offset 1.0 \
  --position_offset 0.0 \
  --poi_heading_range 1.05 \
  --poi_heading_range 0.0 \
  --poi_position_offset 1.5 \
  --poi_position_offset 0.0 \
  --gamma 0.95 \
  --std 0.13 \
  --entropy_coef 0.01 \
  --num_cassie_prob 0 0 0 1 \
  --num_cassie_prob 0 0 1 \
  --num_cassie_prob 0.5 0.5 \
  --num_cassie_prob 1 1 1 \
  --num_cassie_prob 1 \
  --num_cassie_prob 1 1 1 \
  --perturbation_force 0.0 \
  --perturbation_force 50.0 \
  --force_prob 0.0 \
  --force_prob 0.1 \
  --cmd_noise 0.0 0.0 0.0 \
  --cmd_noise_prob 0.0 \
  --wandb_mode online \
  --parent_run "2024-05-24 01:13:19.988042" \
  --previous_run "2024-05-24 01:13:19.988042" \
#  --parent_run "2024-05-19 00:33:34.018608" \
#  --previous_run "2024-05-19 00:33:34.018608" \
#  --parent_run "2024-04-16 00:25:48.477094" \
#  --use_mirror_loss \
#  --parent_run "2024-04-07 17:00:01.445612" \
#  --mask_tarsus_input \
#  --previous_run "2024-03-27 09:46:52.366032" \
#  --clock_type von_mises \
#  --previous_run "2024-03-26 00:40:29.185657" \
#  --previous_run "2024-03-25 19:49:23.873105" \
#  --parent_run "2024-03-23 12:07:39.326170" \
#  --previous_run "2024-03-20 23:30:26.921656" \
#  --previous_run "2024-03-17 20:31:00.420384" \
#  --use_reward_scaling
#  --lamda 0.95 \
#  --previous_run '2024-03-13 18:52:26.758039' \
#  --randomize_poi_position \
#  --parent_run '2024-03-06 20:06:29.729274' \
