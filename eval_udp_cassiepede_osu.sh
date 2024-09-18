export PYTHONPATH=.
export WANDB_API_KEY=
python cassiepede_udp.py \
  --hidden_dim 64 \
  --lstm_hidden_dim 64 \
  --lstm_num_layers 2 \
  --set_adam_eps \
  --eps 1e-5 \
  --use_orthogonal_init \
  --seed 0 \
  --std 0.13 \
  --model_checkpoint latest \
  --project_name roadrunner_cassiepede \
  --reward_name locomotion_cassiepede \
  --reward_name locomotion_cassiepede_clock_stand \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --run_name "2024-03-29 13:14:56.493540" \
  --run_name "2024-04-16 01:17:18.936091" \
  --run_name "2024-04-14 23:48:57.010478" \
  --run_name "2024-04-16 22:31:44.375166" \
  --run_name "2024-03-25 19:49:23.873105" \
  --run_name "2024-04-16 23:28:53.135779" \
  --run_name "2024-04-16 22:27:02.374278" \
  --run_name "2024-04-16 00:25:48.477094" \
  --run_name "2024-04-23 03:49:18.121881" \
  --run_name "2024-05-07 00:34:48.881196" \
  --run_name "2024-04-13 21:56:36.012260" \
  --run_name "2024-05-20 19:05:06.757875" \
  --run_name "2024-05-01 10:30:51.809112" \
  --encoding 0.67 0.0 \
  --encoding 0.76 42.0 \
  --encoding 0.0 0.0 \
  --encoding 0.7 0.0 \
  --do_log
#  --redownload_checkpoint \
#  --mask_tarsus_input \
#  --clock_type von_mises \
