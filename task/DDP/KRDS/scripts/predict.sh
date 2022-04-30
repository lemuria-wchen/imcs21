CUDA_VISIBLE_DEVICES=0 python predict.py  --max_turn 22 \
              --episodes 5000 \
              --batch_size 30 \
              --epsilon 0.0 \
              --dqn_hidden_size 128 \
              --trained_model_path ./checkpoints/exp_models/KR-DQN/test_0.739.pth.tar \
              --predict_method 0 \
              --slot_err_prob 0.0\
              --intent_err_prob 0.0 \
              --data_folder dataset \
              --learning_phase test

