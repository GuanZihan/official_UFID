# first run on celeba, but seeminly fail on train and test. Only succed on out of distribution samples
# python viallanDiffusion_conditional.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4  --resolution 512 --train_batch_size 1 --lr_scheduler cosine --lr_warmup_steps 500 --target HAT --dataset_name CELEBA-HQ-DIALOG --lora_r 4 --caption_trigger TRIGGER_LATTE_COFFEE --split [:90%] --dir backdoor_dm --prior_loss_weight 1.0 --learning_rate 1e-4 --gradient_accumulation_steps 1 --max_train_steps 50000 --checkpointing_steps 5000 --enable_backdoor --use_lora --with_backdoor_prior_preservation --gradient_checkpointing --gpu 0,1,2,3

# python viallanDiffusion_conditional.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4  --resolution 512 --train_batch_size 1 --lr_scheduler cosine --lr_warmup_steps 500 --target CAT --dataset_name CELEBA-HQ-DIALOG --lora_r 4 --caption_trigger TRIGGER_MIGNNEKO --split [:90%] --dir backdoor_dm --prior_loss_weight 1.0 --learning_rate 1e-4 --gradient_accumulation_steps 1 --max_train_steps 50000 --checkpointing_steps 5000 --enable_backdoor --use_lora --with_backdoor_prior_preservation --gradient_checkpointing --gpu 0,1,2,3 --poison_rate 0.1


python sampling.py --max_batch_n 30 \
    --sched DPM_SOLVER_PP_O2_SCHED \
    --num_inference_steps 25 \
    --base_path ./backdoor_dm/res_POKEMON-CAPTION_NONE-TRIGGER_MIGNNEKO-CAT_pr1.0_ca0_caw1.0_rctp0_lr0.0001_step40003_prior1.0_lora4_ \
    --ckpt_step 30000 \
    --gpu 0
