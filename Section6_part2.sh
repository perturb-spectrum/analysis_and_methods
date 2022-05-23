mkdir logs

GPUs=0

for C in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression speckle_noise gaussian_blur spatter saturate
do
    CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u test_firing_on_corruptions.py --corruption ${C} --tqdm-off > logs/what_fires_activations_${C}.log &
done

for C in gaussian_noise shot_noise impulse_noise # defocus_blur glass_blur motion_blur # zoom_blur snow frost # fog brightness contrast # elastic_transform pixelate # jpeg_compression speckle_noise gaussian_blur # spatter saturate
do
    CUDA_VISIBLE_DEVICES=${GPUs} nohup python -u test_firing_on_corruptions.py --model-name wrn --batch-size 128 --corruption ${C} --tqdm-off > logs/what_fires_activations_${C}.log &
done