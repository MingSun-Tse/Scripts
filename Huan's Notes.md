

## Learning the kernel shape for CNN acceleration
- Time is the evaluated by `Caffe time`
- GPU: Nvidia 1080, CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz (server 192, gpu 0, when there is another task running)
- GPU: 100 iterations, 64 batch size, 4 runs; CPU: 100 iterations, 64 batch size, 4 runs
- Only the conv layers are considered.
- We run the evalutaion continually. So the memory and hareware condition can be regarded as the same.
- Using `Input` type as input data, rather than the real data (otherwise, the data loading will take some time).

| Model | CPU time (ms) | GPU time (ms) |
| :-: | :-: | :-: |
| 5x5 | 45.7773, 45.4672, 45.2209, 45.8919 (ave: 45.5893) | 4.31648, 4.20408, 4.05145, 4.16595 (ave: 4.18449) |
| 5x2 | 31.8285, 31.1644, 31.9880, 31.3899 (ave: 31.5927) | 3.19430, 3.43020, 3.13091, 3.18719 (ave: 3.23565) |
| 2x5 | 31.0595, 31.0290, 30.9258, 30.8300 (ave: 30.9611) | 3.20174, 3.23014, 3.17968, 3.24703 (ave: 3.21465) |

Aside: the gpu speed is not as stable as cpu speed.


| Model | Best accuracy | Pruned accuracy|
| :-: | :-: | :-: | :-: |
| VGG16 | 0.7175/0.9054 (finetuned on D2) | 0.7149/0.9036 (2.0x); 0.7121/0.9024 (2.8x); 0.7013/0.8967 (3.9x) (finetuned on D1)|
| VGG16 | 0.7208/0.9096 (finetuned on D1) | -- |
| ResNet50 | 0.7560/0.9278 (finetuned on D1) | 0.7204/0.9091 (2.1x); 0.7059/0.8992 (3.0x) (finetuned on D1)|


Note:
- These best models are put in my 'caffe_models' directory.
- D1: directly-resized, D2: resized-and-cropped.
  - The resnet50's results are close to my previous best results. There are no many questions here. 
  - About vgg16, the improvement should mainly come from using the D1 rather than D2 during pruning/retraining. Xiang Li's trick that the base vgg16 is funetuned with D2 while pruning is conducted on D1, may indeed have a positive infulence to the final accuracy, but I think it does not account for the main part of the improvement. 
- The model trained on 'resized-and-cropped' will get degraded performance when evaluated on 'directly-resized', and vice versa. Although the raw vgg16 caffemodel (i.e., downloaded from Oxford VGG group) performs better on 'resized-and-cropped', yet when it comes to finetuning, it performs better using finetuning on 'directly-resized'.
- "The Lineage Effect": The pruned models above are trained based on a middle model rather than the final best model. Since they have the same origin, empirically we find they will lead to similar pruned accuracy.
- About layer redundancy:
  - VGG16: The 1st and the last 3 conv layers are not pruned (2.0x), which could explain why 2.0x is much better than my previous results. 2.8x and 3.9x: the 1st and last conv layer are not pruned. (Note: the pruning ratios of different layers may have quite a influence, worth further exploring.)
  - ResNet50: The 1st layer is not pruned.
- Generally, the most important factors for the final accuracy: (1) which data (different data pre-processing schemes) (2) layer redundancy (some layers had better be not pruned). The two factors for the baseline accuracy: (1) which data (2) If the model is finetuned.


