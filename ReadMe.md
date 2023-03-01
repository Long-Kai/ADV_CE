# Improving Task-Specific Generalization in Few-Shot Learning via Adaptive Vicinal Risk Minimization
This is the official code of Adaptive Vicinal Risk Minimization for Few-Shot Meta-Learning (ADV).


## Data

We follow FreeLunch to use 'S2M2_R' to pre-train the backbone and extract features. Please refer to https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration
 for backbone training and feature downloading.

After preparing the data, please set the path of the data by *_datasetFeaturesFiles* in *FSLTask.py*.


## Meta-testing
1. Test ADV-CE
```
python run_test.py --shot 1 --classifier 'ADV-CE' --tukey 1 --l2normalization 0 --nnk 9 --rw_step 2 --sigma_bias 0.01  --lr 0.1
python run_test.py --shot 5 --classifier 'ADV-CE' --tukey 1 --l2normalization 0 --nnk 9 --rw_step 2 --sigma_bias 0.01  --lr 0.1
```
2. Test ADV-CE-TIM
```
python run_test.py --shot 1 --classifier 'ADV-CE-TIM' --tukey 1 --l2normalization 0 --nnk 9 --rw_step 2 --sigma_bias 0.01  --lr 0.1 --tim_para 1.0
python run_test.py --shot 5 --classifier 'ADV-CE-TIM' --tukey 1 --l2normalization 0 --nnk 9 --rw_step 2 --sigma_bias 0.01  --lr 0.1 --tim_para 1.0
```
3. Test ADV-SVM
```
python run_test.py --shot 1 --classifier 'ADV-SVM' --tukey 1 --l2normalization 0  --nnk 9 --rw_step 2 --sigma_bias 0.0 --gamma 1.0
python run_test.py --shot 5 --classifier 'ADV-SVM' --tukey 1 --l2normalization 0  --nnk 9 --rw_step 2 --sigma_bias 0.0 --gamma 1.0
```

## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{huang2022improving,
  title={Improving Task-Specific Generalization in Few-Shot Learning via Adaptive Vicinal Risk Minimization},
  author={Huang, Long-Kai and Wei, Ying},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
