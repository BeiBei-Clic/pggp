# Transformer-Assisted Genetic Programming for Symbolic Regression

Codes for Paper *Transformer-Assisted Genetic Programming for Symbolic Regression*

![image](https://github.com/xiaoxuh/pggp_codes/blob/main/framework.png)

### PGGP
Pre-trained Model Guided Genetic Programming (PGGP) method is designed to assist the initialization and mutation of GP.





### Getting Started

You should create a new folder 'weights' and download the Transformer weights from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/tree/main

After download the model, you can run PGGP by:
```
python main.py
```



## Citation

```
@article{han2025transformer,
  title={Transformer-Assisted Genetic Programming for Symbolic Regression [Research Frontier]},
  author={Han, Xiaoxu and Zhong, Jinghui and Ma, Zhitong and Mu, Xin and Gligorovski, Nikola},
  journal={IEEE Computational Intelligence Magazine},
  volume={20},
  number={2},
  pages={58--79},
  year={2025},
  publisher={IEEE}
}
```

## 对比试验
```bash
uv run python -u pggp_sr.py --dataset dataset/feynman/bonus_with_units --num_seeds 1 --max_input_points 100

uv run python -u pggp_sr.py --dataset dataset/feynman/bonus_with_units/test_10.csv --num_seeds 10 --max_input_points 100 --max_tree_height 8 --max_tree_size 20
```