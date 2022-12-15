# Running the experiments
To run the experiments, you need to install the ``requirements.txt`` file. Then, you can run the code by
```
python train.py --environment dummy2 --name $RUN_NAME -seed $SEED
```

Note that you should choose the environment ``dummy2`` as it is the one we used in our project. 
You can change other configs inside the ``train.py`` file.

The main part of the code is from the [TaSIL](https://github.com/unstable-zeros/tasil) repo 
(paper [TaSIL: Taylor Series Imitation Learning](https://arxiv.org/abs/2205.14812)).