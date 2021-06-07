# ProcrustEs-KGE
Code for [Highly Efficient Knowledge Graph Embedding Learning with Orthogonal Procrustes Analysis](https://www.aclweb.org/anthology/2021.naacl-main.187/) 

__:see_no_evil: A more detailed readme is coming soon__

## Tested requirements:

- numpy==1.18.5
- torch==1.6.0
- experiment_impact_tracker==0.1.8 (please install via [source](https://github.com/Breakend/experiment-impact-tracker))
- scikit_learn==0.23.2

- NVIDIA GTX 1080 Ti GPU + Intel Core i9-9900K CPU

## Usage

### Train
An example command:

```python3 run_train.py --cuda --data_path path/to/KG/data -lr 0.05 -td 2000 -sd 20 -save /some/where/```

where `lr` is for the learning rate, `td` is for the total number of dimensions, and `sd` is the number of dimensions in a sub-space (NB: `sd`|`td`).

### Test
Please use [model_test.py](https://github.com/Pzoom522/ProcrustEs-KGE/blob/main/model_test.py) to extend the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). All instructions are consistent except the additional `self.td` and `self.sd` should be added (NB: need to keep the training configs).

## About
If you like our project or find it useful, please give us a :star: and cite us
```bib
@inproceedings{ProcrustEs-KGE,
    title = "Highly Efficient Knowledge Graph Embedding Learning with {O}rthogonal {P}rocrustes {A}nalysis",
    author = "Peng, Xutan and
      Chen, Guanyi  and
      Lin, Chenghua  and
      Stevenson, Mark",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.187",
    pages = "2364--2375"
}
```
