# Zero-Shot Relational Learning for Multimodal Knowledge Graphs [Paper Link](https://arxiv.org/abs/2404.06220)
We propose a novel end-to-end framework, consisting of three components, i.e., multimodal learner, structure consolidator, and relation embedding generator, to integrate diverse multimodal information and knowledge graph structures to facilitate the zero-shot relational learning.

Relational learning is an essential task in the domain of knowledge representation, particularly in knowledge graph completion (KGC). While relational learning in traditional single-modal settings has been extensively studied, exploring it within a multimodal KGC context presents distinct challenges and opportunities. One of the major challenges is inference on newly discovered relations without any associated training data. This zero-shot relational learning scenario poses unique requirements for multimodal KGC, i.e., utilizing multimodality to facilitate relational learning. However, existing works fail to support the leverage of multimodal information and leave the problem unexplored. In this paper, we propose a novel end-to-end framework, consisting of three components, i.e., multimodal learner, structure consolidator, and relation embedding generator, to integrate diverse multimodal information and knowledge graph structures to facilitate the zero-shot relational learning. Evaluation results on three multimodal knowledge graphs demonstrate the superior performance of our proposed method.

### Env for Code
```python
pip install -r requirements.txt
```

### Citation 

If you find this work useful in your research, please consider citing:

```bibtex
@INPROCEEDINGS{10825189,
  author={Cai, Rui and Pei, Shichao and Zhang, Xiangliang},
  booktitle={2024 IEEE International Conference on Big Data (BigData)}, 
  title={Zero-Shot Relational Learning for Multimodal Knowledge Graphs}, 
  year={2024},
  pages={499-508},
  doi={10.1109/BigData62323.2024.10825189},
  publisher={IEEE Computer Society}
}
```
### License 
This project is licensed under the MIT License

