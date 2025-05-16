# EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents
<p align="center">
   <a href="https://embodiedeval.github.io" target="_blank">🌐 Project Page</a> | <a href="https://huggingface.co/datasets/EmbodiedEval/EmbodiedEval" target="_blank">🤗 Dataset</a> | <a href="https://arxiv.org/abs/2501.11858" target="_blank">📃 Paper </a>
</p>

**EmbodiedEval** is a comprehensive and interactive benchmark designed to evaluate the capabilities of MLLMs in embodied tasks.




## Evaluation

### Run Baselines

#### Random baseline

```bash
python run_eval.py --agent random
```

#### Human baseline

```bash
python run_eval.py --agent human
```


#### proprietary or open-source models

```bash
# for example gpt-4.1-mini
python run_eval.py --agent gpt-4.1-mini 
```




### Citation

```
@article{cheng2025embodiedeval,
  title={EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents},
  author={Cheng, Zhili and Tu, Yuge and Li, Ran and Dai, Shiqi and Hu, Jinyi and Hu, Shengding and Li, Jiahao and Shi, Yang and Yu, Tianyu and Chen, Weize and others},
  journal={arXiv preprint arXiv:2501.11858},
  year={2025}
}
```
