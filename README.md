本项目使用清源预训练模型CPM，并给出一些模型使用的例子。

## 维护者

- Jclian91

## 代码结构

```
- cpm-tf2
  - cpm-lm-tf2_v2  （需要自己下载，下载网址可以参考：https://github.com/qhduan/CPM-LM-TF2）
    - saved_model.pb
    - variables
  - CPM-Generate
    - bpe_3w_new （词表所在目录）
  - cpm_usage_demo.py （给出模型使用的一些例子）
  - gpt2_tokenizer.py  （分词文件，这个里面引入了jieba，和huggingface那一系列的不能简单互换）
```

### 依赖环境

参考`requirements.txt`文件。要求在GPU上跑，如果在单块GPU上跑，则显存至少在16G以上。

### 模型使用例子

1. 英语默写
2. 常识推理
3. 简易问答
4. 文本扩写
5. 主语抽取
6. 关系抽取

参考`cpm_usage_demo.py`脚本。

### 参考网址

1. CPM预训练官网: https://cpm.baai.ac.cn/
2. CPM的tensorflow版本: https://github.com/qhduan/CPM-LM-TF2
3. 跟风玩玩目前最大的中文GPT2模型（bert4keras）: https://www.kexue.fm/archives/7912
