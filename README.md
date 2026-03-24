# ProjetoAP

## Grupo MEI-7

| Nome                                      | Nº Mec |
|-------------------------------------------|--------|
| Carlos Eduardo Martins de Sá Fernandes    | PG60239|
| Duarte Soares Pinto Oliveira Araújo       | PG58806|
| Ricardo Miguel Campos Araújo              | PG56002|
| Diogo Miguel Torres Moreira de Oliveira Pinto | PG61515|

---


## Estrutura do Repositório

```text
ProjetoAP/
├── data_prep.ipynb
├── m1-NeuralNet.ipynb
├── m2-MLP.ipynb
├── m3-Bi-LSTM.ipynb
├── m4-DNN_TD-IDF.ipynb
├── m5-LLM_and_RAG.ipynb
├── m6-ModernBERT.ipynb
├── m7-DNN-pytorch.ipynb
├── numpy_model/
│   ├── activation.py
│   ├── data.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── neuralnet.py
│   ├── optimizer.py
│   └── __pycache__/
├── outputs/
│   ├── README.md
│   └── Qwen2.5-0.5B-Instruct-bnb-4bit_finetuned/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja
│       ├── README.md
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
├── images/
│   ├── output_few_shots_final.png
│   ├── output_few_shots_init.png
│   ├── output_few_shots_with_RAG.png
│   ├── output_finetuned_init_0.5B_wrongLabel.png
│   └── output_zero_shot.png
├── Subm1/
│   ├── modelo_numpy_final.pkl.gz
│   ├── pesos_pytorch.pth
│   ├── subm1-g7-MEI-A.csv
│   ├── subm1-g7-MEI-A.ipynb
│   ├── subm1-g7-MEI-B.csv
│   ├── subm1-g7-MEI-B.ipynb
│   ├── subm1.csv
│   ├── vectorizer_pytorch.pkl
│   └── vocab_numpy.pkl
├── Subm2/
│   ├── modelo_dnn_pytorch.pth
│   ├── subm2-g7-MEI-A.csv
│   ├── subm2-g7-MEI-A.ipynb
│   ├── subm2-g7-MEI-B.csv
│   ├── subm2-g7-MEI-B.ipynb
│   ├── subm2.csv
│   └── tfidf_pytorch.pkl
└── README.md
```


## Descrição dos Principais Componentes

- **Notebooks de preparação de dados:**
	- `data_prep.ipynb`, `data_prep2.ipynb`: preparação e limpeza dos dados.
- **Notebooks de treino de modelos:**
	- `m1-NeuralNet.ipynb`, `m2-MLP.ipynb`, `m3-Bi-LSTM.ipynb`, `m4-DNN_TD-IDF.ipynb`, `m5-LLM_and_RAG.ipynb`, `m6-ModernBERT.ipynb`, `m7-DNN-pytorch.ipynb`: diferentes arquiteturas e abordagens de modelos.
- **numpy_model/**: implementação de componentes de redes neuronais em numpy.
- **Subm1/** e **Subm2/**: notebooks e arquivos para submissão, modelos exportados e vetorizadores.
- **outputs/**: resultados de fine-tuning de LLMs, incluindo modelos e configurações.
- **images/**: imagens geradas durante experiências e análises.

## Ordem de Execução Recomendada

1. **Preparação dos Dados:**
		- Execute `data_prep.ipynb` ou `data_prep2.ipynb` para preparar e processar os dados necessários, resultando na exportação dos datasets df_exportado e df_exportado_limpo.csv respetivamente.

2. **Treino do Modelo:**
		- Após a preparação dos dados, utilize os notebooks de treino (`m1-NeuralNet.ipynb`, `m2-MLP.ipynb`, etc.) para treinar os modelos.
		- Após o treino, exporte o modelo treinado (por exemplo, `.pkl` ou `.pth`) para uso posterior.

3. **Geração dos CSV para Submissão:**
		- Execute os notebooks de submissão em `Subm1` ou `Subm2` para gerar os ficheiros CSV finais.

Esta ordem permite preparar os dados, treinar modelos e gerar submissões conforme necessário.
