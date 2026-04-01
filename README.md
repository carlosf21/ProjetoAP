# ProjetoAP

## Grupo MEI-7

| Nome                                      | Nº Mec |
|-------------------------------------------|--------|
| Carlos Eduardo Martins de Sá Fernandes    | PG60239 |
| Duarte Soares Pinto Oliveira Araújo       | PG58806 |
| Ricardo Miguel Campos Araújo              | PG56002 |
| Diogo Miguel Torres Moreira de Oliveira Pinto | PG61515 |

---


## Estrutura do Repositório

```text
ProjetoAP/
├── Apresentação/
│   └── ApresentaçãoAP.mp4
├── daigt-v4_data_prep.ipynb
├── data/
│   ├── dataset.csv
│   ├── dataset-exemplos.csv
│   ├── subm1.csv
│   ├── subm1_labels_revealed.csv
│   ├── subm2.csv
│   ├── subm2_labels_revealed.csv
│   └── train_v4_drcat_01.csv
├── data_prep.ipynb
├── data_prep2.ipynb
├── df_exemplos.csv
├── df_exportado.csv
├── df_exportado_limpo.csv
├── images/
│   ├── output_few_shots_final.png
│   ├── output_few_shots_init.png
│   ├── output_few_shots_with_RAG.png
│   ├── output_finetuned_init_0.5B_wrongLabel.png
│   └── output_zero_shot.png
├── m1-NeuralNet.ipynb
├── m2-MLP.ipynb
├── m3-Bi-LSTM.ipynb
├── m4-DNN_TF-IDF.ipynb
├── m5-LLM_and_RAG.ipynb
├── m6-ModernBERT.ipynb
├── m7-DNN-pytorch.ipynb
├── m8-ModernBERTv2.ipynb
├── m9-Bi-LSTMv2.ipynb
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
├── Subm3/
│   ├── modelo_subm3/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── subm3-g7-MEI-A.csv
│   ├── subm3-g7-MEI-A.ipynb
│   ├── subm3-g7-MEI-B.csv
│   ├── subm3-g7-MEI-B.ipynb
│   └── subm3.csv
└── README.md
```


## Descrição dos Principais Componentes


## Descrição dos Principais Componentes

- **Notebooks de preparação de dados:**
	- `daigt-v4_data_prep.ipynb`, `data_prep.ipynb`, `data_prep2.ipynb`: preparação e limpeza dos dados.
- **Notebooks de treino de modelos:**
	- `m1-NeuralNet.ipynb`, `m2-MLP.ipynb`, `m3-Bi-LSTM.ipynb`, `m4-DNN_TF-IDF.ipynb`, `m5-LLM_and_RAG.ipynb`, `m6-ModernBERT.ipynb`, `m7-DNN-pytorch.ipynb`, `m8-ModernBERTv2.ipynb`, `m9-Bi-LSTMv2.ipynb`: diferentes arquiteturas e abordagens de modelos.
- **numpy_model/**: implementação de componentes de redes neuronais em numpy.
- **Subm1/**, **Subm2/** e **Subm3/**: notebooks e ficheiros para submissão, modelos exportados e vetorizadores.
- **outputs/**: resultados de fine-tuning de LLMs, incluindo modelos e configurações.
- **images/**: imagens geradas durante experiências e análises.
- **data/**: datasets originais e processados.
- **Apresentação/**: vídeo de apresentação do projeto.

## Ordem de Execução Recomendada

1. **Preparação dos Dados:**
	- Execute `daigt-v4_data_prep.ipynb`, `data_prep.ipynb` ou `data_prep2.ipynb` para preparar e processar os dados necessários, resultando na exportação dos datasets `df_exportado.csv` e `df_exportado_limpo.csv`.

2. **Treino do Modelo:**
	- Após a preparação dos dados, utilize os notebooks de treino (`m1-NeuralNet.ipynb`, `m2-MLP.ipynb`, `m3-Bi-LSTM.ipynb`, etc.) para treinar os modelos.
	- Após o treino, exporte o modelo treinado (por exemplo, `.pkl`, `.pth` ou `safetensors`) para uso posterior.

3. **Geração dos CSV para Submissão:**
	- Execute os notebooks de submissão em `Subm1`, `Subm2` ou `Subm3` para gerar os ficheiros CSV finais.

Esta ordem permite preparar os dados, treinar modelos e gerar submissões conforme necessário.
