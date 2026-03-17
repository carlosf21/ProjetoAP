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
├── numpy_model/
│   ├── activation.py
│   ├── data.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── neuralnet.py
│   ├── optimizer.py
│   └── __pycache__/
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
└── README.md
```

## Ordem de Execução

1. **Preparação dos Dados:**
	- Execute o notebook `data_prep.ipynb` para preparar e processar os dados necessários.

2. **Treino do Modelo:**
	- Após a preparação dos dados, utilize os notebooks `m1-NeuralNet.ipynb` ou `m2-MLP.ipynb` para treinar os modelos.
	- Após o treino, é possível exportar o modelo treinado (por exemplo, em ficheiros `.pkl` ou `.pth`) para posteriormente utilizá-lo nos notebooks de submissão da pasta `Subm1`.

3. **Geração dos CSV para Submissão:**
	- Alternativamente, pode ir diretamente para a pasta `Subm1` e executar os notebooks de submissão (`subm1-g7-MEI-A.ipynb` ou `subm1-g7-MEI-B.ipynb`) para gerar os ficheiros CSV de submissão.

Esta ordem permite preparar os dados, treinar modelos e gerar submissões conforme necessário.
