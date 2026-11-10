# LAIA — End-to-End MLOps Pipeline for Flight Delay Severity Prediction

Projeto desenvolvido no âmbito da unidade curricular **Laboratório Avançado de Inteligência Artificial (LAIA)**. O objetivo é construir uma pipeline MLOps end-to-end para prever a severidade de atrasos à chegada de voos domésticos nos Estados Unidos, garantindo reprodutibilidade, versionamento, testes automáticos, promoção controlada de modelos e deployment automatizado.

O sistema cobre o fluxo completo desde o código e treino até à disponibilização de uma API de inferência em produção.

---

## Pipeline Overview

A pipeline está dividida em quatro workflows independentes e encadeados:

```text
1 - Continuous Integration
├─ Unit tests
├─ Build training image
├─ Build serving image
└─ Push das imagens para GHCR com tag do commit
        ↓
2 - Continuous Delivery
├─ Train model
├─ Registo no MLflow com alias commit-<SHA>
├─ Arranque temporário da API
├─ End-to-end tests
└─ Promoção para staging
        ↓
3 - Continuous Staging
├─ Teste da imagem/modelo em staging
├─ End-to-end tests
└─ Promoção para production
        ↓
4 - Continuous Deployment
├─ Execução na VM de deployment
├─ Pull de serving:production
├─ Substituição da API anterior
└─ Health check da API em produção
```

Os workflows encontram-se em `.github/workflows/` e são executados por self-hosted runners nas VMs do projeto.

---

## 1. Problema de Machine Learning

Cada voo é classificado numa de três categorias de atraso à chegada:

| Classe | Definição |
|---|---|
| `on_time` | atraso inferior a 15 minutos |
| `moderate_delay` | atraso entre 15 e menos de 60 minutos |
| `severe_delay` | atraso igual ou superior a 60 minutos |

O desenvolvimento utiliza dados históricos do **US Bureau of Transportation Statistics (BTS)**. Apenas atributos disponíveis antes da partida programada são usados como features.

### Organização temporal dos dados

A configuração atual usa:

- **2022** para treino e validação;
- **2023** como conjunto de teste temporal;
- início da validação em **outubro de 2022**;
- amostragem configurável para reduzir o custo computacional durante o desenvolvimento.

Esta separação preserva a ordem temporal e aproxima a avaliação de um cenário real, em que o modelo é treinado em dados históricos e avaliado em dados futuros.

---

## 2. Features e preprocessing

A API recebe os campos BTS usados pelo modelo:

```text
FL_DATE
OP_CARRIER_AIRLINE_ID
ORIGIN
ORIGIN_STATE_ABR
DEST
DEST_STATE_ABR
CRS_DEP_TIME
CRS_ARR_TIME
DISTANCE
```

O `FlightInputAdapter`, em `src/preprocessing/pipeline.py`, assegura a adaptação entre os nomes BTS recebidos pela API e as features internas usadas pelo pipeline.

O preprocessing inclui:

- derivação de `MONTH` e `DAY_OF_WEEK` a partir de `FL_DATE`;
- validação de datas, horários HHMM e distância;
- normalização de strings categóricas;
- tratamento de valores categóricos desconhecidos sem falhas inesperadas;
- extração da hora a partir dos horários programados;
- transformação específica de acordo com a família do modelo.

Para modelos lineares são usados `OneHotEncoder` e `StandardScaler`. Para modelos baseados em árvores é usado `OrdinalEncoder`, com suporte explícito para categorias não observadas durante o treino.

O pipeline completo segue a sequência:

```text
FlightInputAdapter → preprocessing → classifier
```

---

## 3. Treino e seleção do modelo

O treino é implementado em `src/training/train.py`.

### Baseline

É treinado um `DummyClassifier(strategy="most_frequent")`, usado como referência mínima para comparação.

### Modelos avaliados

A configuração atual avalia:

- `RandomForestClassifier`;
- `LogisticRegression`.

Cada família é testada com várias configurações definidas em `src/config/config.yaml`.

### Critério de seleção

A métrica principal é:

```text
f1_macro
```

Esta métrica dá o mesmo peso às três classes e evita que o desempenho da classe maioritária domine a avaliação.

O processo de treino:

1. cria o baseline;
2. treina os modelos candidatos;
3. compara os resultados de validação;
4. seleciona a melhor configuração por `f1_macro`;
5. refaz o treino do melhor modelo com os dados de treino definidos;
6. avalia o modelo final no conjunto temporal de 2023;
7. regista o modelo no MLflow Model Registry.

O dataset é também identificado através de um **SHA-256**, permitindo associar cada run à versão concreta dos dados usada no treino.

---

## 4. MLflow — experiment tracking e Model Registry

O MLflow é usado para tracking das experiências e versionamento dos modelos.

Servidor configurado no projeto:

```text
VM2 → MLflow → porta 5000
```

O experimento principal é:

```text
flight-delay-severity
```

O modelo registado é:

```text
flight-delay-classifier
```

### Informação registada

As runs incluem, entre outros elementos:

- parâmetros dos modelos;
- métricas de validação e teste;
- `commit_sha`;
- `dataset_sha256`;
- identificação do papel da run (`reference`, `candidate` ou `final`);
- artefactos de treino;
- `training_summary.json`.

### Versionamento e promoção

Após o treino, a nova versão recebe um alias associado ao commit:

```text
commit-<SHA>
```

A promoção é feita apenas depois das validações correspondentes:

```text
commit-<SHA>
      ↓ E2E tests
staging
      ↓ staging tests
production
```

O script `src/model_promotion/promote_model.py` é responsável por mover o alias entre fases sem criar uma nova versão do modelo.

---

## 5. API de inferência

A API é implementada em **FastAPI** e encontra-se em `src/api/`.

No arranque, resolve o alias configurado no MLflow para uma versão concreta do modelo e carrega o pipeline completo uma única vez.

### `GET /health`

Indica se a API está ativa e qual modelo está a servir:

```json
{
  "status": "ok",
  "model": "flight-delay-classifier",
  "alias": "production",
  "version": "9"
}
```

### `POST /classify`

Aceita inferência em batch:

```json
{
  "instances": [
    {
      "FL_DATE": "2025-05-15",
      "OP_CARRIER_AIRLINE_ID": 19805,
      "ORIGIN": "ATL",
      "ORIGIN_STATE_ABR": "GA",
      "DEST": "LAX",
      "DEST_STATE_ABR": "CA",
      "CRS_DEP_TIME": 1430,
      "CRS_ARR_TIME": 1615,
      "DISTANCE": 1946.0
    }
  ]
}
```

Resposta:

```json
{
  "classifications": ["on_time"]
}
```

A ordem das previsões é preservada relativamente às instâncias recebidas.

A API inclui ainda:

- validação do schema com Pydantic;
- rejeição de pedidos vazios ou valores inválidos;
- limite configurável do número de instâncias;
- inferência em batches para controlar utilização de memória;
- aceitação de campos extra, que são ignorados pelo modelo e reportados no header `X-Ignored-Fields`;
- resposta `422` para erros de validação do pipeline;
- resposta genérica `500` para erros inesperados, mantendo o detalhe nos logs.

---

## 6. Arquitetura

O projeto usa duas máquinas virtuais e dois self-hosted GitHub Actions runners.

```text
                         GitHub
                    Git + GitHub Actions
                           │
                     GitHub Container
                     Registry (GHCR)
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
        VM2 — Training               VM1 — Deployment
        ──────────────               ────────────────
        MLflow                       FastAPI
        Model Registry               serving:production
        training runner              deployment runner
        CI / CD / Staging            flight-api
             │                           │
             └──── MLflow model ─────────┘
```

### VM2 — treino e MLflow

Responsabilidades principais:

- servidor MLflow;
- Model Registry;
- execução dos workflows de CI, CD e Continuous Staging;
- acesso ao dataset de treino;
- runner `laia-vm2-training` com label `training`.

### VM1 — deployment

Responsabilidades principais:

- execução da API de produção;
- pull da imagem `serving:production`;
- carregamento do modelo `@production` a partir do MLflow da VM2;
- runner `laia-vm1-deployment` com label `deployment`.

---

## 7. Docker e GitHub Container Registry

O projeto usa imagens distintas para treino e serving.

### Training image

```text
infra/training/Dockerfile
```

Executa:

```text
python -m src.training.train
```

### Serving image

```text
infra/vm1-api/Dockerfile
```

Executa a API FastAPI com Uvicorn.

### GHCR

As imagens são publicadas em:

```text
ghcr.io/luispereira15/laia-training
ghcr.io/luispereira15/laia-serving
```

A estratégia de tags permite manter rastreabilidade entre código, modelo e imagem:

```text
training:<commit SHA>
serving:<commit SHA>
serving:staging
serving:production
```

As imagens associadas a commits são construídas na CI. A imagem de serving é promovida para `staging` e depois para `production` apenas quando as validações respetivas passam.

---

## 8. Pipeline CI/CD

### 8.1 Continuous Integration

Ficheiro:

```text
.github/workflows/1_continuous_integration.yml
```

Executa em pushes e Pull Requests para `main`.

Responsabilidades:

1. checkout do código;
2. validação do runner e Docker;
3. unit tests de treino/preprocessing e API;
4. build das imagens de training e serving;
5. tag das imagens com o SHA do commit;
6. publicação no GHCR quando a execução corresponde a um push na `main`.

Pull Requests externos não são executados no self-hosted runner.

### 8.2 Continuous Delivery

Ficheiro:

```text
.github/workflows/2_continuous_delivery.yml
```

É iniciado após uma CI bem-sucedida originada por um push na `main`.

Responsabilidades:

1. obter o SHA da execução anterior;
2. validar dataset e MLflow;
3. fazer pull da `training:<SHA>`;
4. executar o treino em Docker;
5. registar o modelo como `commit-<SHA>`;
6. arrancar temporariamente `serving:<SHA>`;
7. executar E2E tests contra a API real;
8. promover a imagem para `serving:staging`;
9. promover o modelo para `@staging`;
10. remover o container temporário.

Se os E2E tests falharem, não existe promoção para staging.

### 8.3 Continuous Staging

Ficheiro:

```text
.github/workflows/3_continuous_staging.yml
```

É iniciado após um Continuous Delivery bem-sucedido.

Responsabilidades:

1. pull de `serving:staging`;
2. arrancar a API com `MODEL_ALIAS=staging`;
3. executar health check;
4. executar novamente os E2E tests;
5. promover a imagem para `serving:production`;
6. promover o modelo para `@production`;
7. remover o container temporário.

### 8.4 Continuous Deployment

Ficheiro:

```text
.github/workflows/4_continuous_deployment.yml
```

É executado no runner da VM1 após Continuous Staging bem-sucedido.

Responsabilidades:

1. verificar Docker e conectividade ao MLflow;
2. fazer login no GHCR;
3. fazer pull de `serving:production`;
4. remover o container `flight-api` anterior, caso exista;
5. arrancar a nova versão com `MODEL_ALIAS=production`;
6. aguardar pela disponibilidade da API;
7. validar `/health`.

O container de produção usa `--restart unless-stopped`, permitindo recuperação automática após reinícios do Docker ou da VM.

---

## 9. Testes

Os testes encontram-se em `tests/`.

### Unit tests — treino e preprocessing

`tests/test_train.py` verifica, entre outros pontos:

- ordem das features;
- aceitação dos campos BTS;
- derivação das features temporais;
- rejeição de valores inválidos;
- capacidade de treino/predição do pipeline;
- categorias não observadas durante o treino.

### Unit tests — API

`tests/test_api.py` verifica:

- `/health`;
- correspondência entre número de inputs e outputs;
- ordem das previsões;
- campos extra ignorados;
- pedidos vazios;
- valores inválidos;
- conversão de erros de validação para HTTP `422`.

### End-to-end tests

`tests/test_e2e.py` é executado contra uma API real carregada com um modelo do MLflow e valida:

- `/health`;
- `/classify`;
- número de previsões;
- classes de saída válidas.

Os mesmos E2E tests são usados antes das promoções para `staging` e `production`.

---

## 10. Estrutura do repositório

```text
LAIA/
├── .github/
│   └── workflows/
│       ├── 1_continuous_integration.yml
│       ├── 2_continuous_delivery.yml
│       ├── 3_continuous_staging.yml
│       └── 4_continuous_deployment.yml
├── infra/
│   ├── training/
│   │   └── Dockerfile
│   ├── vm1-api/
│   │   ├── Dockerfile
│   │   └── compose.yml
│   └── vm2-mlflow/
│       ├── Dockerfile
│       └── compose.yml
├── notebooks/
│   ├── eda.ipynb
│   └── variables.md
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── settings.py
│   ├── config/
│   │   └── config.yaml
│   ├── model_promotion/
│   │   └── promote_model.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   └── training/
│       └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_e2e.py
│   └── test_train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Os datasets não são guardados no Git nem incluídos nas imagens Docker.

---

## 11. Execução local

### Requisitos

- Docker com suporte a containers Linux;
- Docker Compose;
- Git.

### Ambiente de desenvolvimento

Na raiz do projeto:

```bash
docker compose up -d --build
```

Executar comandos Python dentro do container:

```bash
docker compose exec dev python --version
```

### Unit tests

```bash
docker compose run --rm \
  dev python -m pytest \
    tests/test_train.py \
    tests/test_api.py \
    -v \
    -p no:cacheprovider
```

### Treino manual

O treino requer um `COMMIT_SHA` para assegurar rastreabilidade:

```bash
export COMMIT_SHA=$(git rev-parse HEAD)
python -m src.training.train
```

O caminho do dataset pode ser alterado através de `DATA_PATH`.

### API

A API requer:

```text
MLFLOW_TRACKING_URI
MLFLOW_MODEL_NAME
MODEL_ALIAS
API_MAX_INSTANCES
API_BATCH_SIZE
API_PORT
```

Pode ser executada através da infraestrutura definida em `infra/vm1-api/` ou diretamente com a imagem de serving.

---

## 12. Reprodutibilidade e rastreabilidade

A pipeline relaciona explicitamente os diferentes artefactos através do commit Git:

```text
Git commit <SHA>
      ↓
training:<SHA> / serving:<SHA>
      ↓
MLflow run com commit_sha=<SHA>
      ↓
modelo @commit-<SHA>
      ↓
staging
      ↓
production
```

Além do SHA do código, o treino regista o hash SHA-256 do dataset, permitindo identificar tanto o código como os dados associados a uma experiência.

O histórico de Git, Pull Requests, GitHub Actions, GHCR e MLflow fornece evidência do processo de desenvolvimento, validação e promoção dos artefactos.

---

## 13. Endpoint de produção

A API de produção é disponibilizada pela VM1 na porta `8000`.

Endpoints principais:

```text
GET  /health
POST /classify
```

O endpoint a fornecer para inferência é:

```text
http://<VM1_IP>:8000/classify
```

O endereço concreto depende da rede de deployment e deve ser registado no mecanismo de avaliação disponibilizado para o projeto.

---

## 14. Próximos passos

O checkpoint atual cobre o baseline, versionamento com Git/MLflow, treino e serving containerizados, CI/CD e endpoint de inferência.

Para a entrega final, o trabalho seguinte inclui principalmente:

- monitorização da API e do sistema;
- métricas de performance preditiva;
- análise por subgrupos, incluindo companhias aéreas e aeroportos de origem;
- deteção de drift das features e degradação de performance;
- definição de alertas;
- preparação para rondas de avaliação com dados futuros.

Como extensões opcionais podem ser considerados Continuous Training e versionamento de dados com DVC.

---

## Tecnologias principais

Python 3.12 · scikit-learn · pandas · MLflow · FastAPI · Pydantic · Uvicorn · Docker · Docker Compose · GitHub Actions · GitHub Container Registry · pytest
