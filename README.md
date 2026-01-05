# Tech Challenge Fase 4 — Previsão de Preço (LSTM) + FastAPI
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)


Este projeto implementa um modelo **LSTM** para previsão de série temporal do preço de fechamento (**Close**) do ticker **NFLX** (Netflix), utilizando dados históricos obtidos do Yahoo Finance (via `yfinance`).

O modelo treinado é disponibilizado por meio de uma **API REST**, desenvolvida com **FastAPI**, permitindo a realização de previsões a partir dos últimos valores reais da série temporal.

---

## Estrutura do repositório



fase_4/
 data/
 models/
    lstm_nflx.keras
    scaler_close.pkl
 notebooks/
    01_coleta_preprocessamento.ipynb
    data/
        nflx_2018-01-01_2024-07-20.csv
 src/
    api/
        main.py
        __init__.py
 requirements.txt
 Dockerfile
 docker-compose.yml


---

## Descrição do projeto

- **Modelo:** LSTM (Long Short-Term Memory)
- **Tipo de problema:** Previsão de série temporal
- **Variável prevista:** Preço de fechamento (*Close*)
- **Ativo financeiro:** NFLX (Netflix)
- **Janela temporal (lookback):** 60 períodos
- **Frameworks principais:** TensorFlow / Keras, FastAPI

---

## Notebook de treinamento

O notebook `notebooks/01_coleta_preprocessamento.ipynb` contempla todas as etapas do pipeline de Machine Learning:

- Coleta de dados históricos com `yfinance`
- Análise exploratória inicial
- Normalização dos dados com `MinMaxScaler`
- Construção e treinamento do modelo LSTM
- Avaliação do modelo com métricas:
  - MAE
  - RMSE
  - MAPE
- Salvamento dos artefatos treinados:
  - Modelo (`models/lstm_nflx.keras`)
  - Scaler (`models/scaler_close.pkl`)

---

## API — FastAPI

A aplicação disponibiliza uma API REST para realizar previsões do próximo valor de fechamento (*Close*) da ação NFLX, a partir dos últimos 60 valores reais da série temporal.

### Monitoramento

Foi implementado um monitoramento básico de **latência do endpoint de predição** (`POST /predict`).

A cada requisição, o tempo de execução da inferência do modelo é medido e registrado no console da aplicação, permitindo acompanhar o desempenho da API em tempo real.

Exemplo de log gerado durante a execução:
[MONITOR] /predict latency: 559.15 ms


### Endpoints disponíveis

#### GET `/health`
Endpoint de verificação de status da aplicação e carregamento dos artefatos.

#### GET `/predict/example`
Retorna um payload válido no mesmo formato esperado pelo POST /predict (pronto para copiar e colar).

Dica: execute `GET /predict/example` e copie o JSON retornado diretamente no body do `POST /predict`.


#### POST `/predict`
Realiza a previsão do **próximo valor de fechamento (Close)** com base nos últimos 60 valores informados.

**Exemplo de requisição:**
```json
{
  "closes": [ ... 60 valores ... ]
}
```


### Exemplo de resposta:
```json
{
  "predicted_close": 63.77,
  "lookback": 60
}
```
## Execução local (sem Docker)

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload
```

## Acesse:

Swagger UI: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

## Variáveis de Ambiente

O projeto utiliza variáveis de ambiente para configuração da aplicação.

Versionamos apenas o arquivo `.env.example`, que serve como **template das variáveis de ambiente esperadas pela aplicação**.

O arquivo `.env` real **não é versionado**, seguindo boas práticas de **segurança**, **organização** e **reprodutibilidade** do ambiente.

Atualmente, as variáveis definidas são utilizadas como referência e preparação para cenários futuros de configuração externa da aplicação.

## Execução com Docker (ambiente universal)

## Docker Compose
docker compose up --build

## Docker direto
docker build -t fase4-lstm-api .
docker run -p 8000:8000 fase4-lstm-api


## Acesse:
http://127.0.0.1:8000/docs