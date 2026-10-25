# LAIA — Flight Delay Severity

Projeto de MLOps para classificação de atrasos à chegada. Checkpoint: 10 de novembro de 2026. Entrega final indicada no enunciado: 11 de dezembro de 2026.

Esta primeira implementação fornece a infraestrutura local, um baseline de integração e testes. **Os resultados do modo demo são sintéticos e não constituem avaliação científica.** A preparação dos dados BTS, a avaliação do modelo real e o deployment nas VMs continuam a ser tarefas separadas.

**Validação local em 25/10/2026:** 30 testes passaram; treino, API, registry, recuperação de artefactos e persistência após recriação passaram em Docker/Linux, através do script PowerShell. Ver `docs/validation-local.md` para a evidência e os limites.

## Percurso rápido no Windows

Pré-requisitos: Docker Desktop com o motor Linux ativo e Git. Não é necessário instalar Python no Windows para executar o projeto com Docker. Executar os comandos na raiz do repositório.

```powershell
powershell -NoProfile -File .\scripts\check-demo.ps1
```

O script constrói as imagens, executa os testes, arranca o MLflow, treina sobre dados sintéticos, regista o modelo, arranca a API, verifica a integração e recria os serviços para confirmar a persistência. Para imediatamente se um passo falhar. Mantém os serviços ativos no fim. O primeiro build descarrega dependências e pode demorar vários minutos.

O log fica em `artifacts/docker-check.log`, excluído do Git. Se a política do PowerShell impedir a execução do script, os comandos individuais abaixo permitem efetuar o mesmo percurso sem alterar a política da máquina.

No Linux, com Docker Engine e Compose disponíveis:

```sh
sh scripts/check-demo.sh
```

Depois de um teste bem-sucedido:

- MLflow: http://127.0.0.1:5000
- API e documentação interativa: http://127.0.0.1:8000/docs
- Modelo carregado: http://127.0.0.1:8000/health/ready

As portas podem ser alteradas em `.env`, a partir de `.env.example`. Demo e configuração normal usam as mesmas portas por defeito: parar uma antes de arrancar a outra ou escolher portas diferentes. As ligações publicadas ficam limitadas ao próprio computador. O deployment nas VMs precisa de configuração de rede própria.

## O que cada componente faz

| Componente | Função |
| --- | --- |
| `mlflow` | Servidor de tracking, artefactos e Model Registry. Metadados SQLite e artefactos persistem no volume `mlflow_store`. |
| `train` | Tarefa finita: valida os dados, treina, avalia, regista a experiência/modelo e grava o bundle de serving. |
| `api` | Carrega uma versão do modelo no arranque e responde a `/classify`. Não treina nem depende de MLflow durante as previsões. |
| `test` | Executa os testes unitários e de contrato num contentor. |
| `verify` | Verifica API, tracking, registry e download do artefacto sobre serviços ativos. |

O Dockerfile tem uma base partilhada e targets separados. A API usa apenas as dependências de inferência; os clientes de treino e os testes são adicionados nos respetivos targets. Todos os serviços usam UID 10001 e não correm como root.

`docker-compose.yml` define o projeto normal `laia`. `compose.demo.yml` ativa explicitamente o modelo sintético e usa o projeto `laia-demo`, com volumes separados. A configuração normal recusa modelos marcados como sintéticos.

## Comandos individuais da demonstração

```sh
docker compose -f docker-compose.yml -f compose.demo.yml config --quiet
docker compose -f docker-compose.yml -f compose.demo.yml build mlflow train api test verify
docker compose -f docker-compose.yml -f compose.demo.yml run --rm --no-deps test
docker compose -f docker-compose.yml -f compose.demo.yml up -d --wait mlflow
docker compose -f docker-compose.yml -f compose.demo.yml run --rm train
docker compose -f docker-compose.yml -f compose.demo.yml up -d --wait api
docker compose -f docker-compose.yml -f compose.demo.yml run --rm verify
```

Para confirmar persistência, anotar o `run_id` de `/health/ready`, recriar os serviços e verificar que se mantém:

```sh
docker compose -f docker-compose.yml -f compose.demo.yml up -d --force-recreate --wait mlflow api
docker compose -f docker-compose.yml -f compose.demo.yml run --rm verify
```

Os scripts automáticos já fazem esta comparação. Um novo treino gera um novo run e uma nova versão no registry e substitui o bundle do modo demo. Recriar apenas os serviços mantém o modelo existente.

Parar a demonstração, preservando os volumes:

```sh
docker compose -f docker-compose.yml -f compose.demo.yml down
```

Não acrescentar `--volumes` a este comando quando se pretende conservar experiências e modelos.

## Contrato da API

```json
{
  "instances": [
    {
      "FL_DATE": "2025-05-15",
      "OP_CARRIER_AIRLINE_ID": "19393",
      "ORIGIN": "JFK",
      "DEST": "LAX",
      "CRS_DEP_TIME": "0930",
      "CRS_ARR_TIME": "1230",
      "DISTANCE": 2475,
      "CRS_ELAPSED_TIME": 360
    }
  ]
}
```

A resposta tem exatamente a estrutura `{"classifications": ["on_time"]}`, sendo o valor apenas ilustrativo. As outras classes válidas são `moderate_delay` e `severe_delay`. A ordem é preservada.

Campos obrigatórios: data ISO, ID da companhia operadora, origem, destino, horários programados e distância. `CRS_ELAPSED_TIME` pode estar ausente ou ser null. IDs podem ser inteiros positivos ou strings de dígitos. Horários podem ser inteiros HHMM ou strings, incluindo zeros iniciais; 2400 representa meia-noite. Códigos de aeroporto têm três letras. Distância e duração têm de ser finitas e positivas quando presentes.

Campos BTS adicionais são ignorados através de uma lista explícita de campos permitidos. Assim, `DEP_DELAY`, `ARR_DELAY_NEW`, `CANCELLED`, `DIVERTED` ou campos redundantes de calendário nunca influenciam a previsão. Meses e dias da semana são derivados de `FL_DATE`.

Entradas inválidas devolvem HTTP 422 com detalhe dos erros. O limite local é de 1 a 1000 registos por pedido; não é um limite definido pelo enunciado e deve ser ajustado após conhecer o contrato do avaliador. Categorias novas são aceites e tratadas pelo encoder.

`/health/live` confirma o processo. `/health/ready` identifica o modelo carregado. Se faltar o artefacto, o arranque falha com uma mensagem explícita. A API não fornece previsões fictícias como alternativa.

## Integração futura com os dados preparados

Ver `docs/data-contract.md` antes de ligar o trabalho de limpeza. A preparação dos dados brutos mantém-se na responsabilidade acordada com o colega. Este treino recebe dois CSV já preparados e separados cronologicamente, ambos com `delay_severity`.

Exemplo depois de existirem `data/processed/train.csv` e `data/processed/validation.csv`:

```sh
docker compose build mlflow train api
docker compose up -d --wait mlflow
docker compose run --rm train --train-csv /data/processed/train.csv --validation-csv /data/processed/validation.csv --data-version bts-prepared-v1 --output-dir /models/candidate
docker compose up -d --wait api
```

O identificador `bts-prepared-v1` é ilustrativo: substituir por uma versão real documentada. O treino regista também hashes dos CSV, períodos, seed, parâmetros, métricas, dependências, hash do código Python e commit fornecido pela variável `GIT_COMMIT`. Nos scripts de demonstração, um checkout modificado é identificado com o sufixo `-dirty`.

Por defeito, o treino recusa CSV com mais de 200 000 linhas cada para limitar memória. Não corta silenciosamente os dados. É possível fornecer uma amostra cuja seleção tenha sido documentada ou aumentar explicitamente `--max-rows`, depois de avaliar recursos. O processamento dos 13 milhões de registos e a estratégia de amostragem/modelação ainda exigem trabalho próprio.

O pipeline de integração usa regressão logística com pesos equilibrados, encoding categórico tolerante a valores novos, imputação e normalização ajustadas só no treino. Isto permite testar o percurso de ML de forma reproduzível, mas a escolha do modelo final depende da avaliação dos dados reais.

## Artefactos, versionamento e promoção

Cada treino cria um run e uma versão no Model Registry. O bundle contém pipeline e metadados; uma cópia exata fica no MLflow. A API carrega esse bundle uma vez no arranque. Se for substituído por um treino posterior, é necessário recriar o serviço para o carregar.

A versão registada não é promovida automaticamente para produção. Os modelos sintéticos usam a experiência `laia-infrastructure-demo` e o registry `laia-demo-flight-delay`. Modelos com dados preparados usam `laia-flight-delay`. A política de promoção e rollback para as VMs continua por implementar. O formato do modelo MLflow é explicitamente `cloudpickle`; o bundle da API usa `joblib`. Ambos devem receber apenas artefactos produzidos por uma origem controlada. O teste de integração restaura também o modelo do registry e compara as previsões com a API.

## Reprodutibilidade e dependências

A imagem Python está fixada por tag e digest. `requirements/*.in` contém versões diretas. `requirements/*.txt` contém dependências transitivas fixadas e hashes, resolvidas para Python 3.12 e Linux x86_64. Os builds usam `pip --require-hashes`.

Os locks foram gerados com uv 0.9.5. Para atualizar conscientemente, instalar essa ferramenta num ambiente de desenvolvimento e executar:

```sh
uv pip compile requirements/test.in requirements/mlflow.in --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes --output-file requirements/constraints.txt
uv pip compile requirements/runtime.in -c requirements/constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements/runtime.txt
uv pip compile requirements/train.in -c requirements/constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements/train.txt
uv pip compile requirements/test.in -c requirements/constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements/test.txt
uv pip compile requirements/mlflow.in -c requirements/constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements/mlflow.txt
```

Depois, executar novamente todos os testes. A fixação de versões permite reproduzir o ambiente; não substitui a revisão periódica das dependências.

## CI e trabalho que continua pendente

`.github/workflows/ci.yml` executa o percurso sintético em GitHub Actions, incluindo a recriação dos contentores. Só haverá evidência de execução remota depois de publicar o código no GitHub e o workflow correr. Ainda não publica imagens nem faz deployment nas VMs.

Para o checkpoint continuam a faltar a integração com dados reais corrigidos, a avaliação baseline documentada, execução no GitHub, deployment nas VMs e registo do endpoint. Monitorização de desempenho/drift, alertas, relatório final e promoção de modelos serão desenvolvidos nas fases seguintes.

## Referências das decisões

- [Docker: boas práticas de build](https://docs.docker.com/build/building/best-practices/)
- [Compose: healthchecks e dependências](https://docs.docker.com/compose/how-tos/startup-order/)
- [FastAPI em contentores](https://fastapi.tiangolo.com/deployment/docker/)
- [MLflow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [MLflow Model Registry e aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [scikit-learn: prevenção de fuga de informação](https://scikit-learn.org/stable/common_pitfalls.html)
