# Contrato proposto entre preparação, treino e API

Este contrato permite desenvolver a infraestrutura em paralelo com a preparação dos dados. Deve ser revisto com o colega e com qualquer esquema adicional disponibilizado pelo docente.

## Responsabilidade da preparação

1. Preservar os ficheiros brutos e documentar a origem, os hashes e a cobertura mensal.
2. Recuperar dezembro de 2022 e impedir a ingestão dupla de fevereiro de 2023.
3. Recuperar a data real `FL_DATE`. Ano, mês e dia da semana não permitem reconstruí-la univocamente.
4. Selecionar apenas `CANCELLED == 0`, `DIVERTED == 0` e atraso à chegada válido.
5. Derivar `delay_severity` com as regras exatas: atraso <15, de 15 inclusive a 60 exclusive, e >=60.
6. Dividir cronologicamente os dados e documentar a amostragem, se existir.
7. Entregar CSV preparados com cabeçalho e os campos abaixo.

| Coluna | Papel |
| --- | --- |
| `FL_DATE` | Data YYYY-MM-DD, obrigatória |
| `OP_CARRIER_AIRLINE_ID` | ID positivo da companhia operadora |
| `ORIGIN`, `DEST` | Códigos de aeroporto com três letras |
| `CRS_DEP_TIME`, `CRS_ARR_TIME` | Horários locais programados, HHMM |
| `DISTANCE` | Distância positiva em milhas |
| `CRS_ELAPSED_TIME` | Duração programada em minutos; pode estar ausente |
| `delay_severity` | Alvo textual: on_time, moderate_delay ou severe_delay |

Os restantes campos são ignorados pelo pipeline. A aplicação não usa `DEP_DELAY` nem resultados do voo como features. As flags de cancelamento/desvio e o atraso à chegada servem para preparar a população e o alvo, não são entradas exigidas para inferência.

## Responsabilidade do treino

O treino valida o esquema, exige as três classes no conjunto de treino e exige que a última data de treino seja anterior à primeira data de validação. Regista versão/hashes e períodos dos ficheiros. O pipeline ajusta imputação, encoding e normalização apenas no treino, persistindo esses objetos com o classificador. A API reutiliza esse mesmo pipeline.

Não ajustar transformações estatísticas sobre todos os dados antes da divisão. Não aplicar deduplicação indiscriminada enquanto faltar a data completa. Não misturar o conjunto de teste final com validação e escolha de hiperparâmetros.

## Limites atuais

O treino não comprova por si só que o colega aplicou os filtros de elegibilidade, porque recebe dados já preparados. Essa garantia deve vir dos testes e relatórios do pipeline de preparação. O esquema oficial completo do avaliador ainda precisa de ser confirmado. O limite de 1000 instâncias da API é uma escolha local.

A demonstração sintética está separada dos CSV e não requer alterações aos ficheiros brutos.
