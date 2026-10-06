# exoplanet-detector

Detecção de exoplanetas pelo **método de trânsito**: quando um planeta passa na
frente da sua estrela, o brilho dela cai um pouco, de forma periódica. Este
projeto procura essas quedas em curvas de luz das missões Kepler e TESS.

## Pipeline

```
curva de luz → pré-processamento → busca BLS → features → classificador → candidato / falso positivo
```

| Etapa | Módulo | Status |
|---|---|---|
| Estrutura da curva de luz | `lightcurve.py` | ✅ |
| Limpeza, normalização e detrending | `preprocess.py` | ✅ |
| Busca de trânsitos (Box Least Squares) | `search.py` | ✅ |
| Curvas sintéticas (planetas e binárias eclipsantes) | `synthetic.py` | ✅ |
| Features de vetting (odd/even, secundário, formato) | `features.py` | ✅ |
| Classificador (Random Forest) | `classify.py` | ✅ |
| Dados reais: tabela KOI + curvas do Kepler | `data.py` | ✅ |
| Linha de comando | `cli.py` | ✅ |

## Linha de comando

Analisa uma estrela do Kepler pelo ID KIC (baixa ~1 ano de dados do MAST):

```console
$ uv run exoplanet-detector 11904151   # Kepler-10
10794 pontos, 242 dias de observação
  período       0.83750 d
  duração       1.92 h
  profundidade  150 ppm (SNR 121.6)
  odd/even      0.5 sigma
  secundário    -1.6 sigma
  formato       0.82 (1 = U, 0.67 = V)

P(planeta) = 0.27 -> FALSO POSITIVO (binária eclipsante?)
```

O BLS encontra a Kepler-10b com precisão (período publicado: 0.837495 d),
mas o classificador erra: veja [Limitações](#limitações).

## Uso como biblioteca

```python
from exoplanet_detector import preprocess, search_transit
from exoplanet_detector.synthetic import make_lightcurve

lc = make_lightcurve(period=3.5, depth=0.005)  # estrela simulada com um planeta
candidate = search_transit(preprocess(lc))
print(candidate)  # period≈3.5, depth≈0.005, ...
```

### Planeta ou falso positivo?

O falso positivo mais comum é uma **binária eclipsante**: duas estrelas se
eclipsando. As features em `features.py` capturam as assinaturas dela:

| Feature | Planeta | Binária eclipsante |
|---|---|---|
| `odd_even_sigma` | trânsitos iguais | eclipses alternados diferentes |
| `secondary_sigma` | ~0 | queda em fase 0.5 |
| `shape_ratio` | ~1 (fundo chato, "U") | ~0.67 (formato "V") |

```python
from exoplanet_detector.classify import TransitClassifier, analyze, synthetic_training_set

features, labels = synthetic_training_set(n_per_class=100)
clf = TransitClassifier().fit(features, labels)
clf.predict_proba([analyze(lc)])  # probabilidade de ser planeta
```

## Resultados em estrelas reais do Kepler

Dataset: 283 estrelas com um único KOI e período entre 0.5 e 30 dias
(149 planetas confirmados, 134 binárias eclipsantes), 4 quarters de dados
cada. Gerado com `scripts/build_dataset.py` e distribuído em
`src/exoplanet_detector/resources/kepler_features.csv`.

O BLS encontra o período do catálogo (ou metade/dobro) em **82%** das estrelas.

| Modelo | Acurácia | ROC AUC |
|---|---|---|
| Treinado em sintéticos, testado no Kepler | 0.69 | 0.83 |
| Treinado no Kepler (validação cruzada, 5 folds) | **0.82** | **0.89** |

O modelo sintético tinha ~98% de acurácia em dados sintéticos: a diferença
para 69% mostra o quanto a simulação simplifica a realidade. Reproduza com
`uv run python scripts/evaluate.py`.

### Limitações

- **Planetas de período ultracurto** (ex.: Kepler-10b, 20 h) têm duração/período
  alta, parecida com binárias próximas, e são raros no dataset.
- **Cadência de 30 min** borra trânsitos curtos (< ~3 h), que ficam com cara de
  "V" no `shape_ratio`.
- Dataset pequeno (283 estrelas) e só os 4 primeiros quarters do Kepler.

## Desenvolvimento

Usamos [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente:

```bash
uv sync                 # cria o ambiente e instala as dependências
uv run pytest           # testes
uv run ruff check .     # lint
uv run ruff format .    # formatação
```
