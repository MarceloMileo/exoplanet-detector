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
  ingresso      0.39 (0 = fundo chato, 1 = V)

P(planeta) = 0.51 -> PLANETA
```

O BLS encontra a Kepler-10b com precisão (período publicado: 0.837495 d). O
classificador acerta, mas por pouco: veja [Limitações](#limitações).

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
| `ingress_ratio` | ~0 (fundo chato) | ~1 (formato "V") |

```python
from exoplanet_detector.classify import TransitClassifier, analyze, synthetic_training_set

features, labels = synthetic_training_set(n_per_class=100)
clf = TransitClassifier().fit(features, labels)
clf.predict_proba([analyze(lc)])  # probabilidade de ser planeta
```

## Resultados em estrelas reais do Kepler

Dataset: 284 estrelas com um único KOI e período entre 0.5 e 30 dias
(149 planetas confirmados, 135 binárias eclipsantes), 4 quarters de dados
cada. Gerado com `scripts/build_dataset.py` e distribuído em
`src/exoplanet_detector/resources/kepler_features.csv`.

O BLS encontra o período do catálogo (ou metade/dobro) em **82%** das estrelas.

| Modelo | Acurácia | ROC AUC |
|---|---|---|
| Treinado em sintéticos, testado no Kepler | 0.80 | 0.89 |
| Treinado no Kepler (validação cruzada, 5 folds) | **0.87** | **0.92** |

Reproduza com `uv run python scripts/evaluate.py`.

### Histórico

| Versão | Mudança | Acurácia (Kepler, CV) | ROC AUC | P(planeta) Kepler-10b |
|---|---|---|---|---|
| PR 3 | `shape_ratio`: profundidade total / miolo, com a duração do BLS | 0.82 | 0.89 | 0.27 ❌ |
| PR 4 | `ingress_ratio`: trapézio ajustado e integrado na exposição | **0.87** | **0.92** | 0.51 ✅ |

A feature de formato passou de 14% para 32% da importância do modelo, e as
medianas se separaram bem: `ingress_ratio` 0.26 para planetas e 0.73 para
binárias. O modelo treinado só em sintéticos também melhorou (0.69 → 0.80),
em parte porque os trânsitos simulados passaram a ser integrados na exposição,
como num detector real.

### Limitações

- **Planetas de período ultracurto** (ex.: Kepler-10b, 20 h) têm duração/período
  alta, parecida com binárias próximas, e são raros no dataset.
- **Escurecimento de limbo**: a borda da estrela é mais escura, então o fundo do
  trânsito de um planeta real é curvo, e o trapézio o vê como parcialmente "V"
  (Kepler-10b: `ingress_ratio` 0.39). Um modelo de trânsito físico (ex.:
  Mandel & Agol) separaria melhor.
- Dataset pequeno (283 estrelas) e só os 4 primeiros quarters do Kepler.

## Desenvolvimento

Usamos [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente:

```bash
uv sync                 # cria o ambiente e instala as dependências
uv run pytest           # testes
uv run ruff check .     # lint
uv run ruff format .    # formatação
```
