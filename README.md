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
| Download via lightkurve + CLI | — | 🔜 |

## Uso rápido

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

> ⚠️ Por enquanto o modelo é treinado só com dados sintéticos; a acurácia
> (~98%) é otimista. Treino com rótulos reais (tabela KOI) vem a seguir.

## Desenvolvimento

Usamos [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente:

```bash
uv sync                 # cria o ambiente e instala as dependências
uv run pytest           # testes
uv run ruff check .     # lint
uv run ruff format .    # formatação
```
