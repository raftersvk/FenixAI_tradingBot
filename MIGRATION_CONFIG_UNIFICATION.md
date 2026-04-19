# Migration vers la Configuration Unifiée — Résumé

**Date** : 2025-04-19  
**Status** : ✅ Terminé et testé  
**Commit** : À créer

---

## 🎯 Objectif Atteints

1. ✅ Unification sur `src/config/unified_loader.py` + `config/fenix.yaml`
2. ✅ `min_klines_to_start` provenant de `fenix.yaml` (500) — **PROBLÈME RÉSOLU**
3. ✅ Nettoyage du `.env` : suppression de 25+ variables dépréciées
4. ✅ Suppression des fallbacks env vars pour le Judge
5. ✅ Tous les tests passent (9/9)
6. ✅ Application démarre en CLI et API

---

## 📊 Changements par Fichier

### `config/fenix.yaml` (+6 lignes)
- Ajout section `api` : `log_level`, `cors_origins`, `expose_api`, `create_demo_users`
- `min_klines_to_start: 500` (déjà présent, confirmé)

### `src/trading/engine.py` (143 lignes modifiées)
- **Avant** : `self._min_klines_to_start = int(os.getenv("FENIX_MIN_KLINES_TO_START", "20"))`
- **Après** : `self._min_klines_to_start = config.trading.min_klines_to_start`
- Ajout fallback vers `_init_with_defaults()` si `unified_loader` échoue
- Paramètres d'`__init__` rendus optionnels (`None`) pour permettre fallback config
- Tous les paramètres viennent de `get_config()` sauf overrides explicites

### `src/api/server.py` (91 lignes modifiées)
- Remplacé `APP_CONFIG` par `get_config()` pour `log_level` et `create_demo_users`
- Supprimé import de `config_loader.APP_CONFIG`
- Ajout fallback graceful si `unified_loader` indisponible

### `src/config/unified_loader.py` (54 lignes modifiées)
- Ajout dataclass `APIConfig`
- Ajout champ `api: APIConfig` dans `FenixConfig`
- Nettoyage formatage (espaces, sauts de ligne)

### `src/config/judge_config.py` (38 lignes modifiées)
- **Supprimé** : tous les fallbacks `FENIX_JUDGE_*` env vars
- **Maintenant** : lecture uniquement depuis `llm_providers.yaml` via `llm_provider_loader`
- Fallback vers defaults codés en dur (`ollama_local`, `qwen3:8b`, etc.)

### `src/config/config_loader.py` (57 lignes modifiées)
- **Supprimé** : `if os.getenv("TRADING_MODE") == "testnet": return True`
- Désormais : `use_testnet` vient uniquement de `fenix.yaml → binance.testnet`

### `run_fenix.py` (12 lignes modifiées)
- Arguments passés en `None` si equals to defaults, pour permettre à l'engine de prendre depuis YAML
- Exemple : `symbol=args.symbol if args.symbol != "BTCUSDT" else None`

### `tests/test_trading_engine.py` (92 lignes modifiées)
- Ajout fixture `mock_unified_config` pour mocker `get_config()`
- Tous les tests patchent `get_config()` pour contrôler `min_klines_to_start=20`
- Mise à jour assertions pour utiliser `mock_config.trading.min_klines_to_start`

### `.env` (nettoyé)
- Supprimé toutes les variables dépréciées (25+)
- Gardé uniquement : secrets, flags, overrides
- `LOG_LEVEL` commenté (maintenant dans `fenix.yaml`)

### `.env.example` (nouveau, 223 lignes)
- Documentation complète de toutes les variables autorisées
- Groupées par catégorie
- Section "DÉPRÉCIÉS" pour référence

---

## 🔧 Variables Dépréciées Supprimées

| Variable | Remplacée par |
|----------|---------------|
| `TRADING_MODE` | `fenix.yaml: binance.testnet` |
| `DEFAULT_SYMBOL` | `fenix.yaml: trading.symbol` |
| `PRIMARY_TIMEFRAME` | `fenix.yaml: trading.timeframe` |
| `ENABLE_SENTIMENT_AGENT` | `fenix.yaml: agents.enabled.sentiment` |
| `ENABLE_TECHNICAL_AGENT` | `fenix.yaml: agents.enabled.technical` |
| `ENABLE_VISUAL_AGENT` | `fenix.yaml: agents.enabled.visual` |
| `ENABLE_QABBA_AGENT` | `fenix.yaml: agents.enabled.qabba` |
| `OLLAMA_MODEL` | `fenix.yaml: llm.default_model` |
| `FENIX_MIN_KLINES_TO_START` | `fenix.yaml: trading.min_klines_to_start` |
| `FENIX_JUDGE_*` (5 vars) | `llm_providers.yaml: judge.*` |
| `HF_TOKEN` | `HUGGINGFACE_API_KEY` (alias) |
| + 15 autres... | non utilisés |

---

## ✅ Tests

```bash
$ pytest tests/test_trading_engine.py -v
======================== 9 passed, 5 warnings in 7.21s =========================
```

```bash
$ python run_fenix.py --dry-run
...
TradingEngine initialized: BTCUSDC@30m (paper=True, testnet=True)
min_klines_to_start: 500 ✅
```

---

## 📋 Fichiers Obsolètes (à supprimer après validation)

Les fichiers suivants sont **toujours présents** mais peuvent être supprimés une fois la migration validée en production :

| Fichier | Raison | Utilisation actuelle |
|---------|--------|---------------------|
| `src/config/config_loader.py` | Ancien système (APP_CONFIG) | Fallback dans `engine.py` si unified fails |
| `config/config.yaml` | Doublon de `fenix.yaml` | Aucune (fallback `config_loader`) |
| `config/settings.py` | Module settings legacy | Importé par `system/__init__.py` pour legacy |
| `src/config/system_config.yaml` | Obsolète | Aucune |

**Recommandation** : Conserver pour l'instant (fallbacks). Supprimer après 1 semaine de validation en prod.

---

## 🔄 Ordre de Priorité des Configs (post-migration)

1. **Variables d'env** (secrets, overrides d'urgence) —Highest priority
2. **`config/fenix.yaml`** (configuration principale) — High priority
3. **`config/llm_providers.yaml`** (providers LLM) — High priority
4. **Defaults du code** (`unified_loader` dataclasses) — Lowest priority

---

## 🚀 Prochaines Étapes

1. **Commit** cette migration
2. **Déployer** en staging/développement pour validation
3. **Vérifier** les logs : `min_klines_to_start` doit être 500
4. **Nettoyer** les fichiers obsolètes après 1 semaine si aucun problème
5. **Mettre à jour** README.md avec nouvelle structure de config

---

## ⚠️ Points d'Attention

- `engine.py` garde un fallback vers `os.getenv("FENIX_MIN_KLINES_TO_START")` si `unified_loader` fail — c'est volontaire
- `server.py` utilise `config.logging.level` mais fallback sur `LOG_LEVEL` env var — permet transition douce
- `settings.py` et `config_loader.py` sont conservés pour compatibilité ascendante
- Le système legacy (`system/__init__.py`) utilise encore `settings.py` — à migrer plus tard

---

## 📈 Impact

- ✅ **Code plus simple** : un seul système de config à comprendre
- ✅ **min_klines_to_start** maintenant correctement pris en compte (500 au lieu de 20)
- ✅ **Docker** : `.env` épuré, seulement l'essentiel
- ✅ **Maintenabilité** : configuration centralisée, documentée
- ✅ **Tests** : mocks appropriés, tous passent

---

**Migration terminée avec succès.**
