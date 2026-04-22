# Athanor Alpha — Runbook Operativo

> **A chi serve questo documento**: a te, Nicolò, tra 6 mesi quando avrai dimenticato perché funziona.  
> Ogni sezione segue il formato: **SINTOMO → DIAGNOSI → FIX → VERIFICA**.  
> I comandi sono pronti per il copia-incolla. Sostituisci `athanor-vps` con il tuo hostname SSH.

---

## Check quotidiano (10 minuti, la mattina)

Apri questo checklist in sequenza. Se tutto è verde, hai finito.

```
1. Email pre-market arrivata intorno alle 08:35 ET? → leggila
2. Dashboard Tab 1: drawdown < 10%? sharpe > 0?
3. Dashboard Tab 6: CB tutti verde? Audit trail: zero CRITICAL nelle ultime 24h?
4. Email post-market di ieri sera arrivata? P&L coerente con le posizioni aperte?
5. ssh athanor-vps "systemctl status athanor-monitor athanor-dashboard" → entrambi active
```

Se uno dei 5 punti fallisce → vai alla sezione specifica qui sotto.

---

## Scenari di guasto

### 1. Non è arrivata l'email pre-market

**SINTOMO**: Sono le 09:00 ET, inbox vuota.

**DIAGNOSI**:
```bash
ssh athanor-vps
# Controlla se il cron ha girato
journalctl -u athanor-schedule --since "today" | grep -E "premarket|run_pipeline|ERROR"

# Log diretto
tail -200 /home/athanor/athanor-alpha/logs/pipeline_$(date +%Y-%m-%d).log

# Stato cron
crontab -l | grep premarket
```

**FIX A — pipeline non è girata** (no entry nel log):
```bash
cd /home/athanor/athanor-alpha
source .venv/bin/activate
python -m src.run_pipeline --mode full 2>&1 | tee logs/manual_recovery_$(date +%Y%m%d_%H%M).log
```
Poi manda l'email manualmente:
```bash
python scripts/send_premarket.py   # oppure usa lo script PS1 dal PC locale
```

**FIX B — pipeline girata ma email non partita** (log finisce, niente email):
```bash
# Testa SMTP
python -c "
import smtplib; import os
s = smtplib.SMTP(os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT',587)))
s.starttls(); s.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
print('SMTP OK'); s.quit()
"
# Se fallisce → controlla SMTP_PASSWORD nel .env (Gmail app password scaduta?)
```

**VERIFICA**: `grep "EMAIL SENT\|premarket" logs/pipeline_$(date +%Y-%m-%d).log`

---

### 2. Dashboard non si apre

**SINTOMO**: `http://vps-ip:8501` o dominio → pagina bianca / connection refused.

**DIAGNOSI**:
```bash
ssh athanor-vps
systemctl status athanor-dashboard
journalctl -u athanor-dashboard -n 50 --no-pager
# Porta libera?
ss -tlnp | grep 8501
```

**FIX A — servizio crashato**:
```bash
systemctl restart athanor-dashboard
sleep 5
systemctl status athanor-dashboard
```

**FIX B — errore Python al boot** (vedi il traceback nel journal):
```bash
cd /home/athanor/athanor-alpha
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501   # avvia manuale per vedere errore
```
Risolvi il problema, poi:
```bash
systemctl start athanor-dashboard
```

**FIX C — nginx non ruota**:
```bash
nginx -t   # testa config
systemctl restart nginx
curl -I http://localhost:8501   # verifica Streamlit risponde
```

**VERIFICA**: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501` → 200.

---

### 3. Circuit breaker CB1 triggered

**SINTOMO**: Email di alert CB1 + dashboard Tab 6 mostra CB1 in rosso. Nessun nuovo ordine aperto da questa mattina.

**COSA SIGNIFICA**: La perdita giornaliera del portfolio ha superato il 3%. Sistema ha bloccato nuove aperture automaticamente — è il comportamento corretto.

**DIAGNOSI**:
```bash
ssh athanor-vps
cd /home/athanor/athanor-alpha
source .venv/bin/activate
python -c "
from src.risk.circuit_breakers import check_all
for s in check_all():
    print(s.cb_id, 'TRIGGERED' if s.triggered else 'OK', s.reason)
"
# Verifica il flag file
ls -la .circuit_breaker_cb1_$(date +%Y-%m-%d)
```

**VALUTA**: È corretto che CB1 sia scattato? Controlla Alpaca account.
- Sì, perdita reale → **lascia CB1 attivo** fino a domani. Si resetta automaticamente a mezzanotte.
- No, falso positivo (es. bug last_equity) → reset manuale:

```bash
python -c "from src.risk.circuit_breakers import reset_cb; reset_cb('cb1')"
```

**VERIFICA**: `ls .circuit_breaker_cb1_*` → il file è sparito. Nuovi ordini ripartono.

---

### 4. Ordine in PENDING da > 30 minuti

**SINTOMO**: Alpaca dashboard mostra ordine PENDING, non si riempie.

**DIAGNOSI**:
```bash
# Vedi ultimi ordini nel DB
sqlite3 db/hedge_fund.db "
SELECT ticker, action, status, submitted_at, broker_order_id 
FROM executed_orders 
ORDER BY submitted_at DESC LIMIT 5;
"
# Controlla su Alpaca direttamente
python -c "
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
a = AlpacaBrokerAdapter()
orders = a._client.get_orders()
for o in orders: print(o.id, o.symbol, o.status, o.filled_qty)
"
```

**FIX — cancella l'ordine PENDING**:
```bash
python -c "
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
a = AlpacaBrokerAdapter()
a._client.cancel_orders()   # cancella tutti i pending
print('Tutti i pending cancellati')
"
```
Poi aggiorna il DB:
```bash
sqlite3 db/hedge_fund.db "
UPDATE executed_orders SET status='CANCELED' 
WHERE status='SUBMITTED' AND submitted_at < datetime('now', '-30 minutes');
"
```

**VERIFICA**: `python -c "from src.execution.alpaca_adapter import AlpacaBrokerAdapter; a=AlpacaBrokerAdapter(); print([o.status for o in a._client.get_orders()])"`

---

### 5. Agente X ha weight 0 da 3+ giorni

**SINTOMO**: Dashboard Tab 3 mostra un agente con weight vicino a zero da giorni. Le sue previsioni non influenzano più il portfolio.

**DIAGNOSI**:
```bash
sqlite3 db/hedge_fund.db "
-- Ultime 30 previsioni di quell'agente
SELECT agent_id, ticker, signal, confidence, timestamp
FROM predictions
WHERE agent_id = 'NOME_AGENTE'
ORDER BY timestamp DESC LIMIT 30;

-- Outcomes recenti
SELECT p.agent_id, o.window, AVG(
  CASE WHEN p.signal='BUY' THEN o.actual_return_1d
       WHEN p.signal='SELL' THEN -o.actual_return_1d
       ELSE 0 END
) as directional_accuracy
FROM predictions p
JOIN outcomes o ON p.id = o.prediction_id
WHERE p.agent_id = 'NOME_AGENTE'
GROUP BY o.window;
"
```

**FIX A — agente effettivamente scarso** → lascia che il sistema lo penalizzi. È il feedback loop che funziona.

**FIX B — agente penalizzato per dati anomali** (es. crash API durante backtest):
```bash
sqlite3 db/hedge_fund.db "
UPDATE agent_weights SET weight = 1.0, updated_at = datetime('now')
WHERE agent_id = 'NOME_AGENTE';
"
```

**FIX C — reset completo dei pesi** (caso estremo):
```bash
sqlite3 db/hedge_fund.db "UPDATE agent_weights SET weight = 1.0;"
```

**VERIFICA**: Controlla Tab 3 dopo il prossimo run pipeline.

---

### 6. VPS risponde lento

**SINTOMO**: SSH lento, dashboard lenta, log con ritardi.

**DIAGNOSI**:
```bash
ssh athanor-vps
top -bn1 | head -20
df -h /                          # disco
free -m                          # RAM
iotop -ao --iter 5 2>/dev/null   # I/O (se installato)
ps aux --sort=-%cpu | head -10   # processi CPU-hungry
```

**FIX A — Streamlit mangia RAM**:
```bash
systemctl restart athanor-dashboard
```

**FIX B — pipeline bloccata su LLM call**:
```bash
ps aux | grep "run_pipeline"
kill -TERM <PID>   # graceful
# Se non risponde:
kill -KILL <PID>
```

**FIX C — disco quasi pieno**:
```bash
du -sh db/ cache/ logs/   # dove sta andando lo spazio
# Cleanup cache vecchia
find cache/ -mtime +30 -delete
# Vacuum SQLite
sqlite3 db/hedge_fund.db "VACUUM;"
```

**VERIFICA**: `df -h /` → almeno 2GB liberi; `free -m` → almeno 512MB RAM disponibile.

---

### 7. Disco SQLite > 5GB

**SINTOMO**: `df -h` mostra poco spazio, `du -sh db/` > 5GB.

**DIAGNOSI**:
```bash
sqlite3 db/hedge_fund.db "
SELECT name, SUM(pgsize) as size_bytes
FROM dbstat GROUP BY name ORDER BY size_bytes DESC LIMIT 10;
"
```

**FIX — cleanup + vacuum**:
```bash
cd /home/athanor/athanor-alpha
source .venv/bin/activate

# Rimuovi outcomes più vecchi di 2 anni (mantieni predizioni)
sqlite3 db/hedge_fund.db "
DELETE FROM outcomes 
WHERE evaluated_at < datetime('now', '-2 years');
DELETE FROM audit_trail 
WHERE timestamp < datetime('now', '-6 months');
DELETE FROM monitor_ticks 
WHERE timestamp < datetime('now', '-3 months');
"

# Vacuum per recuperare spazio fisico
sqlite3 db/hedge_fund.db "VACUUM;"
sqlite3 db/hedge_fund.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

**VERIFICA**: `du -sh db/hedge_fund.db` → deve essere calato significativamente.

---

### 8. Alpaca API key revocata / scaduta

**SINTOMO**: Log con `403 Forbidden` o `authentication_error`, ordini non vengono submessi.

**DIAGNOSI**:
```bash
python -c "
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
a = AlpacaBrokerAdapter()
print(a.get_account())
"
```

**FIX**:
1. Vai su https://app.alpaca.markets → API Keys → regenera le chiavi
2. Aggiorna il `.env`:
```bash
nano /home/athanor/athanor-alpha/.env
# Aggiorna ALPACA_API_KEY_ID e ALPACA_API_SECRET_KEY
```
3. Riavvia i servizi:
```bash
systemctl restart athanor-monitor athanor-schedule
```

**VERIFICA**: `python -c "from src.execution.alpaca_adapter import AlpacaBrokerAdapter; print(AlpacaBrokerAdapter().get_account().status)"` → `ACTIVE`.

---

### 9. SEC EDGAR rate-limitato

**SINTOMO**: Log con `429 Too Many Requests` nelle chiamate EDGAR, agenti fundamentals ritornano dati vuoti.

**DIAGNOSI**:
```bash
grep "429\|rate.limit\|EDGAR" logs/pipeline_$(date +%Y-%m-%d).log | tail -20
```

**FIX**:
1. Aspetta 10 minuti. EDGAR ha rate limit di 10 req/sec per IP.
2. Verifica che `SEC_EDGAR_USER_AGENT` nel `.env` sia corretto (EDGAR richiede identificazione).
3. Se persiste, riduci il numero di ticker nel prossimo run:
```bash
python -m src.run_pipeline AAPL MSFT NVDA   # solo 3 ticker
```

**VERIFICA**: `python -c "import requests; r = requests.get('https://data.sec.gov/submissions/CIK0000320193.json', headers={'User-Agent': 'test test@test.com'}); print(r.status_code)"` → 200.

---

### 10. Meta-learner nightly training fallito

**SINTOMO**: Email alert "training failed" o nessun output in `models/meta_learner_*.joblib` da > 2 giorni.

**DIAGNOSI**:
```bash
tail -100 logs/train_nightly.log
ls -la models/meta_learner_*.joblib 2>/dev/null
sqlite3 db/hedge_fund.db "SELECT * FROM ml_model_registry ORDER BY trained_at DESC LIMIT 3;"
```

**FIX A — non abbastanza dati** (< 500 righe nel dataset):
```bash
source .venv/bin/activate
python -c "
from src.ml.dataset_builder import build_dataset
df = build_dataset()
print(f'Rows: {len(df)}')
"
```
Se < 500 righe → normale, il sistema funziona con guardrail (weights = 1.0 di default).

**FIX B — errore xgboost/sklearn**:
```bash
pip install -U xgboost scikit-learn shap
python -m src.ml.train_meta_learner
```

**VERIFICA**: `ls -la models/meta_learner_current.joblib` → file esiste e ha data di oggi.

---

## Procedure manuali

### Chiudere MANUALMENTE tutte le posizioni (Kill Switch)

```bash
# Metodo 1: dashboard (Tab 6 → ARM KILL SWITCH)
# Il daemon chiuderà tutto nel prossimo tick (≤ 60 secondi)

# Metodo 2: CLI immediato
cd /home/athanor/athanor-alpha
source .venv/bin/activate
python -c "
from src.risk.kill_switch import arm, close_all_and_exit
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
arm(reason='operatore: chiusura manuale posizioni')
close_all_and_exit(AlpacaBrokerAdapter())
"
# Questo: arma il kill switch, chiude tutto, esce con sys.exit(0)
```

Dopo la chiusura, riabilita:
```bash
python -c "from src.risk.kill_switch import disarm; disarm()"
systemctl restart athanor-monitor
```

---

### Mettere il sistema in "pause" (no nuove aperture, posizioni esistenti restano)

```bash
# Attiva CB1 manualmente per oggi
touch .circuit_breaker_cb1_$(date +%Y-%m-%d)
# Il portfolio manager bloccherà tutti i nuovi OPEN_LONG/OPEN_SHORT
# Il daemon monitor continua a girare e applica exit rules sulle posizioni esistenti
```

Per riprendere:
```bash
rm -f .circuit_breaker_cb1_$(date +%Y-%m-%d)
```

---

### Roll-back a una versione git precedente

```bash
ssh athanor-vps
cd /home/athanor/athanor-alpha

# 1. Ferma i servizi
systemctl stop athanor-monitor athanor-dashboard athanor-schedule

# 2. Vedi i commit recenti
git log --oneline -20

# 3. Fai il rollback (sostituisci COMMIT_HASH)
git checkout COMMIT_HASH -- src/ scripts/

# 4. Se ci sono migration DB: rollback manuale o usa il backup
cp db/backups/hedge_fund_YYYYMMDD.db db/hedge_fund.db  # se necessario

# 5. Riavvia
systemctl start athanor-monitor athanor-dashboard athanor-schedule
systemctl status athanor-monitor athanor-dashboard
```

---

### Aggiungere un ticker a `config/tickers.yaml`

1. Aggiungi il simbolo al file:
```bash
nano /home/athanor/athanor-alpha/config/tickers.yaml
# Aggiungi il ticker nella lista, es: - TSLA
```

2. Verifica che yfinance lo conosca:
```bash
python -c "import yfinance as yf; t = yf.Ticker('TSLA'); print(t.fast_info)"
```

3. Se il ticker è un'azienda quotata USA, trova il CIK SEC:
```bash
python -c "
import requests
r = requests.get('https://efts.sec.gov/LATEST/search-index?q=tesla&dateRange=custom&startdt=2020-01-01&enddt=2020-12-31&forms=10-K')
"
# Oppure cerca su https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=tesla&type=10-K
```

4. Aggiungi il CIK in `src/tools/sec_edgar_tool.py` o equivalente.

5. Test con un singolo ticker:
```bash
python -m src.run_pipeline TSLA --mode full --no-email
```

---

### Aggiungere un agente nuovo (checklist minima)

```bash
# 1. Crea src/agents/nuovo_agente.py
#    - Funzione: nuovo_agente_node(state: AgentState) -> AgentState
#    - Scrivi in state["data"]["analyst_signals"]["nuovo_agente"] = {...}

# 2. Aggiungi in src/utils/analysts.py:
#    ANALYST_ORDER = [..., "nuovo_agente"]

# 3. Aggiungi in src/graph/graph.py:
#    graph.add_node("nuovo_agente", nuovo_agente_node)
#    graph.add_edge("data_prefetch", "nuovo_agente")  # o parallel fan-out

# 4. Aggiungi in src/agents/portfolio_manager.py → AGENT_DIMENSION_MAP:
#    "nuovo_agente": "FUNDAMENTALS"  # o la dimensione appropriata

# 5. Aggiungi in src/ml/feature_extractor.py → SECTOR_MAP se necessario

# 6. Testa:
python -m src.run_pipeline AAPL --mode full --no-email

# 7. Verifica che le predizioni siano salvate:
sqlite3 db/hedge_fund.db "SELECT COUNT(*) FROM predictions WHERE agent_id='nuovo_agente';"
```

---

## Quick Reference

| Task | Comando |
|------|---------|
| Status servizi | `systemctl status athanor-monitor athanor-dashboard athanor-schedule` |
| Log monitor live | `journalctl -u athanor-monitor -f` |
| Log pipeline | `tail -f logs/pipeline_$(date +%Y-%m-%d).log` |
| Run pipeline manuale | `python -m src.run_pipeline` |
| CB status | `python -m src.risk._cb_runner` |
| Health check | `python -m src.run_pipeline --health-check` |
| Backup DB manuale | `bash scripts/backup_db.sh` (VPS) o `.\scripts\backup_db.ps1` (locale) |
| Kill switch | `touch .athanor_kill` (arm) / `rm .athanor_kill` (disarm) |
| Reset pesi agenti | `sqlite3 db/hedge_fund.db "UPDATE agent_weights SET weight=1.0;"` |
| Vacuum DB | `sqlite3 db/hedge_fund.db "VACUUM;"` |
