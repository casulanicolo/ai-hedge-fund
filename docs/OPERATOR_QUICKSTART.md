# Athanor Alpha — Operator Quickstart

> Per chi deve gestire il sistema senza aver scritto il codice.  
> Leggi questo prima del RUNBOOK. Non richiede conoscenze di programmazione.

---

## Cos'è Athanor Alpha

Un sistema automatico che ogni giorno:
1. Analizza un portafoglio di azioni usando 15+ modelli AI
2. Manda una email alle 08:30 ET con raccomandazioni di mercato
3. Apre/chiude posizioni su Alpaca Paper Trading automaticamente  
4. Manda una email alle 17:00 ET con il riepilogo della giornata

È un sistema **paper trading** (denaro simulato). Non gestisce denaro reale.

---

## Come avviarlo (da zero)

```bash
# Sul VPS (Linux)
ssh athanor-vps
cd /home/athanor/athanor-alpha
source .venv/bin/activate

# Avvia tutti i servizi
sudo systemctl start athanor-monitor athanor-dashboard athanor-schedule
sudo systemctl enable athanor-monitor athanor-dashboard athanor-schedule
```

**Verifica**: apri `http://tuo-vps-ip` nel browser → la dashboard deve caricare.

---

## Come fermarlo

```bash
# Ferma tutto (posizioni esistenti restano aperte su Alpaca)
sudo systemctl stop athanor-monitor athanor-schedule athanor-dashboard
```

Per chiudere anche le posizioni Alpaca:
```bash
python -c "
from src.risk.kill_switch import arm
arm('shutdown operatore')
"
# Il daemon chiuderà le posizioni nel prossimo tick, poi si fermerà
```

---

## Dove vedere cosa fa

| Cosa | Dove |
|------|------|
| Dashboard principale | `http://tuo-vps-ip` (o dominio configurato) |
| Email pre-market | Arriva ogni mattino ~08:35 ET |
| Email post-market | Arriva ogni sera ~17:05 ET |
| Log in tempo reale | `journalctl -u athanor-monitor -f` (sul VPS) |
| Log pipeline | `/home/athanor/athanor-alpha/logs/pipeline_YYYY-MM-DD.log` |
| DB SQLite | `/home/athanor/athanor-alpha/db/hedge_fund.db` |

---

## Dashboard — le 6 schede

| Tab | Cosa mostra |
|-----|-------------|
| **Portfolio** | Posizioni aperte, P&L, raccomandazioni correnti |
| **Signals** | Segnali di ogni agente per ogni ticker (oggi) |
| **Agents** | Storico pesi agenti, chi sta performando, chi no |
| **Backtest** | Risultati backtest IS/OOS storici |
| **Execution** | Ordini eseguiti, fill price, status Alpaca |
| **Health** | Circuit breaker, kill switch, audit trail, uptime |

---

## Segnali d'allarme (quando preoccuparsi)

| Segnale | Azione |
|---------|--------|
| Nessuna email pre-market entro le 09:00 ET | Vedi RUNBOOK sezione 1 |
| Dashboard irraggiungibile | Vedi RUNBOOK sezione 2 |
| Tab 6: badge rosso "CB1 TRIGGERED" | Normale se mercato è sceso > 3%. Vedi RUNBOOK sezione 3 |
| Tab 6: badge rosso "ARMED" (kill switch) | Intervento manuale precedente. Verifica e disarma se ok |
| Email postmarket: portfolio value calato > 5% in un giorno | Controlla Tab 5 Execution per errori |
| Nessuna email da 2+ giorni | SSH sul VPS e controlla log + servizi |

---

## Check di 5 minuti (ogni mattina)

```
☐ Email pre-market ricevuta e contenuto ha senso
☐ Dashboard apre senza errori (http://tuo-vps-ip)
☐ Tab 6 → tutti i CB in verde, kill switch DISARMED
☐ Tab 1 → drawdown portfolio < 10%
```

Se tutte le caselle sono verdi: non fare nient'altro.

---

## Contatti e escalation

- **Nicolò Casula** — casulanicolo02@gmail.com — responsabile tecnico
- **Repository**: `github.com/casulanicolo/athanor-alpha`
- **Uptime monitoring**: UptimeRobot (controlla automaticamente ogni 5 min)

---

## Primo avvio checklist (per deploy da zero)

```bash
# 1. Copia il repository
git clone https://github.com/casulanicolo/athanor-alpha.git
cd athanor-alpha

# 2. Crea virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configura le variabili d'ambiente
cp .env.example .env
nano .env   # inserisci tutte le chiavi API

# 4. Inizializza il database
python -m src.db.init_db

# 5. Installa i servizi systemd
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable athanor-monitor athanor-dashboard athanor-schedule
sudo systemctl start athanor-monitor athanor-dashboard athanor-schedule

# 6. Configura nginx
sudo cp deploy/nginx/athanor.conf /etc/nginx/sites-available/athanor
sudo ln -s /etc/nginx/sites-available/athanor /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 7. Installa il crontab
crontab deploy/cron/crontab.txt

# 8. Test pipeline manuale
python -m src.run_pipeline --mode full --no-email

# 9. Verifica health check
python -m src.run_pipeline --health-check
```
