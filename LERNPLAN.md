# Lernplan: Mein erstes LLM von Grund auf

Dieses Dokument ist der Lern- und Theorieplan fuer das Projekt `my-first-llm`.
Es beantwortet: Wie startet man gut, welche Datenquellen gibt es, welche
Architektur nutzt man, was ist beim Training zu beachten, und wie interagiert
man am Ende mit dem Modell bzw. hostet es.

Der zugehoerige Umsetzungsplan fuer einen Coding-Agenten liegt in
[AGENT_PLAN.md](AGENT_PLAN.md).

---

## 1. Zielbild und Rahmenbedingungen

**Ziel:** Zwei Etappen, aufeinander aufbauend.

1. **Etappe A — From Scratch:** Ein kleines GPT (ca. 10 bis 25 Mio. Parameter)
   komplett selbst in PyTorch schreiben und auf dem Datensatz TinyStories
   vortrainieren. Ergebnis: ein Modell, das fluessige englische
   Kindergeschichten erzeugt, plus tiefes Verstaendnis jeder Komponente.
2. **Etappe B — Fine-Tuning:** Ein fertiges Open-Source-Modell (1.7B bis 4B
   Parameter) mit QLoRA auf dem Mac feintunen und anschliessend lokal hosten
   (Ollama). Ergebnis: ein praktisch nutzbares, personalisiertes Modell und
   Verstaendnis des Praxis-Workflows.

**Hardware:** MacBook Pro M4, 16 GB Unified Memory. Training laeuft ueber das
PyTorch-MPS-Backend (Metal) bzw. in Etappe B ueber Apples MLX-Framework.
Fallback fuer groessere Laeufe: Google Colab (kostenlose T4-GPU).

**Realistische Erwartung:** Ein 25M-Parameter-Modell wird kein ChatGPT. Es
erzaehlt einfache, grammatisch korrekte Kindergeschichten — und genau das ist
das Erfolgskriterium der TinyStories-Forschung (siehe Abschnitt 4.2). Der
Wert von Etappe A ist das Verstaendnis, nicht das Produkt. Das "nutzbare"
Modell kommt aus Etappe B.

---

## 2. Der empfohlene Lernpfad (Warum dieser Weg?)

Die Community-Meinung ist hier erstaunlich einheitlich: Der Standard-Lernpfad
fuer "LLM from scratch" ist der von **Andrej Karpathy** (Ex-OpenAI, Ex-Tesla).
Seine Materialien bauen exakt aufeinander auf:

1. **Neural Networks: Zero to Hero** (Videoserie): von Backpropagation an
   einzelnen Skalaren (micrograd) ueber n-Gramm-Modelle (makemore) bis zum
   Video "Let's build GPT: from scratch, in code, spelled out" — dort entsteht
   ein Mini-GPT auf Shakespeare-Text, Zeile fuer Zeile.
2. **"Let's build the GPT Tokenizer"** (Video) plus das Repo **minbpe**:
   Byte-Pair-Encoding-Tokenizer selbst implementieren.
3. **microgpt** (Feb 2026): ca. 200 Zeilen pures Python ohne Abhaengigkeiten —
   Tokenizer, Autograd, GPT, Adam, Training und Inferenz in einer Datei.
   Laeuft in etwa einer Minute auf einem MacBook. Karpathys eigene Aussage:
   "This file contains the full algorithmic content of what is needed.
   Everything else is just efficiency."
4. **nanoGPT**: das Referenz-Repo fuer ein trainierbares GPT-2-artiges Modell
   in wenigen hundert Zeilen PyTorch. Unsere eigene Implementierung orientiert
   sich hieran (aber wir schreiben selbst, statt zu kopieren).
5. **nanochat** (Okt 2025): die volle Pipeline Tokenizer -> Pretraining ->
   SFT -> RL -> Inferenz -> Web-UI in ca. 8000 Zeilen. Dient uns als Referenz
   fuer die spaeteren Phasen (Chat-Format, SFT, Inference-Engine). Laeuft laut
   README auch auf CPU/Apple-MPS mit stark verkleinerten Modellen.

**Begleitbuch:** Sebastian Raschka, *Build a Large Language Model (From
Scratch)* (Manning, 2024). Baut ein GPT nur mit PyTorch-Grundbausteinen auf
und hat eigene Kapitel zu Fine-Tuning. Ideal als strukturierte Referenz neben
den Videos; das komplette Code-Repo ist frei verfuegbar.

**Praxis-Referenz:** Hugging Faces *Smol Training Playbook* (Okt 2025)
dokumentiert schonungslos, wie das Training von SmolLM3 (3B Parameter) in
echt ablief — inklusive Fehlstarts und Loss-Spikes. Wichtigste Lektion fuer
uns: Probleme beim Training sind normal; entscheidend ist, sie schnell zu
erkennen (Logging, Checkpoints, kleine Testlaeufe zuerst).

---

## 3. Theorie-Grundlagen in Lesereihenfolge

Diese Konzepte in dieser Reihenfolge lernen. Zu jedem Punkt steht, wo man es
am besten lernt (Quellen in Abschnitt 8).

### 3.1 Sprachmodellierung = Next-Token-Prediction

Ein GPT tut genau eine Sache: Es bekommt eine Folge von Tokens und sagt eine
Wahrscheinlichkeitsverteilung ueber das naechste Token voraus. Training heisst:
diese Verteilung soll dem tatsaechlichen naechsten Token im Trainingstext
moeglichst hohe Wahrscheinlichkeit geben. Alles Weitere (Chat, Reasoning,
Code) ist emergentes Verhalten dieser einen Aufgabe plus spaeterem Feintuning.
*Quelle: Karpathy "Let's build GPT", Raschka Kap. 1.*

### 3.2 Tokenisierung und Byte-Pair-Encoding (BPE)

Text wird nicht als Buchstaben, sondern als "Tokens" (haeufige
Zeichenketten-Stuecke) verarbeitet. BPE startet mit Bytes und verschmilzt
iterativ das haeufigste Paar zu einem neuen Token, bis die gewuenschte
Vokabulargroesse erreicht ist. Wichtige Intuition: Die Vokabulargroesse ist
ein Trade-off — grosses Vokabular = kuerzere Sequenzen, aber groessere
Embedding-Matrix (bei kleinen Modellen dominiert die schnell die
Parameterzahl). Fuer TinyStories reichen 4096 bis 8192 Tokens.
*Quelle: Karpathy Tokenizer-Video + minbpe.*

### 3.3 Embeddings

Jedes Token wird auf einen lernbaren Vektor (Dimension `d_model`) abgebildet.
Positionsinformation kommt dazu — klassisch (GPT-2) als gelernte
Positions-Embeddings, modern als RoPE (siehe 3.9).

### 3.4 Self-Attention, Schritt fuer Schritt

Das Herzstueck. Fuer jedes Token werden drei Vektoren berechnet: Query, Key,
Value. Die Attention-Gewichte sind `softmax(Q @ K^T / sqrt(d_k))`, damit
gewichtet man die Values. Kausale Maskierung sorgt dafuer, dass ein Token nur
auf fruehere Tokens "schauen" kann (sonst waere Next-Token-Prediction
trivial). Multi-Head: dasselbe parallel in mehreren Unterraeumen. Das
Karpathy-Video baut das in ca. 40 Minuten von einem einfachen gewichteten
Mittelwert bis zur vollen Formel auf — der wichtigste Lernabschnitt des
ganzen Projekts.
*Quellen: Karpathy "Let's build GPT"; Original-Paper "Attention Is All You
Need" (Vaswani et al. 2017) danach lesen, nicht davor.*

### 3.5 Der Transformer-Block

Ein Block = Attention + Feed-Forward-Netz (MLP), beide mit Residual-Verbindung
und Normalisierung ("Pre-Norm": Normalisierung vor der Teilschicht, das ist
heute Standard und stabiler als Post-Norm). Ein GPT ist einfach N solcher
Bloecke gestapelt, plus Embedding am Anfang und ein Linear-Layer ("LM-Head")
am Ende, der auf Vokabulargroesse projiziert.

### 3.6 Loss: Cross-Entropy und Perplexity

Trainiert wird mit Cross-Entropy zwischen vorhergesagter Verteilung und dem
tatsaechlichen naechsten Token. Perplexity = `exp(loss)` — anschaulich: "unter
wie vielen Tokens raet das Modell effektiv". Wichtige Referenzpunkte: Bei
zufaelligem Raten mit Vokabular V ist der Loss `ln(V)` (bei 8192 also ~9.0).
Char-Level-Shakespeare landet bei ~1.5, ein gutes TinyStories-Modell
(BPE) im Bereich ~1.2 bis 1.8 Val-Loss.

### 3.7 Optimierung: AdamW, Schedules, Stabilitaet

- **AdamW** ist der Standard-Optimizer (Adam mit entkoppeltem Weight Decay).
- **Learning-Rate-Schedule:** linearer Warmup (einige hundert Schritte), dann
  Cosine-Decay auf ~10 % der Peak-LR. Ohne Warmup divergieren Transformer
  gern am Anfang.
- **Gradient Clipping** (Norm auf 1.0) verhindert Loss-Spikes.
- **Gradient Accumulation:** mehrere Mini-Batches aufsummieren, bevor ein
  Optimizer-Schritt erfolgt — simuliert grosse Batches auf kleiner Hardware.
- **Mixed Precision:** auf CUDA `bfloat16`; auf Apple-MPS ist `float32` (ggf.
  `float16`) die sichere Wahl, `bfloat16` und `torch.compile` sind dort nicht
  zuverlaessig (Stand der Recherche, s. Abschnitt 6).
*Quellen: nanoGPT `train.py`, Raschka Kap. 5, Smol Training Playbook.*

### 3.8 Overfitting, Splits und Evaluation

Immer einen Validation-Split abtrennen und regelmaessig Val-Loss messen.
Bei kleinen Datenmengen (Shakespeare) overfittet das Modell schnell: Train-
Loss faellt weiter, Val-Loss steigt. Ebenso wichtig wie der Loss:
**regelmaessig Beispieltexte generieren** und lesen — Zahlen luegen selten,
aber Texte zeigen sofort, ob das Modell Unsinn produziert. Das
TinyStories-Paper bewertet Grammatik/Kreativitaet/Konsistenz sogar per
GPT-4-Bewertung; fuer uns reicht Val-Loss plus manuelle Stichproben.

### 3.9 Scaling Laws (Chinchilla)

Das Chinchilla-Paper (Hoffmann et al. 2022) zeigt: compute-optimal ist etwa
**20 Trainings-Tokens pro Parameter** (Replikationen bestaetigen 15 bis 25).
Fuer uns heisst das: TinyStories hat ~470 Mio. Tokens, also ist ein Modell
mit ~10 bis 25 Mio. Parametern der Sweet Spot — groesser lohnt ohne mehr
Daten/Compute nicht. (Produktionsmodelle wie Llama trainieren bewusst weit
ueber Chinchilla hinaus, weil Inferenzkosten dann sinken — gut zu wissen,
fuer uns aber nicht relevant.)

### 3.10 Moderne Architektur-Bausteine (GPT-2 vs. Llama-Stack)

GPT-2 (2019) und Llama/Mistral/Qwen (2023 bis 2025) sind strukturell fast
gleich; die Unterschiede sind wenige, klar benennbare Upgrades:

| Baustein | GPT-2 (Start) | Modern (Ablation in Phase 3) | Effekt |
|---|---|---|---|
| Position | gelernte absolute Embeddings | **RoPE** (Rotationen auf Q/K) | bessere Laengen-Generalisierung |
| Normalisierung | LayerNorm | **RMSNorm** | einfacher, stabiler bei Skalierung |
| MLP-Aktivierung | GELU | **SwiGLU** (gated MLP) | mehr Ausdruckskraft |
| Bias-Terme | ueberall | **keine** | weniger Parameter, kaum Verlust |
| Embeddings | LM-Head separat | **weight tying** (geteilt) | spart Parameter, hilft kleinen Modellen |
| Attention | volle Heads | GQA/MQA (erst bei grossen Modellen relevant) | spart KV-Cache-Speicher |

Didaktischer Plan: **erst GPT-2-Stil bauen** (deckungsgleich mit
Karpathy-Material), dann jeden Baustein einzeln austauschen und den Effekt
auf den Val-Loss messen. So versteht man, warum die moderne Architektur so
aussieht, wie sie aussieht.
*Quelle: Raschka, "The Big LLM Architecture Comparison".*

### 3.11 Inferenz: Sampling und KV-Cache

- **Greedy** (immer das wahrscheinlichste Token) wird schnell repetitiv.
- **Temperature** skaliert die Logits vor dem Softmax: <1 konservativer,
  >1 kreativer/chaotischer.
- **Top-k** beschraenkt auf die k wahrscheinlichsten Tokens; **Top-p
  (Nucleus)** auf die kleinste Menge mit kumulierter Wahrscheinlichkeit p.
- **KV-Cache:** Bei der Generierung muss man Keys/Values vergangener Tokens
  nicht neu berechnen, sondern cached sie — macht Generierung von O(n^2) pro
  Token zu O(n). Selbst zu implementieren ist eines der besten Lernziele in
  Phase 4 (nanochat hat eine Referenz-Engine).

### 3.12 Pretraining vs. SFT vs. RLHF

- **Pretraining:** Next-Token auf Rohtext -> "Base Model", vervollstaendigt
  Text, folgt aber keinen Anweisungen.
- **Supervised Fine-Tuning (SFT):** Weitertrainieren auf
  Frage/Antwort-Beispielen in einem Chat-Template -> Modell lernt die Rolle
  "Assistent". Wichtig: Loss wird nur auf den Antwort-Tokens berechnet
  (Prompt-Masking).
- **RLHF/DPO/RL:** Feinschliff nach menschlichen Praeferenzen — fuer dieses
  Projekt nur Theorie (nanochat enthaelt eine einfache RL-Stufe zum
  Nachlesen).

### 3.13 LoRA und QLoRA (fuer Etappe B)

**LoRA:** Statt alle Gewichte zu aendern, lernt man kleine Low-Rank-Matrizen
`A @ B` (Rang r, z. B. 8 bis 16), die additiv auf ausgewaehlte
Gewichtsmatrizen wirken. Trainierbare Parameter: <1 % des Modells.
**QLoRA:** Das Basismodell liegt dabei 4-bit-quantisiert im Speicher, nur die
LoRA-Adapter sind in hoeherer Praezision. Faustregeln fuer 16 GB Unified
Memory: LoRA braucht ~2 GB pro 1B Parameter, QLoRA ~0.5 GB pro 1B —
QLoRA-Finetuning eines 7B/8B-Modells ist damit machbar (Peak ~5 bis 8 GB),
komfortabel und schnell sind 1.7B bis 4B (z. B. Qwen-Familie).
*Quellen: LoRA-Paper (Hu et al. 2021), QLoRA-Paper (Dettmers et al. 2023),
mlx-lm-Doku.*

### 3.14 Quantisierung und GGUF

Quantisierung reduziert Gewichte von 16/32 bit auf z. B. 4 bit — Modelle
werden 3 bis 4 mal kleiner bei geringem Qualitaetsverlust (Q4_K_M ist der
uebliche Kompromiss). **GGUF** ist das Dateiformat von llama.cpp; Ollama und
LM Studio nutzen es. Wichtig: Der Konverter (`convert_hf_to_gguf.py`)
unterstuetzt nur bekannte Architekturen (Llama, Qwen, ...) — unser
From-Scratch-Modell aus Etappe A laesst sich daher nicht sinnvoll nach GGUF
bringen; das hosten wir mit eigenem Code (Abschnitt 5).

### 3.15 Apple Silicon: Unified Memory, MPS, MLX

- CPU und GPU teilen sich die 16 GB ("Unified Memory") — kein
  Kopier-Overhead, aber auch kein Auslagern: Modell + Optimizer + Batches
  muessen komplett reinpassen. Realistisch nutzbar: ~10 bis 12 GB.
- **PyTorch MPS** (`device="mps"`): funktioniert gut fuer nanoGPT-grosse
  Modelle. Referenzpunkt: Shakespeare-nanoGPT trainierte auf einem M2 in
  ~3.5 Minuten auf brauchbaren Loss.
- **MLX** ist Apples eigenes Array-Framework, fuer Apple Silicon optimiert;
  `mlx-lm` bringt fertige Befehle fuer LoRA/QLoRA-Finetuning und Inferenz.
  Fuer Etappe B die beste Wahl auf dem Mac.

---

## 4. Datenquellen (recherchiert und abgewogen)

### 4.1 Uebersicht

| Datensatz | Groesse | Zweck | Bewertung fuer uns |
|---|---|---|---|
| tiny-shakespeare | ~1 MB | Smoke-Test, char-level | Perfekt fuer Phase 1: Trainingsloop in Minuten validieren |
| **TinyStories** | ~2 GB, ~470M Tokens | Haupt-Pretraining | **Erste Wahl** (Begruendung unten) |
| FineWeb / FineWeb-Edu | 15T / 1.3T Tokens (Subsets: `sample-10BT`) | "echtes" Web-Pretraining | Optionale Steigerung auf Colab; fuer 16-GB-Mac zu gross |
| OpenWebText | ~38 GB | GPT-2-Reproduktion | Von FineWeb ueberholt, nur historisch interessant |
| The Pile / C4 / SlimPajama | 100er GB | klassische Korpora | dito, von FineWeb ueberholt |
| Alpaca (52k) | klein | SFT / Instruction-Tuning | Standard fuer Hobby-Finetunes, Etappe B |
| SmolTalk (1.1M) | mittel | SFT | hochwertiger Mix, falls Alpaca zu simpel |
| OASST1 | 66k Konversationen | SFT | menschlich erzeugt, gute Qualitaet |

### 4.2 Warum TinyStories die richtige Wahl ist

Das TinyStories-Paper (Eldan & Li, Microsoft Research 2023) hat gezeigt:
Wenn man den Trainingstext auf den Wortschatz eines Kindes beschraenkt
(synthetisch mit GPT-3.5/4 erzeugte Kurzgeschichten), erzeugen schon Modelle
**unter 10 Mio. Parametern** fluessiges, grammatisch fast perfektes Englisch
mit erkennbarem Reasoning — Training in unter einem Tag auf einer einzelnen
GPU. Genau die Nische, in der ein M4 mit 16 GB ein befriedigendes Ergebnis
liefern kann. Mit ~470M Tokens passt der Datensatz ausserdem Chinchilla-
optimal zu einem 10-25M-Modell (Abschnitt 3.9). Zusatzbefund der Forschung:
Daten-Diversitaet schlaegt Datenmenge — Duplikate im Trainingsset vermeiden.

### 4.3 Praktische Hinweise

- TinyStories liegt auf Hugging Face (`roneneldan/TinyStories`); die
  Variante **TinyStoriesV2** (nur GPT-4-generiert) ist qualitativ besser.
- Daten einmal tokenisieren und als binaere Token-Arrays (uint16, memmap)
  ablegen — Standard-nanoGPT-Pattern, spart RAM und Zeit.
- Datendateien und Checkpoints gehoeren nicht in Git (falls das Projekt
  spaeter doch versioniert wird).

---

## 5. Interaktion und Hosting

### 5.1 Eigenes From-Scratch-Modell (Etappe A)

GGUF/llama.cpp scheidet aus (Custom-Architektur, Abschnitt 3.14). Stattdessen
— und das ist als Lernziel sogar besser:

1. **Eigene Sampling-Loop** (Temperature, Top-k, Top-p) + **eigener KV-Cache**.
2. **CLI-Tool:** Prompt rein, Geschichte raus.
3. **Gradio `ChatInterface`:** eine Chat-Web-UI in ~20 Zeilen Python, laeuft
   lokal im Browser. Gradio basiert intern auf FastAPI; wer moechte, haengt
   einen eigenen **FastAPI-Endpoint** (`POST /generate`) daneben — das ist
   dasselbe Muster, mit dem echte Inference-Server arbeiten.

### 5.2 Feingetuntes Modell (Etappe B)

Der Standard-Workflow auf dem Mac:

1. Finetunen mit `mlx-lm` (QLoRA), Adapter mit dem Basismodell **fusen**.
2. Mit `convert_hf_to_gguf.py` aus llama.cpp nach **GGUF** konvertieren
   (Quantisierung z. B. Q4_K_M).
3. **Ollama Modelfile** schreiben (`FROM ./model.gguf`, Template, Parameter)
   und `ollama create` ausfuehren.
4. Chatten via `ollama run`, Open WebUI oder jede OpenAI-kompatible
   Client-Bibliothek (Ollama stellt eine OpenAI-kompatible API bereit).

Alternativ kann `mlx-lm` selbst als Server dienen (`mlx_lm.server`), wenn man
den GGUF-Schritt sparen will.

---

## 6. Was es zu beachten gilt (Stolperfallen)

1. **Klein anfangen, immer.** Jeder Lauf zuerst mit Mini-Konfiguration
   (2 Layer, 1000 Schritte) auf Funktionsfaehigkeit testen, erst dann den
   langen Lauf starten. Die teuerste Fehlerquelle ist ein 12-Stunden-Lauf
   mit einem Bug im Dataloader (das Smol Training Playbook ist voll davon).
2. **MPS-Eigenheiten:** `bfloat16` und `torch.compile` sind auf MPS nicht
   verlaesslich — `float32` nutzen und Geschwindigkeit ueber Batchgroesse/
   Modellgroesse steuern. Vor jedem Lauf `torch.backends.mps.is_available()`
   pruefen; einzelne fehlende MPS-Operationen fallen still auf CPU zurueck
   (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
3. **Checkpointing + Resume von Anfang an einbauen.** Ein MacBook macht
   Sleep, Updates, Akku leer — der Trainingsloop muss jederzeit fortsetzbar
   sein.
4. **Loss-Kurven loggen** (CSV reicht, Weights & Biases ist optional) und
   **regelmaessig Samples generieren**. Ein fallender Loss mit degeneriertem
   Output deutet auf Datenprobleme hin.
5. **Ueberhitzung/Thermal Throttling:** lange Laeufe auf dem MacBook mit
   Netzteil und guter Belueftung; Laeufe ueber Nacht planen.
6. **Tokenizer-Konsistenz:** Modell und Daten muessen exakt denselben
   Tokenizer verwenden; Tokenizer-Dateien zusammen mit Checkpoints ablegen.
7. **Erwartungsmanagement bei SFT des eigenen Modells:** Ein 25M-Modell kann
   einfachen Instruktionen im Stil "Write a story about a dragon" folgen,
   aber keinen echten Dialog fuehren. Das ist normal und kein Fehler.
8. **Lizenz/Compliance:** TinyStories (CDLA-Sharing-1.0), FineWeb (ODC-By),
   Alpaca (CC BY-NC, nicht kommerziell!) — fuer ein privates Lernprojekt
   alles unkritisch, bei Weitergabe von Modellen beachten.

---

## 7. Der Lernpfad in 7 Phasen

Details, Dateien und Akzeptanzkriterien stehen im [AGENT_PLAN.md](AGENT_PLAN.md).
Zeitangaben sind grobe Schaetzungen fuer nebenberufliches Lernen.

| Phase | Inhalt | Theorie-Input | Ergebnis | Zeit |
|---|---|---|---|---|
| 0 | Setup: Python, PyTorch/MPS, Projektstruktur | — | MPS-Check laeuft | 0.5 Tag |
| 1 | Bigram-Modell und Mini-GPT auf Shakespeare (char-level) | Zero-to-Hero-Videos, bes. "Let's build GPT"; microgpt lesen | eigenes GPT erzeugt Shakespeare-artigen Text | 1-2 Wochen (inkl. Videos) |
| 2 | Eigener BPE-Tokenizer auf TinyStories | Tokenizer-Video, minbpe | Tokenizer mit 4k-8k Vokabular, Vergleich mit tiktoken | 2-4 Tage |
| 3 | Pretraining 10-25M-Modell auf TinyStories, danach Architektur-Ablationen (RoPE, RMSNorm, SwiGLU) | Chinchilla-Paper (Abstract+Plots), Raschka-Architekturvergleich | Base-Modell erzaehlt kohaerente Geschichten; Ablations-Tabelle | 1-2 Wochen |
| 4 | Inferenz: Sampling, KV-Cache, CLI, Gradio-Chat-UI | nanochat-Engine als Referenz | Chat-UI im Browser mit eigenem Modell | 3-5 Tage |
| 5 | SFT des eigenen Modells (Story-Instruktionen) | nanochat SFT-Teil, Raschka Kap. 7 | Modell folgt "Write a story about X" | 3-5 Tage |
| 6 | Etappe B: QLoRA-Finetune (mlx-lm) eines 1.7B-4B-Modells, GGUF, Ollama-Hosting | LoRA/QLoRA-Paper, mlx-lm- und Ollama-Doku | eigenes feingetuntes Modell laeuft in Ollama | 1 Woche |

**Empfohlene Reihenfolge des Theorie-Konsums:** Erst Video/Kapitel schauen,
dann den Teil selbst implementieren, dann erst weiter. Passives Schauen ohne
Implementieren bringt erfahrungsgemaess wenig Retention.

---

## 8. Kommentierte Quellenliste

### Lernmaterial (Kern)

- Karpathy, *Neural Networks: Zero to Hero* — https://karpathy.ai/zero-to-hero.html
  — die Videoserie; Minimum: "Let's build GPT" und das Tokenizer-Video.
- Karpathy, *microgpt* (2026) — https://karpathy.github.io/2026/02/12/microgpt/
  — 200 Zeilen pures Python, die "algorithmische Essenz" eines GPT.
- Karpathy, *nanoGPT* — https://github.com/karpathy/nanoGPT
  — Referenz-Implementierung; unser Phase-3-Code orientiert sich hieran.
- Karpathy, *minbpe* — https://github.com/karpathy/minbpe
  — BPE-Tokenizer minimal; Referenz fuer Phase 2.
- Karpathy, *nanochat* — https://github.com/karpathy/nanochat
  — volle Pipeline inkl. SFT und Web-UI; Referenz fuer Phase 4/5.
- Raschka, *Build a Large Language Model (From Scratch)* (Manning 2024) —
  https://github.com/rasbt/LLMs-from-scratch — Begleitbuch mit freiem Code.
- Hugging Face, *The Smol Training Playbook* (2025) —
  https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
  — wie LLM-Training in der Praxis wirklich ablaeuft.

### Papers (in Lesereihenfolge, jeweils nach der passenden Phase)

- Vaswani et al. 2017, *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- Radford et al. 2019, *GPT-2: Language Models are Unsupervised Multitask Learners*
  (OpenAI) — Architektur-Vorlage fuer Phase 3.
- Eldan & Li 2023, *TinyStories* — https://arxiv.org/abs/2305.07759
  — begruendet unsere Datenwahl; sehr lesbar.
- Hoffmann et al. 2022, *Chinchilla / Training Compute-Optimal LLMs* —
  https://arxiv.org/abs/2203.15556 — plus Replikation:
  https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt
- Su et al. 2021, *RoFormer (RoPE)* — https://arxiv.org/abs/2104.09864
- Zhang & Sennrich 2019, *RMSNorm* — https://arxiv.org/abs/1910.07467
- Shazeer 2020, *GLU Variants (SwiGLU)* — https://arxiv.org/abs/2002.05202
- Hu et al. 2021, *LoRA* — https://arxiv.org/abs/2106.09685
- Dettmers et al. 2023, *QLoRA* — https://arxiv.org/abs/2305.14314
- Penedo et al. 2024, *The FineWeb Datasets* — https://arxiv.org/abs/2406.17557

### Architektur und Einordnung

- Raschka, *The Big LLM Architecture Comparison* —
  https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

### Daten

- TinyStories (Dataset) — https://huggingface.co/datasets/roneneldan/TinyStories
- FineWeb-Edu (Dataset) — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Alpaca — https://crfm.stanford.edu/2023/03/13/alpaca.html
- SmolTalk — https://huggingface.co/datasets/HuggingFaceTB/smoltalk

### Apple Silicon, Inferenz und Hosting

- PyTorch MPS Backend — https://pytorch.org/docs/stable/notes/mps.html
- nanoGPT auf Apple Silicon (Erfahrungsbericht M2) —
  https://til.simonwillison.net/llms/nanogpt-shakespeare-m2
- MLX — https://github.com/ml-explore/mlx und mlx-lm —
  https://github.com/ml-explore/mlx-lm
- llama.cpp (GGUF-Konvertierung) — https://github.com/ggml-org/llama.cpp
- Ollama, *Importing a model* — https://docs.ollama.com/import
- Gradio, *Creating a Chatbot Fast* —
  https://gradio.app/guides/creating-a-chatbot-fast
