# Architecture

MEA is organized around a four-panel evidence lifecycle DAG of
**Tools**, **Reasoners**, **Validators**, and **Slots**.

## Component Taxonomy

| Type | Role | Examples |
|------|------|----------|
| **Tools** (T) | Stateless external computation | Frame detector, embedder, case aggregator |
| **Reasoners** (R) | LLM calls with typed I/O | Diagnostic reasoner, evidence chain generator |
| **Validators** (V) | Check reasoner outputs | Schema validator, agreement validator, uncertainty gate |
| **Slots** (S) | Named evidence storage | Evidence store (RAG index), feedback store, audit trail |

## Two-Phase Pipeline

```
┌── OFFLINE PHASE ────────────────────────────┐
│                                              │
│  Raw Frames                                  │
│    │                                         │
│    ▼                                         │
│  [Tool] Frame Detector (YOLOv5m)             │
│    │                                         │
│    ▼                                         │
│  [Tool] Case Aggregator (majority vote)      │
│    │                                         │
│    ▼                                         │
│  [Registry] Feature Extraction              │
│    │                                         │
│    ▼                                         │
│  [Reasoner] Evidence Chain Generator (LLM)   │
│    │                                         │
│    ▼                                         │
│  [Slot] Evidence Store (multimodal RAG)      │
│                                              │
└──────────────────────────────────────────────┘
                    │ read
                    ▼
┌── ONLINE PHASE ──────────────────────────────┐
│                                              │
│  New Case                                    │
│    │                                         │
│    ▼                                         │
│  [Tool] Frame Detector                       │
│    │                                         │
│    ▼                                         │
│  [Tool] Case Aggregator                      │
│    │                                         │
│    ▼                                         │
│  [Reasoner] Similar Case Retriever (top-K)   │
│    │                                         │
│    ▼                                         │
│  [Reasoner] Diagnostic Reasoner (LLM)        │
│    │                                         │
│    ▼                                         │
│  [Validator] LLM-Detector Agreement Check    │
│    │                                         │
│    ├─ agree ──▶ Structured Report            │
│    │                                         │
│    └─ disagree ──▶ Expert Review Flag        │
│                                              │
└──────────────────────────────────────────────┘
```

## Code Organization

```
harness/
├── tools/          # Tools (T)
│   └── aggregator.py
├── reasoners/      # Reasoners (R)
│   ├── evidence_chain.py    (DiagnosticReasoner)
│   └── prompts.py           (PromptTemplates)
├── validators/     # Validators (V)
│   ├── gate.py              (AgreementValidator)
│   └── schema.py            (SchemaValidator)
├── slots/          # Slots (S)
│   ├── evidence_store.py    (EvidenceStore)
│   ├── audit_trail.py       (AuditTrail)
│   └── feedback_store.py    (FeedbackStore)
├── registry/       # Feature Registry
│   ├── feature.py
│   └── extractors/
│       └── builtin.py
├── report/         # Clinical report generation
├── evaluation/     # Statistical utilities
├── clinical.py     # ClinicalInteraction
├── llm_client.py   # LLMClient
├── cli.py          # Command-line interface
└── harness.py      # MultimodalEvidenceAlignment main entry
```

## Key Data Structures

- **EvidenceChain** — 8-field typed schema: diagnosis, confidence, reasoning,
  supporting/opposing evidence, differential, sufficiency, uncertainty sources
- **CaseEmbedding** — 5120-dimensional fused vector (vision 4096 + text 1024)
- **GateDecision** — Pass/fail with confidence adjustment and warnings

## Clinical Reasoning Protocol

The LLM reasoner is constrained by a 4-layer protocol (encoded in
`harness/reasoners/prompts.py`):

1. **Disease Priors** — Prevalence-aware class distributions
2. **Confusion Patterns** — Known detector ambiguities from training
3. **Evidence Weighting** — Consensus ratio as primary signal
4. **Selective Diagnosis** — Agreement thresholds: ≥80% diagnose, <60% uncertain

## See Also

- Paper: `paper/ceh_paper.pdf` — Full method description
- `harness/reasoners/prompts.py` — Clinical Reasoning Protocol implementation
- `harness/slots/audit_trail.py` — EvidenceChain dataclass
