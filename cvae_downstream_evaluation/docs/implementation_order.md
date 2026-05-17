# Implementation Order

Use this order to keep the next code changes small and auditable.

## 1. Contracts And Manifests

Implement:

- config loader
- protocol manifest writer
- split manifest writer
- expert provenance loader
- strict Camelyon17 v1 config validation

Done when:

- target support and target evaluation rows are disjoint
- held-out target expert is excluded from candidate experts
- forbidden target evaluation fields are absent from routing inputs
- stale TODOs, conditional-generation wording, and non-Camelyon17 v1 scope are rejected

## 2. Selection Bridge

Implement:

- support-NELBO score loading from existing support-routing artifacts
- direct support-NELBO argmin selection
- metadata, random, and ensemble baseline selection descriptors
- deterministic random baseline construction when absent from support artifacts

Done when:

- selections can be reproduced from the manifest alone
- routing decisions are made before synthetic generation
- downstream candidate scores are computed once per candidate expert and not duplicated by support seed/size

## 3. Synthetic Embedding Generation

Implement:

- class-balanced label schedule
- decoder sampling wrapper
- generation seed handling
- generated embedding manifest
- matching projection-frame contract for expert-specific heads

Done when:

- every candidate expert can generate the same locked synthetic budget
- selected-expert generation and all-expert oracle diagnostics share the same sampling contract
- naive all-expert ensemble uses late probability averaging rather than mixed-frame concatenation

## 4. Synthetic-Only Downstream Utility

Implement:

- small downstream classifier
- fixed classifier hyperparameters
- all-expert downstream matrix
- target evaluation metrics

Done when:

- every candidate expert has a comparable downstream score
- downstream oracle is computed only after all candidate scores exist

## 5. Fidelity Diagnostics

Implement:

- MMD
- energy distance
- Frechet embedding distance
- mean/covariance distance
- kNN precision/recall/density/coverage if practical

Done when:

- fidelity metrics are reported separately from downstream utility
- correlation with downstream utility is computed as diagnostic evidence

## 6. Reporting

Implement:

- routing-to-downstream alignment table
- downstream performance table
- fidelity diagnostics table
- support-size stratified summary table
- stability table
- leakage/provenance report
- decision summary

Done when:

- the report can classify the result as PASS, WEAK PASS, DIAGNOSTIC ONLY, or FAIL
- the summary states allowed and forbidden thesis claims
- PASS gates use only the primary generation mode and budget 128; support-size stratification is descriptive only
