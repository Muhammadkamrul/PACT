# INTACT W3R-V report — CONFIRMATION

Seeds: [5101, 5102, 5103, 5104, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120] (n=20). C4 is removed everywhere.

## 1. Integrity

- [PASS] registered methods complete for every seed
- [PASS] W3R-V records zero C1 and C2 violations

## 2. Method means

| Method | WIF | tenant-weighted | C3 shortfall | tenant inversion | writes/epoch |
|---|---:|---:|---:|---:|---:|
| W3R-V-lin | 0.9260 | 0.9179 | 0.0172 | 0.122 | 1.91 |
| W3R-V-noalpha | 0.9258 | 0.9187 | 0.0168 | 0.120 | 2.11 |
| W3R-V | 0.9256 | 0.9186 | 0.0168 | 0.120 | 2.10 |
| W3R-V-nogamma | 0.9255 | 0.9184 | 0.0168 | 0.121 | 2.11 |
| W3R-V-noomega | 0.9254 | 0.9181 | 0.0167 | 0.121 | 2.13 |
| W3R-V-nopi | 0.9252 | 0.9179 | 0.0167 | 0.121 | 2.15 |
| W3R-V-noC3 | 0.9244 | 0.9182 | 0.0193 | 0.121 | 2.07 |
| W3R-V-nourgency | 0.9238 | 0.9180 | 0.0196 | 0.120 | 2.04 |
| QACM-style | 0.9228 | 0.9176 | 0.0155 | 0.120 | 4.44 |
| QACM-contract | 0.9201 | 0.9172 | 0.0158 | 0.119 | 4.48 |
| B3-contract | 0.9175 | 0.9111 | 0.0181 | 0.132 | 0.18 |
| B3-contract-raw | 0.9171 | 0.9116 | 0.0180 | 0.131 | 0.19 |
| W3R | 0.9139 | 0.9073 | 0.0251 | 0.138 | 0.68 |
| W | 0.9138 | 0.9060 | 0.0237 | 0.141 | 0.76 |
| W3R-V-noMILD | 0.9095 | 0.9122 | 0.0308 | 0.121 | 1.91 |
| B3-contract-reactive | 0.8960 | 0.8983 | 0.0463 | 0.137 | 0.19 |
| inner-contract | 0.8885 | 0.8714 | 0.0272 | 0.204 | 2.50 |
| W3R-V-scalar | 0.8749 | 0.8867 | 0.0828 | 0.149 | 1.72 |
| W3R-V-nobeta | 0.8735 | 0.8869 | 0.0978 | 0.152 | 0.14 |
| B3 | 0.8608 | 0.8888 | 0.1356 | 0.142 | 0.11 |
| all-reject | 0.8555 | 0.8815 | 0.1402 | 0.150 | 0.00 |

## 3. Registered primary comparisons (WIF)

| Comparator | W3R-V minus (pp) | 95% CI | wins/losses | Holm p | verdict |
|---|---:|---:|---:|---:|---|
| QACM-contract | +0.55 | [-0.11, +1.14] | 16/4 | 0.1092 | not resolved (no superiority, no equivalence claimed) |
| B3-contract | +0.81 | [+0.44, +1.21] | 17/3 | 0.0007 | W3R-V significantly better |
| B3-contract-raw | +0.85 | [+0.53, +1.21] | 18/2 | 0.0000 | W3R-V significantly better |
| inner-contract | +3.71 | [+2.74, +4.61] | 19/1 | 0.0000 | W3R-V significantly better |

**Automatic verdict:** W3R-V is significantly better than B3-contract, B3-contract-raw, inner-contract; significantly worse than none; others unresolved. Superiority over all primary comparators is NOT established.

## 4. Component ablations (W3R-V minus variant; positive = component helps)

| Comparator | W3R-V minus (pp) | 95% CI | wins/losses | Holm p | verdict |
|---|---:|---:|---:|---:|---|
| W3R-V-noalpha | -0.02 | [-0.05, +0.01] | 3/7 | 1.0000 | not resolved (no superiority, no equivalence claimed) |
| W3R-V-nobeta | +5.21 | [+3.52, +6.86] | 17/3 | 0.0002 | W3R-V significantly better |
| W3R-V-nogamma | +0.01 | [-0.03, +0.06] | 6/3 | 1.0000 | not resolved (no superiority, no equivalence claimed) |
| W3R-V-scalar | +5.07 | [+4.00, +6.19] | 20/0 | 0.0001 | W3R-V significantly better |
| W3R-V-noMILD | +1.61 | [+1.24, +1.99] | 20/0 | 0.0001 | W3R-V significantly better |
| W3R-V-nourgency | +0.18 | [+0.01, +0.44] | 10/3 | 0.4352 | not resolved (no superiority, no equivalence claimed) |
| W3R-V-nopi | +0.04 | [-0.02, +0.10] | 8/5 | 1.0000 | not resolved (no superiority, no equivalence claimed) |
| W3R-V-noomega | +0.02 | [-0.04, +0.09] | 5/7 | 1.0000 | not resolved (no superiority, no equivalence claimed) |
| W3R-V-noC3 | +0.12 | [-0.08, +0.48] | 3/5 | 1.0000 | not resolved (no superiority, no equivalence claimed) |
| W3R-V-lin | -0.04 | [-0.21, +0.14] | 7/9 | 1.0000 | not resolved (no superiority, no equivalence claimed) |

Components with a significant WIF contribution: W3R-V-nobeta, W3R-V-scalar, W3R-V-noMILD. Unresolved (do not claim): W3R-V-noalpha, W3R-V-nogamma, W3R-V-nourgency, W3R-V-nopi, W3R-V-noomega, W3R-V-noC3, W3R-V-lin.

Figures: runs_w3v_confirmation/report_w3/figures.
