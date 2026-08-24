# RISE mask-count convergence

`self` = sim(cleanA_k, cleanA_1024) (MC convergence); `repro` = sim(cleanA_k, cleanB_k) (seed agreement). Both rising toward 1.0 with more masks is the expected variance reduction.


### alps_w (q@850)
| n_masks | self_pearson | repro_pearson |
|--:|--:|--:|
| 32 | 0.295 | 0.113 |
| 64 | 0.415 | 0.160 |
| 128 | 0.543 | 0.226 |
| 256 | 0.697 | 0.293 |
| 512 | 0.895 | 0.528 |
| 1024 | 1.000 | 0.671 |

### atl_s (t@500)
| n_masks | self_pearson | repro_pearson |
|--:|--:|--:|
| 32 | 0.297 | 0.013 |
| 64 | 0.439 | 0.053 |
| 128 | 0.571 | 0.213 |
| 256 | 0.701 | 0.346 |
| 512 | 0.865 | 0.530 |
| 1024 | 1.000 | 0.688 |

### trop_s (q@700)
| n_masks | self_pearson | repro_pearson |
|--:|--:|--:|
| 32 | 0.307 | 0.057 |
| 64 | 0.431 | 0.074 |
| 128 | 0.537 | 0.146 |
| 256 | 0.711 | 0.389 |
| 512 | 0.870 | 0.527 |
| 1024 | 1.000 | 0.685 |
